import os
import sys
import gc
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. 核心 Tucker 专家算子定义
# ==========================================
class TuckerExpert(nn.Module):
    def __init__(self, core, factors, proj_type="gate_proj"):
        super().__init__()
        # 针对 BF16 优化
        self.u_in = nn.Parameter(factors[1].to(torch.bfloat16).contiguous())   # [In_Dim, Rank_In]
        self.core = nn.Parameter(core.to(torch.bfloat16).contiguous())         # [E_group, Rank_In, Rank_Out]
        self.u_out = nn.Parameter(factors[2].to(torch.bfloat16).contiguous())  # [Rank_Out, Out_Dim]
        self.proj_type = proj_type

    def forward(self, x, expert_indices_in_group):
        """
        x: [tokens_in_group, In_Dim]
        expert_indices_in_group: [tokens_in_group] 路由到该组内的相对索引
        """
        if x.shape[0] == 0:
            return x

        # 阶段 1: 输入投影
        # [tokens, In_Dim] @ [In_Dim, Rank_In] -> [tokens, Rank_In]
        x = torch.matmul(x, self.u_in)

        # 阶段 2: Core Tensor 计算 (Batch Matrix Multiplication)
        # 获取每个 token 对应的 Core 切片
        # core_slices: [tokens, Rank_In, Rank_Out]
        core_slices = self.core[expert_indices_in_group]
        
        # [tokens, 1, Rank_In] @ [tokens, Rank_In, Rank_Out] -> [tokens, 1, Rank_Out]
        x = torch.bmm(x.unsqueeze(1), core_slices).squeeze(1)

        # 阶段 3: 输出投影
        # [tokens, Rank_Out] @ [Rank_Out, Out_Dim] -> [tokens, Out_Dim]
        x = torch.matmul(x, self.u_out)
        return x

# ==========================================
# 2. 动态替换 DeepSeek MoE 层
# ==========================================
class TuckerMoELayer(nn.Module):
    def __init__(self, original_moe, layer_idx, decomp_dir, n_group=8, experts_per_group=32):
        super().__init__()
        self.gate = original_moe.gate
        self.layer_idx = layer_idx
        self.n_group = n_group
        self.experts_per_group = experts_per_group
        
        # 存储 8 个组的 Tucker 组件
        self.groups = nn.ModuleDict()
        
        for g_idx in range(n_group):
            file_path = os.path.join(decomp_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing decomposition file: {file_path}")
            
            group_weights = torch.load(file_path, map_location="cpu")
            
            # 为每一组创建三个投影的 Tucker 组合
            group_module = nn.ModuleDict({
                "gate_proj": TuckerExpert(group_weights["gate_proj"]["core"], group_weights["gate_proj"]["factors"]),
                "up_proj":   TuckerExpert(group_weights["up_proj"]["core"],   group_weights["up_proj"]["factors"]),
                "down_proj": TuckerExpert(group_weights["down_proj"]["core"], group_weights["down_proj"]["factors"])
            })
            self.groups[f"group_{g_idx}"] = group_module

    def forward(self, hidden_states):
        # 1. Router 获取权重和索引
        router_logits = self.gate(hidden_states)
        # 模拟 DeepSeek-V3 的 Top-K 路由 (简化逻辑，实际需参考原模型代码)
        weights, selected_experts = torch.topk(router_logits, k=8, dim=-1)
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)
        
        # 2. 分组处理
        for g_idx in range(self.n_group):
            group_mask = (selected_experts // self.experts_per_group) == g_idx
            if not group_mask.any():
                continue
            
            # 提取属于该组的 tokens 和对应的专家内索引
            # 实际实现中需处理 Flattened tokens 以加速
            group_experts = self.groups[f"group_{g_idx}"]
            
            # --- Tucker 推理核心逻辑 ---
            # 这里简化为逐 token 处理，生产环境建议使用并行 Mask 索引
            # ... 具体的算子加速逻辑 ...
            
        return final_hidden_states

# ==========================================
# 3. 分布式加载与重建
# ==========================================
def setup_distributed():
    if not dist.is_initialized():
        # 针对昇腾 NPU 的分布式初始化
        dist.init_process_group(backend="hccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    return local_rank

def reconstruct_and_run():
    # 参数设置
    model_path = "/home/models/DeepSeek-V3-bf16"
    decomp_dir = "./output/decomp_results"
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    
    # PP=2 策略：前一半层给机器0，后一半层给机器1
    node_rank = int(os.environ.get("NODE_RANK", 0)) # 0 或 1
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    total_layers = config.num_hidden_layers
    layers_per_node = total_layers // 2
    
    start_layer = node_rank * layers_per_node
    end_layer = (node_rank + 1) * layers_per_node if node_rank == 0 else total_layers

    logging.info(f"Node {node_rank} loading layers {start_layer} to {end_layer}")

    # 1. 加载空模型结构 (为了省显存)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # 2. 逐层替换并搬运到 NPU
    for i in range(start_layer, end_layer):
        if i < 3 or i > 60: # 非 MoE 层
            # 加载原始权重逻辑
            continue
        
        # 替换为 Tucker 层
        # 注意：这里需要先从磁盘加载原 gate 权重
        orig_moe = model.model.layers[i].mlp 
        model.model.layers[i].mlp = TuckerMoELayer(orig_moe, i, decomp_dir).to("npu").to(torch.bfloat16)
        
        logging.info(f"Layer {i} reconstructed on NPU:{local_rank}")
        gc.collect()
        torch.npu.empty_cache()

    # 3. 这里的推理逻辑需要对接 MindIE 或自定义的 PP 通信
    # if node_rank == 0: 
    #     recv_input_and_send_to_node1()
    # else:
    #     recv_from_node0_and_output()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reconstruct_and_run()
