import sys
from unittest.mock import MagicMock

# 模拟 flash_attn 模块，防止模型加载时报错
mock_flash_attn = MagicMock()
sys.modules["flash_attn"] = mock_flash_attn
sys.modules["flash_attn.flash_attn_interface"] = mock_flash_attn

# 环境初始化
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
curr_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(curr_dir)
sys.path.append(proj_dir)

import os
import sys
import gc
import logging
import argparse
import torch
import torch_npu
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM



# ==============================================================================
# 1. 定义 Tucker 专家层
# ==============================================================================

class TuckerDecomposedExpert(nn.Module):
    def __init__(self, proj_name, core, factors, device="cpu"):
        super().__init__()
        self.proj_name = proj_name
        # 强制 bf16 确保性能，并移动到对应设备
        self.u_in = nn.Parameter(factors[1].to(torch.bfloat16).to(device))
        self.core = nn.Parameter(core.to(torch.bfloat16).to(device))
        self.u_out = nn.Parameter(factors[2].to(torch.bfloat16).to(device))

    def forward(self, x, expert_idx_in_group):
        if x.shape[0] == 0:
            return x
        # 1. 输入投影
        x = torch.matmul(x, self.u_in)
        # 2. 核心交互 (Batch MatMul)
        core_slices = self.core[expert_idx_in_group]
        x = torch.bmm(x.unsqueeze(1), core_slices).squeeze(1)
        # 3. 输出投影
        x = torch.matmul(x, self.u_out)
        return x

# ==============================================================================
# 2. 动态替换与局部层装载逻辑 (针对 PP2 优化)
# ==============================================================================

def apply_tucker_to_model_partial(model, decomp_dir, active_layers, device="cpu"):
    """
    只处理当前 Rank 负责的层范围，节省内存
    """
    n_group = 8
    experts_per_group = 32

    for layer_idx in active_layers:
        layer = model.model.layers[layer_idx]
        if not hasattr(layer.mlp, 'experts'): continue
        
        # 检查该层文件
        check_file = os.path.join(decomp_dir, f"layer_{layer_idx}_group_0.pt")
        if not os.path.exists(check_file):
            continue

        logging.info(f"Applying Tucker to Layer {layer_idx}...")
        for g_idx in range(n_group):
            decomp_path = os.path.join(decomp_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
            group_weights = torch.load(decomp_path, map_location="cpu")
            
            for exp_in_g in range(experts_per_group):
                abs_exp_idx = g_idx * experts_per_group + exp_in_g
                expert = layer.mlp.experts[abs_exp_idx]
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    data = group_weights[proj_name]
                    tucker_mod = TuckerDecomposedExpert(proj_name, data['core'], data['factors'], device)
                    setattr(expert, proj_name, tucker_mod)
            del group_weights
            gc.collect()
    return model

# ==============================================================================
# 3. 主函数 (PP=2 核心逻辑)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--decomp_dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 初始化分布式
    if args.local_rank != -1:
        torch.npu.set_device(args.local_rank)
        dist.init_process_group(backend="hccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
        torch.npu.set_device(0)

    # 1. 确定流水线切分 (PP=2)
    # 机器0 (Rank 0-7) 处理 0-30 层; 机器1 (Rank 8-15) 处理 31-61 层
    total_layers = 62
    mid = 31
    if rank < 8:
        active_range = range(0, mid) 
        neighbor_rank = rank + 8  # 机器0的卡对应发给机器1的卡
        is_first_stage = True
    else:
        active_range = range(mid, total_layers)
        neighbor_rank = rank - 8
        is_first_stage = False

    logging.info(f"Rank {rank}: Handling layers {list(active_range)}")

    # 2. 使用 Meta Device 创建模型框架 (不占显存)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16)

    # 3. 执行局部层重构
    # 将模型搬运到 NPU 之前，只针对 active_range 进行权重替换
    model = apply_tucker_to_model_partial(model, args.decomp_dir, active_range, device="cpu")

    # 将负责的层移动到 NPU
    for i in active_range:
        model.model.layers[i].to(torch.npu.current_device())
    
    # Embedding 和 Norm 需要根据阶段归属
    if is_first_stage:
        model.model.embed_tokens.to(torch.npu.current_device())
    else:
        model.model.norm.to(torch.npu.current_device())
        model.lm_head.to(torch.npu.current_device())

    logging.info(f"Rank {rank} Partial Model Loaded.")

    # 4. 推理演示 (单步 Handshake)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 简单的流水线执行流程
    with torch.no_grad():
        if is_first_stage:
            # 机器0：执行
            text = "Explain the Tucker decomposition."
            inputs = tokenizer(text, return_tensors="pt").to(torch.npu.current_device())
            hidden_states = model.model.embed_tokens(inputs['input_ids'])
            
            # 跑前一半层
            for i in active_range:
                hidden_states = model.model.layers[i](hidden_states)[0]
            
            # 发送给机器1
            # 注意：DeepSeek-V3 hidden_size=7168
            if rank == 0: # 仅由主卡发送，或各卡发送自己的 TP 部分
                dist.send(tensor=hidden_states.float().cpu(), dst=8)
                logging.info("Rank 0: Handover data sent to Rank 8.")
        
        else:
            # 机器1：接收
            if rank == 8:
                # 预分配接收缓存 (根据输入 SeqLen 调整)
                recv_buf = torch.empty([1, 6, 7168], dtype=torch.float32) 
                dist.recv(tensor=recv_buf, src=0)
                hidden_states = recv_buf.to(torch.bfloat16).to(torch.npu.current_device())
                logging.info("Rank 8: Data received from Rank 0.")
            else:
                # 其它卡等待同步 (如果做了 TP 需要广播，这里简化处理)
                hidden_states = torch.zeros([1, 6, 7168], dtype=torch.bfloat16).to(torch.npu.current_device())

            # 跑后一半层
            for i in active_range:
                hidden_states = model.model.layers[i](hidden_states)[0]
            
            # 输出结果
            if rank == 8:
                logits = model.lm_head(model.model.norm(hidden_states))
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                print(f"Rank 8 Prediction Token ID: {next_token.item()}")
                print(f"Decoded: {tokenizer.decode(next_token)}")

if __name__ == "__main__":
    main()
