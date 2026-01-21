import os
import sys
import gc
import logging
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

# 环境初始化
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
curr_dir = os.path.dirname(os.path.abspath(__file__))
proj_dir = os.path.dirname(curr_dir)
sys.path.append(proj_dir)

# ==============================================================================
# 1. 定义 Tucker 专家层 (取代原生的 DeepseekV3Expert)
# ==============================================================================

class TuckerDecomposedExpert(nn.Module):
    """
    使用 Tucker 分解组件重构的专家层
    实现公式: y = ((x @ U_in) @ Core_i) @ U_out
    """
    def __init__(self, proj_name, core, factors, device="cpu"):
        super().__init__()
        self.proj_name = proj_name
        
        # factors[0]: 专家选择因子 (通常在重建时已经包含在 Core 中)
        # factors[1]: 输入投影矩阵 [In_Dim, Rank_In]
        # factors[2]: 输出投影矩阵 [Rank_Out, Out_Dim]
        # core: [Group_Size, Rank_In, Rank_Out]
        
        self.u_in = nn.Parameter(factors[1].to(torch.bfloat16).to(device))
        self.core = nn.Parameter(core.to(torch.bfloat16).to(device))
        self.u_out = nn.Parameter(factors[2].to(torch.bfloat16).to(device))

    def forward(self, x, expert_idx_in_group):
        """
        x: [tokens_in_group, In_Dim]
        expert_idx_in_group: [tokens_in_group] 该组内 token 对应的专家索引
        """
        if x.shape[0] == 0:
            return x
            
        # 1. 输入投影 (Input Projection)
        # [tokens, In_Dim] @ [In_Dim, Rank_In] -> [tokens, Rank_In]
        x = torch.matmul(x, self.u_in)
        
        # 2. 核心张量交互 (Core Interaction)
        # 根据每个 token 的专家索引，从 Core 中提取对应的 [Rank_In, Rank_Out] 矩阵
        # core_slices: [tokens, Rank_In, Rank_Out]
        core_slices = self.core[expert_idx_in_group]
        
        # 使用批量矩阵乘法 (Batch MatMul)
        # [tokens, 1, Rank_In] @ [tokens, Rank_In, Rank_Out] -> [tokens, 1, Rank_Out]
        x = torch.bmm(x.unsqueeze(1), core_slices).squeeze(1)
        
        # 3. 输出投影 (Output Projection)
        # [tokens, Rank_Out] @ [Rank_Out, Out_Dim] -> [tokens, Out_Dim]
        x = torch.matmul(x, self.u_out)
        
        return x

# ==============================================================================
# 2. 动态替换与权重装载逻辑
# ==============================================================================

def apply_tucker_to_model(model, decomp_dir, device="cpu"):
    """
    遍历模型，将指定的 MoE 层替换为 Tucker 重构层
    """
    n_group = 8
    experts_per_group = 32 # DeepSeek-V3: 256 / 8 = 32

    for layer_idx, layer in enumerate(model.model.layers):
        # 检查该层是否有分解文件 (例如 layer_16)
        check_file = os.path.join(decomp_dir, f"layer_{layer_idx}_group_0.pt")
        if not os.path.exists(check_file):
            continue

        logging.info(f"Replacing Layer {layer_idx} with Tucker layers...")
        
        # 遍历该层的所有专家
        for g_idx in range(n_group):
            decomp_path = os.path.join(decomp_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
            group_weights = torch.load(decomp_path, map_location="cpu")
            
            for exp_in_g in range(experts_per_group):
                abs_exp_idx = g_idx * experts_per_group + exp_in_g
                expert = layer.mlp.experts[abs_exp_idx]
                
                # 替换 gate_proj, up_proj, down_proj
                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                    data = group_weights[proj_name]
                    # 创建 Tucker 重构层
                    tucker_module = TuckerDecomposedExpert(
                        proj_name=proj_name,
                        core=data['core'],
                        factors=data['factors'],
                        device=device
                    )
                    # 替换原有的 Linear 层
                    setattr(expert, proj_name, tucker_module)
            
            del group_weights
            gc.collect()

    return model

# ==============================================================================
# 3. 分布式推理启动框架
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--decomp_dir", type=str, required=True, help="Tucker分解组件路径")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # 1. 分布式环境初始化 (解决双机死锁)
    if args.local_rank != -1:
        torch.npu.set_device(args.local_rank)
        dist.init_process_group(backend="hccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = "npu:0" if torch.npu.is_available() else "cpu"

    logging.info(f"Rank {rank}/{world_size} loading configuration...")

    # 2. 加载基础模型 (仅加载 Config 和非专家层的权重，避免 OOM)
    # 在双机 PP2 模式下，不同 Rank 加载不同的层
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 注意：为了节省内存，可以先在 CPU 加载空模型，替换完后再 .to(npu)
    with torch.device("cpu"):
        # 实际部署时建议使用从 transformers 导入的 DeepseekV3ForCausalLM
        # 这里为了演示通用性使用 AutoModel
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16)

    # 3. 执行 Tucker 重构
    model = apply_tucker_to_model(model, args.decomp_dir, device="cpu")

    # 4. (可选) 流水线并行切分逻辑
    # 如果是双机，Rank 0 屏蔽后半部分层，Rank 1 屏蔽前半部分层
    # 这里需要根据你的分布式框架（如 DeepSpeed 或 MindIE-Torch）进行对接
    
    model.to(torch.npu.current_device())
    logging.info(f"Rank {rank} Model Reconstruction Complete.")

    # 5. 推理测试 (示例)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    inputs = tokenizer("Hello, DeepSeek!", return_tensors="pt").to(torch.npu.current_device())
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
        if rank == 0:
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
