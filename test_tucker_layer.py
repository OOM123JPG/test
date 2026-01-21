import os
import sys
import time

# 必须在 import torch 之前，避免环境变量覆盖失败
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = 'error' 
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29506' # 更换一个不常用的端口

import torch
import torch_npu
import torch.distributed as dist

# 核心：精准获取 local_rank
def get_rank_info():
    rank = int(os.environ.get("RANK", 0))
    l_rank = os.environ.get("LOCAL_RANK")
    if l_rank is None or int(l_rank) == -1:
        l_rank = rank % 8
    else:
        l_rank = int(l_rank)
    return rank, l_rank

rank, l_rank = get_rank_info()
torch.npu.set_device(l_rank) # 必须在 init 之前绑定

import torch.nn as nn
import torch.nn.functional as F
import torch.library

if not hasattr(torch.library, "infer_schema"):
    torch.library.infer_schema = lambda *args, **kwargs: None

sys.path.append("/vllm-workspace/vllm")

from vllm.distributed import (
    initialize_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from transformers import DeepseekV2Config

def run_test():
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 打印每个 Rank 的状态，确保 4 个都在跑
    print(f"[RANK {rank}] 到达初始化点。LocalRank: {l_rank}, WorldSize: {world_size}")

    # 1. 尝试初始化
    try:
        if not dist.is_initialized():
            # 这里是死锁高发区
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method="tcp://127.0.0.1:29506",
                backend="hccl"
            )
            print(f"[RANK {rank}] HCCL 组握手成功！")

        if not ensure_model_parallel_initialized(4, 1):
            initialize_model_parallel(4, 1)
            print(f"[RANK {rank}] TP4 并行状态已建立。")
    except Exception as e:
        print(f"[RANK {rank}] 初始化崩溃: {e}")
        return

    # 2. 配置 (按你的要求：V3 规格 + NFS 路径)
    config = DeepseekV2Config()
    config.hidden_size = 7168
    config.moe_intermediate_size = 2048
    config.n_routed_experts = 256
    config.num_experts_per_tok = 8
    config.use_tucker = True
    config.tucker_layers = [16]
    config.tucker_path = "/nfs-share/wx1463835/tdmoe/output/decomp_results"
    
    class MockParallelConfig:
        def __init__(self):
            self.use_sequence_parallel_moe = False
            self.enable_eplb = False
            self.eplb_config = type('obj', (object,), {'num_redundant_experts': 0})()

    # 3. 加载模型
    try:
        print(f"[RANK {rank}] 开始加载 Tucker 专家权重...")
        # 确保 layer_idx 传入，触发 deepseek_v2.py 里的加载逻辑
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        # 4. 同步并推理
        dist.barrier()
        if rank == 0: print(">>> 所有 Rank 加载完毕，开始执行推理...")
        
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n[SUCCESS] 推理成功！")
            print(f"Output Mean: {output.abs().mean().item():.8f}")
            
    except Exception as e:
        print(f"[RANK {rank}] 运行报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
