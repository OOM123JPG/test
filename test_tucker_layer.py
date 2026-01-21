import os
import sys
import time

# --- 1. 环境变量诊断日志 ---
def diagnose_env():
    rank = os.environ.get("RANK", "N/A")
    l_rank = os.environ.get("LOCAL_RANK", "N/A")
    world_size = os.environ.get("WORLD_SIZE", "N/A")
    master_addr = os.environ.get("MASTER_ADDR", "N/A")
    master_port = os.environ.get("MASTER_PORT", "N/A")
    print(f"[DIAGNOSE] Global Rank: {rank}, Local Rank: {l_rank}, World Size: {world_size}")
    print(f"[DIAGNOSE] Master: {master_addr}:{master_port}")

diagnose_env()

# 设置基础环境变量
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = 'error' 
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'

import torch
import torch_npu
import torch.distributed as dist

# 获取并校验 local_rank
def get_safe_local_rank():
    l_rank = os.environ.get("LOCAL_RANK")
    if l_rank is None or int(l_rank) < 0:
        # 这里的推算仅适用于单机环境
        return int(os.environ.get("RANK", 0)) % 8
    return int(l_rank)

l_rank = get_safe_local_rank()

import torch.nn as nn
import torch.nn.functional as F

# 补丁
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
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # --- 2. NPU 设备绑定诊断 ---
    try:
        device_count = torch.npu.device_count()
        print(f"[RANK {rank}] 系统检测到 NPU 数量: {device_count}")
        print(f"[RANK {rank}] 尝试绑定 Device ID: {l_rank}")
        
        # 强制设置设备
        torch.npu.set_device(l_rank)
        current_dev = torch.npu.current_device()
        print(f"[RANK {rank}] 成功绑定到 NPU:{current_dev}")
    except Exception as e:
        print(f"[RANK {rank}] 设备绑定阶段崩溃! 错误: {e}")
        return

    # 3. 分布式初始化
    try:
        if not dist.is_initialized():
            print(f"[RANK {rank}] 正在执行 init_distributed_environment...")
            init_distributed_environment(
                world_size=world_size,
                rank=rank,
                distributed_init_method="tcp://127.0.0.1:29505",
                backend="hccl"
            )
            print(f"[RANK {rank}] HCCL 环境基础初始化完成")
        
        if not ensure_model_parallel_initialized(4, 1):
            print(f"[RANK {rank}] 正在执行 initialize_model_parallel (TP4)...")
            initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
            print(f"[RANK {rank}] TP4 并行组建立成功")
    except Exception as e:
        print(f"[RANK {rank}] 分布式握手阶段失败! 错误: {e}")
        return

    # 4. 模型加载与推理 (规格与路径按要求设置)
    config = DeepseekV2Config()
    config.hidden_size = 7168
    config.moe_intermediate_size = 2048
    config.n_routed_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 1
    config.norm_topk_prob = True
    config.scoring_func = "softmax"
    
    config.use_tucker = True
    config.tucker_layers = [16]
    config.tucker_path = "/nfs-share/wx1463835/tdmoe/output/decomp_results"
    
    class MockParallelConfig:
        def __init__(self):
            self.use_sequence_parallel_moe = False
            self.enable_eplb = False
            self.eplb_config = type('obj', (object,), {'num_redundant_experts': 0})()

    try:
        print(f"[RANK {rank}] 准备实例化 DeepseekV2MoE (Layer 16)...")
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        print(f"[RANK {rank}] 权重加载完成，显存占用检查点")
        dist.barrier()

        # 生成输入测试
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n>>> [SUCCESS] 推理成功! 均值: {output.abs().mean().item():.8f}")
    except Exception as e:
        print(f"[RANK {rank}] 模型运行阶段报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
