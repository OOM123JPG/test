import os
import sys
import time
from datetime import timedelta

# 1. 强行修正环境变量，确保 local_rank 绝不是 -1
rank = int(os.environ.get("RANK", 0))
l_rank = os.environ.get("LOCAL_RANK")
if l_rank is None or int(l_rank) == -1:
    l_rank = rank % 8
else:
    l_rank = int(l_rank)
os.environ["LOCAL_RANK"] = str(l_rank)

# 使用一个新的 MASTER_PORT
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '33445'
os.environ['HCCL_WHITELIST_DISABLE'] = '1'

import torch
import torch_npu
import torch.distributed as dist

# 2. 必须在任何分布式调用前绑定设备
torch.npu.set_device(l_rank)
print(f"[RANK {rank}] 到达初始化点，Device ID: {l_rank}")

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
    
    # 3. 步骤一：分布式握手
    if not dist.is_initialized():
        print(f"[RANK {rank}] 正在执行 init_distributed_environment (端口 33445)...")
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:33445",
            backend="hccl",
            timeout=timedelta(seconds=120)
        )
    
    # 4. 步骤二：建立 TP 组
    if not ensure_model_parallel_initialized(4, 1):
        print(f"[RANK {rank}] 正在建立并行组 (TP4)...")
        initialize_model_parallel(4, 1)
    
    print(f"[RANK {rank}] 分布式握手全部成功！准备进入模型加载。")

    # 5. 配置与加载
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

    # 这里的实例化会触发 deepseek_v2.py 里的读取打印
    moe_layer = DeepseekV2MoE(
        config=config,
        parallel_config=MockParallelConfig(),
        prefix="model.layers.16.mlp",
        layer_idx=16
    ).to("npu").to(torch.bfloat16)

    print(f"[RANK {rank}] 推理开始...")
    dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
    with torch.no_grad():
        output = moe_layer(dummy_input)

    if rank == 0:
        print(f"\n[SUCCESS] 输出均值: {output.abs().mean().item():.8f}")

if __name__ == "__main__":
    run_test()
