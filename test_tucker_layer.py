import os
import sys

# 必须在 import torch 之前
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = 'error' 
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'

# 解决 Device ID 报错的核心：强制解析 local_rank
l_rank = os.environ.get("LOCAL_RANK")
if l_rank is None or int(l_rank) < 0:
    # 兜底方案：如果 torchrun 没传，尝试从 RANK 推算或默认为 0
    l_rank = int(os.environ.get("RANK", 0)) % 8
else:
    l_rank = int(l_rank)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
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
    
    # 1. 严格绑定 NPU
    torch.npu.set_device(l_rank)
    
    # 2. 初始化分布式
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:29505",
            backend="hccl"
        )
    
    # 3. 初始化 TP4
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(f">>> [TP4] 环境就绪，Device ID 检查通过。")

    # 4. 配置 (DeepSeek-V3 规格)
    config = DeepseekV2Config()
    config.hidden_size = 7168
    config.moe_intermediate_size = 2048
    config.n_routed_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 1
    config.norm_topk_prob = True
    config.scoring_func = "softmax"
    config.hidden_act = "silu"
    
    config.use_tucker = True
    config.tucker_layers = [16]
    config.tucker_path = "/nfs-share/wx1463835/tdmoe/output/decomp_results"
    
    class MockParallelConfig:
        def __init__(self):
            self.use_sequence_parallel_moe = False
            self.enable_eplb = False
            self.eplb_config = type('obj', (object,), {'num_redundant_experts': 0})()

    # 5. 加载模型
    try:
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        dist.barrier() # 等待所有卡加载完毕

        # 6. 推理
        torch.manual_seed(42 + rank) # 保证输入一致性
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n>>> [SUCCESS] TP4 推理成功！")
            print(f">>> Output Shape: {output.shape}")
            print(f">>> Output Mean: {output.abs().mean().item():.8f}")

    except Exception as e:
        print(f"Rank {rank} Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
