import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu

# 补丁：解决 infer_schema
import torch.library
if not hasattr(torch.library, "infer_schema"):
    torch.library.infer_schema = lambda *args, **kwargs: None

# 环境变量
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'
sys.path.append("/vllm-workspace/vllm")

from vllm.distributed import (
    initialize_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from transformers import DeepseekV2Config

def test_tp4_tucker():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))

    # 1. 初始化 vLLM 底层环境 (解决 world group 报错的核心)
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:29505",
            backend="hccl"
        )
    
    torch.npu.set_device(local_rank)

    # 2. 初始化并行组 (TP=4)
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(f"\n[TEST] TP4 环境初始化成功，开始加载层...")

    # 3. 构建配置
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

    # 4. 实例化层 (此处会调用修改后的 __init__)
    moe_layer = DeepseekV2MoE(
        config=config,
        parallel_config=MockParallelConfig(),
        prefix="model.layers.16.mlp",
        layer_idx=16
    ).to("npu").to(torch.bfloat16)

    # 5. 推理测试
    dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
    with torch.no_grad():
        output = moe_layer(dummy_input)

    if rank == 0:
        print(f"\n[SUCCESS] 推理完成！输出形状: {output.shape}")
        print(f"输出均值: {output.abs().mean().item():.6f}")

if __name__ == "__main__":
    test_tp4_tucker()


# torchrun --nproc_per_node=4 test_tucker_layer.py
