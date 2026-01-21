import os
import sys
from datetime import timedelta

# 1. 强制环境对齐 (必须最先执行)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '31288'
os.environ['HCCL_WHITELIST_DISABLE'] = '1'

import torch
import torch_npu
import torch.distributed as dist

# 获取 Rank 并绑定设备 (延用 check_hccl.py 的成功逻辑)
rank = int(os.environ.get("RANK", 0))
l_rank = int(os.environ.get("LOCAL_RANK", rank % 8))
torch.npu.set_device(l_rank)

import torch.nn as nn
import torch.nn.functional as F

# vLLM 路径与补丁
sys.path.append("/vllm-workspace/vllm")
import torch.library
if not hasattr(torch.library, "infer_schema"):
    torch.library.infer_schema = lambda *args, **kwargs: None

from vllm.distributed import (
    initialize_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from transformers import DeepseekV2Config

def run_tucker_test():
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 2. 步骤一：基础分布式握手 (参考 check_hccl.py)
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:31288",
            backend="hccl",
            timeout=timedelta(seconds=60)
        )
    
    # 3. 步骤二：初始化 vLLM 并行组 (TP4)
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(">>> [STEP 1] 并行环境初始化成功")

    # 4. 步骤三：构造配置
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

    # 5. 步骤四：实例化模型 (触发 DistributedTuckerLinear 的分片加载)
    try:
        if rank == 0: print(">>> [STEP 2] 开始分片加载 Tucker 权重...")
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        # 强制同步加载结果
        dist.barrier()
        
        # 6. 推理验证
        torch.manual_seed(42)
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n>>> [SUCCESS] Tucker TP4 推理完成！")
            print(f">>> 输出形状: {output.shape}")
            print(f">>> 输出均值: {output.abs().mean().item():.8f}")

    except Exception as e:
        print(f"[RANK {rank}] 运行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_tucker_test()
