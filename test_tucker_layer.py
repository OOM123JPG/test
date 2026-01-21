import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu

# --- 1. 强制环境对齐 (必须在所有 import 之前) ---
def setup_env():
    rank = int(os.environ.get("RANK", 0))
    l_rank = os.environ.get("LOCAL_RANK", "-1")
    if l_rank == "-1":
        l_rank = rank % 8
    else:
        l_rank = int(l_rank)
    
    os.environ['LOCAL_RANK'] = str(l_rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '31288'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'
    # 强制让 NPU 看到正确的卡
    torch.npu.set_device(l_rank)
    return rank, l_rank

rank, l_rank = setup_env()

# 补丁：解决 vLLM 的配置引用问题
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

def run_test():
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 2. 初始化分布式 (显式传参，不依赖自动注入)
    if not dist.is_initialized():
        print(f"[RANK {rank}] 启动 HCCL 握手，LocalRank={l_rank}...")
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:31288",
            backend="hccl"
        )
    
    # 3. 核心：在实例化模型前，必须先建立 TP 组
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(">>> [SUCCESS] 并行环境就绪，开始构造配置...")

    # 4. 构造配置 (DeepSeek-V3 规格)
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

    # 5. 加载模型
    try:
        print(f"[RANK {rank}] 正在从 NFS 加载 Tucker 专家...")
        # 传入 layer_idx 触发你的 Tucker 加载逻辑
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        dist.barrier()
        
        # 推理测试
        torch.manual_seed(42)
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n[RESULT] 推理成功！")
            print(f"Output Mean: {output.abs().mean().item():.8f}")

    except Exception as e:
        print(f"[RANK {rank}] 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
