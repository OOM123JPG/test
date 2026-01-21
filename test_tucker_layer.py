import os
import sys

# 1. 环境变量必须在 import torch 之前设置
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = 'info'
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'
# 强制指定当前进程使用的本地卡号，防止 local_rank=-1 导致的问题
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(local_rank)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
import torch.nn.functional as F

# 补丁：解决某些版本 infer_schema 缺失问题
import torch.library
if not hasattr(torch.library, "infer_schema"):
    torch.library.infer_schema = lambda *args, **kwargs: None

# 添加 vLLM 路径
sys.path.append("/vllm-workspace/vllm")

from vllm.distributed import (
    initialize_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from transformers import DeepseekV2Config

def test_tp4_tucker():
    # 获取分布式参数
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 绑定设备
    torch.npu.set_device(local_rank)
    
    if rank == 0:
        print(f"\n>>> [START] 正在启动 TP4 测试...")
        print(f">>> [INFO] 专家分解路径: /nfs-share/wx1463835/tdmoe/output/decomp_results")
        print(f">>> [INFO] 模型路径: /nfs-share/wx1463835/download/model/Deepseek-V3-bf16")

    # 1. 初始化分布式环境
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:29505",
            backend="hccl"
        )
    
    # 2. 初始化张量并行 TP=4
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(f">>> [STEP] 分布式组初始化完成，准备实例化 MoE 层...")

    # 3. 构造 DeepSeek-V3 配置
    config = DeepseekV2Config()
    config.hidden_size = 7168
    config.moe_intermediate_size = 2048
    config.n_routed_experts = 256
    config.num_experts_per_tok = 8
    config.n_group = 8
    config.topk_group = 1
    config.norm_topk_prob = True
    config.scoring_func = "softmax"
    
    # Tucker 专用配置
    config.use_tucker = True
    config.tucker_layers = [16]
    config.tucker_path = "/nfs-share/wx1463835/tdmoe/output/decomp_results"
    
    class MockParallelConfig:
        def __init__(self):
            self.use_sequence_parallel_moe = False
            self.enable_eplb = False
            self.eplb_config = type('obj', (object,), {'num_redundant_experts': 0})()

    # 4. 实例化 MoE 层
    try:
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)
        
        if rank == 0:
            print(f">>> [SUCCESS] MoE 层加载成功，开始推理验证...")

        # 5. 推理验证
        # 确保所有 rank 看到相同的随机输入
        torch.manual_seed(42)
        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        
        # 同步一下，防止某张卡跑太快
        dist.barrier()
        
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n>>> [RESULT] 推理完成！")
            print(f">>> [RESULT] 输出形状: {output.shape}")
            print(f">>> [RESULT] 输出均值: {output.abs().mean().item():.8f}")
            
    except Exception as e:
        print(f"Rank {rank} 运行报错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_tp4_tucker()
