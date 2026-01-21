import os
import sys

# 1. 基础环境配置 (去掉手动设备隔离，由 torchrun 处理)
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = 'error' 
os.environ['HCCL_WHITELIST_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29505'

import torch
import torch_npu  # 必须先 import torch_npu
import torch.distributed as dist

# 2. 强制获取 LOCAL_RANK 并绑定
# 注意：在 8 卡机跑 4 卡，torchrun 会给每个进程分配 LOCAL_RANK 为 0,1,2,3
l_rank = int(os.environ.get("LOCAL_RANK", 0))

import torch.nn as nn
import torch.nn.functional as F

# 补丁
import torch.library
if not hasattr(torch.library, "infer_schema"):
    torch.library.infer_schema = lambda *args, **kwargs: None

sys.path.append("/vllm-workspace/vllm")

# 导入 vllm 相关组件
from vllm.distributed import (
    initialize_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from transformers import DeepseekV2Config



# --- 核心修复：强制获取正确的 local_rank ---
def get_local_rank():
    # 优先尝试 torchrun 注入的变量
    l_rank = os.environ.get("LOCAL_RANK")
    if l_rank is not None and int(l_rank) != -1:
        return int(l_rank)
    # 兜底方案：如果为 -1，根据全局 RANK 推算（假设单机运行）
    g_rank = os.environ.get("RANK")
    if g_rank is not None:
        return int(g_rank) % 8
    return 0

l_rank = get_local_rank()
# 强制让当前进程只能看到对应的那张 NPU
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(l_rank)

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
    # 获取全局参数
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 1. 严格绑定 NPU 设备
    torch.npu.set_device(l_rank)
    
    if rank == 0:
        print(f">>> [DEBUG] Rank 0 确认：WorldSize={world_size}, LocalRank={l_rank}")
        print(f">>> [DEBUG] 正在建立 HCCL 通信组...")

    # 2. 初始化分布式环境
    if not dist.is_initialized():
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method="tcp://127.0.0.1:29505",
            backend="hccl"
        )
    
    # 3. 初始化 TP4 (此处会阻塞直到 4 张卡全部就绪)
    if not ensure_model_parallel_initialized(4, 1):
        initialize_model_parallel(tensor_model_parallel_size=4, pipeline_model_parallel_size=1)
    
    if rank == 0:
        print(f">>> [SUCCESS] HCCL 握手成功！开始加载权重...")

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
    
    config.use_tucker = True
    config.tucker_layers = [16]
    config.tucker_path = "/nfs-share/wx1463835/tdmoe/output/decomp_results"
    
    class MockParallelConfig:
        def __init__(self):
            self.use_sequence_parallel_moe = False
            self.enable_eplb = False
            self.eplb_config = type('obj', (object,), {'num_redundant_experts': 0})()

    # 5. 实例化并推理
    try:
        moe_layer = DeepseekV2MoE(
            config=config,
            parallel_config=MockParallelConfig(),
            prefix="model.layers.16.mlp",
            layer_idx=16
        ).to("npu").to(torch.bfloat16)

        dist.barrier() # 同步

        dummy_input = torch.randn(8, 7168).to("npu").to(torch.bfloat16)
        with torch.no_grad():
            output = moe_layer(dummy_input)

        if rank == 0:
            print(f"\n>>> 推理完成！")
            print(f">>> Output Shape: {output.shape}")
            print(f">>> Output Mean: {output.abs().mean().item():.8f}")

    except Exception as e:
        print(f"Rank {rank} 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    run_test()
