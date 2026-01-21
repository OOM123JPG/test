import os
import torch
import torch.distributed as dist
import torch_npu
from datetime import timedelta

def test_hccl_connectivity():
    # 1. 环境变量与设备绑定诊断
    rank = int(os.environ.get("RANK", 0))
    l_rank = os.environ.get("LOCAL_RANK")
    if l_rank is None or int(l_rank) == -1:
        l_rank = rank % 8
    else:
        l_rank = int(l_rank)
    
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 强制绑定 NPU
    torch.npu.set_device(l_rank)
    
    print(f"[RANK {rank}] 尝试初始化 HCCL 组: LocalRank={l_rank}, WorldSize={world_size}")
    
    # 2. 初始化分布式 (设置 60 秒超时，防止无限卡死)
    try:
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60) # 修正处：使用 datetime.timedelta
        )
        print(f"[RANK {rank}] HCCL 握手成功！")

        # 3. 简单的通信测试：All-Reduce
        # 每张卡贡献一个 1.0，求和后 4 张卡都应该得到 4.0
        tensor = torch.ones(1, device=f"npu:{l_rank}")
        dist.all_reduce(tensor)
        
        # 强制同步等待结果
        torch.npu.synchronize()
        
        if rank == 0:
            print(f"\n>>> [RESULT] HCCL 通信完全打通！")
            print(f">>> [RESULT] 4卡求和验证结果: {tensor.item()} (预期: {float(world_size)})")

    except Exception as e:
        print(f"[RANK {rank}] 通信失败或超时! 错误信息: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test_hccl_connectivity()
