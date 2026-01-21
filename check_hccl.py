import os
import torch
import torch.distributed as dist
import torch_npu

def test_hccl_connectivity():
    # 自动获取 torchrun 注入的参数
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 4))
    
    # 1. 绑定设备
    torch.npu.set_device(local_rank)
    
    print(f"[RANK {rank}] 正在尝试初始化进程组 (MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')})")
    
    # 2. 初始化分布式 (设置超时时间为 60 秒，防止无限死等)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        timeout=torch.timedelta(seconds=60)
    )
    
    print(f"[RANK {rank}] HCCL 握手成功！")

    # 3. 执行一次简单的 All-Reduce 测试
    tensor = torch.ones(1, device=f"npu:{local_rank}")
    dist.all_reduce(tensor)
    
    if rank == 0:
        print(f">>> [RESULT] HCCL 测试通过！4张卡求和结果: {tensor.item()} (预期应为 {world_size}.0)")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_hccl_connectivity()


# export HCCL_DEBUG=1
# export ASCEND_GLOBAL_LOG_LEVEL=info

# # 启动 4 卡测试
# torchrun --nproc_per_node=4 check_hccl.py
