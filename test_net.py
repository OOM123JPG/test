import os
import torch
import torch.distributed as dist
import torch_npu
from datetime import timedelta

def test_dist():
    # 环境变量由 Shell 传入或在此手动指定
    rank = int(os.environ.get('NODE_RANK', 0))
    master_addr = os.environ.get('MASTER_ADDR', '10.120.72.45')
    
    print(f"[Rank {rank}] 正在尝试连接 Master: {master_addr}...")
    
    # 初始化
    dist.init_process_group(
        backend='hccl', 
        rank=rank, 
        world_size=2, 
        timeout=timedelta(seconds=60) # 测试时超时设短一点
    )
    
    print(f"[Rank {rank}] 握手成功！准备进入 Barrier...")
    dist.barrier()
    print(f"[Rank {rank}] 所有节点同步完成！网络通路 100% 正常。")

if __name__ == "__main__":
    test_dist()
