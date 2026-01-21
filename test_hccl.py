import torch
import torch_npu
import torch.distributed as dist
import os
import datetime

# 增加初始化超时设置 (10分钟)
def test():
    try:
        # 设置超时时间，避免无限死锁
        timeout = datetime.timedelta(seconds=600)
        dist.init_process_group(backend="hccl", timeout=timeout)
        
        rank = dist.get_rank()
        # 这里的设备绑定要准确
        device_id = rank % 8
        torch.npu.set_device(device_id)
        
        tensor = torch.ones([1024], dtype=torch.float32).to(f"npu:{device_id}")
        
        print(f"Rank {rank} is starting All-Reduce...")
        dist.all_reduce(tensor)
        print(f"Rank {rank} successfully all-reduced! Value: {tensor[0].item()}")
        
    except Exception as e:
        print(f"Error on Rank {rank if 'rank' in locals() else 'unknown'}: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    test()
