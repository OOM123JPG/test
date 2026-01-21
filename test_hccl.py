import torch
import torch_npu
import torch.distributed as dist

dist.init_process_group(backend="hccl")
rank = dist.get_rank()
tensor = torch.ones([1024], dtype=torch.float32).to(f"npu:{rank % 8}")

# 尝试一次全集约，如果能打印出结果且不卡住，说明双机网络彻底通了
dist.all_reduce(tensor)
print(f"Rank {rank} successfully all-reduced! Value: {tensor[0]}")
dist.destroy_process_group()
