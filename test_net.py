import torch
import torch_npu

for i in range(8):
    try:
        device = f"npu:{i}"
        print(f"Testing {device}...")
        # 尝试在该卡上分配一个小张量
        x = torch.randn(1024, 1024).to(device)
        print(f"{device} OK.")
        del x
        torch.npu.empty_cache()
    except Exception as e:
        print(f"{device} FAILED: {e}")
        
import torch
import os

path = "你的whitening_dir/layer_3_group_0.pt" # 换成你报错那一层的路径
print("Attempting to load file...")
data = torch.load(path, map_location='cpu')
print("Load success. Keys:", data.keys())

# import os
# import torch
# import torch.distributed as dist
# import torch_npu
# from datetime import timedelta

# def test_dist():
#     # 环境变量由 Shell 传入或在此手动指定
#     rank = int(os.environ.get('NODE_RANK', 0))
#     master_addr = os.environ.get('MASTER_ADDR', '10.120.72.45')
    
#     print(f"[Rank {rank}] 正在尝试连接 Master: {master_addr}...")
    
#     # 初始化
#     dist.init_process_group(
#         backend='hccl', 
#         rank=rank, 
#         world_size=2, 
#         timeout=timedelta(seconds=60) # 测试时超时设短一点
#     )
    
#     print(f"[Rank {rank}] 握手成功！准备进入 Barrier...")
#     dist.barrier()
#     print(f"[Rank {rank}] 所有节点同步完成！网络通路 100% 正常。")

# if __name__ == "__main__":
#     test_dist()

# export HCCL_IF_IP=$LOCAL_IP
# export MASTER_ADDR=10.120.72.45
# export MASTER_PORT=29506
# export NODE_RANK=0
# export HCCL_IF_IP=10.120.72.45
# python3 test_net.py


# export MASTER_ADDR=10.120.72.45
# export MASTER_PORT=29506
# export NODE_RANK=1
# export HCCL_IF_IP=10.120.xx.xx # Node 1 自己的 IP
# python3 test_net.py
