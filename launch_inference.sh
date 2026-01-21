# launch_inference.sh
export MASTER_ADDR=192.168.1.100 # 机器0的实际IP
export MASTER_PORT=29500
export nnodes=2
export nproc_per_node=8

# 关键环境变量：防止昇腾多网卡环境死锁
export HCCL_IF_IP=$(hostname -I | awk '{print $1}') 
export FI_PROVIDER=tcp
export HCCL_WHITELIST_DISABLE=1

# 根据是哪台机器设置 node_rank
# 机器0设为0，机器1设为1
export node_rank=0 

torchrun \
    --nnodes=$nnodes \
    --node_rank=$node_rank \
    --nproc_per_node=$nproc_per_node \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/reconstruct_and_inference.py \
    --model_path /home/models/DeepSeek-V3-bf16 \
    --decomp_dir ./output/decomp_results
######################################


机器 0 (192.168.1.10):

export HCCL_IF_IP=192.168.1.10
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 --master_addr=192.168.1.10 --master_port=29500 src/reconstruct_and_inference.py --model_path /models/ds-v3 --decomp_dir ./output/decomp_results

机器 1 (192.168.1.11):
export HCCL_IF_IP=192.168.1.11
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 --master_addr=192.168.1.10 --master_port=29500 src/reconstruct_and_inference.py --model_path /models/ds-v3 --decomp_dir ./output/decomp_results


###########################################
# 清理之前的残余进程
pkill -9 python

# 指定 NPU 通信使用的网卡 IP
export HCCL_IF_IP=7.242.104.207
# 允许动态组网
export HCCL_WHITELIST_DISABLE=1
# 某些环境下需要指定具体的网卡名，例如 eth0 或 enp... (用 ifconfig 查看)
# export HCCL_SOCKET_IFNAME=eth0 

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \
--master_addr=7.242.104.207 --master_port=29500 \
src/test_hccl.py



pkill -9 python

export HCCL_IF_IP=7.242.109.127
export HCCL_WHITELIST_DISABLE=1

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=8 \
--master_addr=7.242.104.207 --master_port=29500 \
src/test_hccl.py
