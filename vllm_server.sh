ray start --head --node-ip-address 7.242.104.207 --port 6379
ray start --address 7.242.104.207:6379 --node-ip-address 7.242.109.127

# 在两台机器的终端分别执行：
export HCCL_IF_IP=$(hostname -I | awk '{print $1}') # 自动获取内网IP
export TP_SOCKET_IFNAME=eth0   # 请根据 ifconfig 确认你的网卡名
export HCCL_WHITELIST_DISABLE=1
export HCCL_DETERMINISTIC=1
export HCCL_CONNECT_TIMEOUT=300

# Node0
pkill -9 python

python3 -m vllm.entrypoints.openai.api_server \
    --model /home/GZGKD001/tmp/models/DeepSeek-V3-bf16 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --distributed-executor-backend ray \
    --enforce-eager \
    --served-model-name deepseek-v3-tucker \
    --block-size 16 \
    --disable-log-requests
