### 昇腾双机分布式部署指令集 (TP8 PP4)

---

#### 1. Node 0 (7.242.104.207) - 主节点指令
# 1. 彻底清理
pkill -9 python3 && pkill -9 ray

# 2. 设置环境变量 (延用你之前的成功设置)
export ASCEND_GLOBAL_LOG_LEVEL=error
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=7.242.104.207
export TP_SOCKET_IFNAME=eth0   # 确保这是你 torchrun 时用的网卡
export HCCL_CONNECT_TIMEOUT=300

# 3. 启动 Ray Head (强制指定物理 IP 避免解析失败)
ray start --head --node-ip-address 7.242.104.207 --port 6379

# 第四步：启动 vLLM 服务 (等待 Node 1 加入后执行)
# python3 -m vllm.entrypoints.openai.api_server \
#     --model /home/GZGKD001/tmp/models/DeepSeek-V3-bf16 \
#     --tensor-parallel-size 8 \
#     --pipeline-parallel-size 4 \
#     --dtype bfloat16 \
#     --trust-remote-code \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.90 \
#     --distributed-executor-backend ray \
#     --enforce-eager \
#     --served-model-name deepseek-v3-tucker

---

#### 2. Node 1 (7.242.109.127) - 工作节点指令
# 1. 彻底清理
pkill -9 python3 && pkill -9 ray

# 2. 设置环境变量 (延用你之前的成功设置)
export ASCEND_GLOBAL_LOG_LEVEL=error
export HCCL_WHITELIST_DISABLE=1
export HCCL_IF_IP=7.242.109.127
export TP_SOCKET_IFNAME=eth0
export HCCL_CONNECT_TIMEOUT=300

# 3. 加入集群 (直接用 IP)
ray start --address 7.242.104.207:6379 --node-ip-address 7.242.109.127



# 第三步：加入 Ray 集群
ray start --address 7.242.104.207:6379 --node-ip-address 7.242.109.127

顺序：先在 Node 0 启动 Ray Head，然后在 Node 1 启动 Ray Worker，最后在 Node 0 运行 Python 服务指令。

验证：在 Node 0 启动 Python 前，运行 ray status。如果看到 Logical Resources: 16/16 NPU，说明通信已彻底打通。

日志：如果启动过程中显存卡在 115MB，请观察 Node 0 的控制台输出，你应该能看到你修改的 [TUCKER] Rank X 正在读取... 的分片加载日志。
