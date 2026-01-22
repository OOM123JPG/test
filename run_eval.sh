#!/bin/bash

# ==========================================
# 配置区域
# ==========================================

# 评估任务列表 (对应你要求的 arc_c, arc_e, winogrande, piqa)
TASKS=("arc_challenge" "arc_easy" "winogrande" "piqa")

# 日志存放目录 (建议存放在当前路径下的 logs 文件夹)
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 基础参数配置
MODEL_TYPE="local-completions"
BASE_URL="http://10.120.72.45:8888/v1/completions"
TOKENIZER="/home/GZGKD001/tmp/models/DeepSeek-V3-bf16"
CONCURRENT=8

# lm_eval 的绝对路径 (确保后台运行时环境准确)
LM_EVAL_BIN="/home/GZGKD001/miniconda3/envs/evalscope/bin/lm_eval"

# ==========================================
# 执行逻辑
# ==========================================

echo "批量评估启动时间: $(date)"
echo "待执行任务: ${TASKS[*]}"
echo "------------------------------------------"

for TASK in "${TASKS[@]}"
do
    TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/${TASK}_${TIMESTAMP}.log"
    
    echo "[$(date "+%H:%M:%S")] 正在评估任务: $TASK ..."
    echo "详细日志请查看: $LOG_FILE"

    # 执行评估
    # 使用 || true 确保即便报错退出，循环也会继续执行下一个任务
    $LM_EVAL_BIN --model $MODEL_TYPE \
        --tasks $TASK \
        --model_args model=deepseek-v3-tucker,base_url=$BASE_URL,num_concurrent=$CONCURRENT,tokenized_requests=False,logprobs=1,tokenizer=$TOKENIZER \
        > "$LOG_FILE" 2>&1 || {
            echo "!! 错误: 任务 $TASK 执行失败，已跳过。"
        }

    if [ $? -eq 0 ]; then
        echo "[$(date "+%H:%M:%S")] 任务 $TASK 评估完成。"
    fi

    # 任务间稍作停顿，释放连接
    sleep 5
done

echo "------------------------------------------"
echo "所有任务处理完毕。结束时间: $(date)"
