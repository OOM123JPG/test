export MASTER_ADDR=10.120.72.45
export MASTER_PORT=29500
export NODE_RANK=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

torchrun --nproc_per_node=8 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    inference_recap.py --model_path /home/GZGKD001/tmp/models/DeepSeek-V3-bf16 \
    --whitening_dir /home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/decomp_results
