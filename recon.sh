export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export HCCL_BUFFSIZE=120
export HCCL_IF_IP="10.120.72.45"

python /home/GZGKD001/tmp/yanhong/tdmoe_deepseek/distribution/reconstruct_inference.py \
    --model_path /home/GZGKD001/tmp/models/DeepSeek-V3-bf16 \
    --whitening_dir /home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/decompostion/wikitext2_n128 \
    --node_rank 0 \
    --master_addr 10.120.72.45 \
    --layers 3 4 5 6 7 8 11 12 13 14 15 16 17 34 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 58 60


export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export HCCL_BUFFSIZE=120

python /home/GZGKD001/tmp/yanhong/tdmoe_deepseek/distribution/reconstruct_inference.py \
    --model_path /home/GZGKD001/tmp/models/DeepSeek-V3-bf16 \
    --whitening_dir /home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/decompostion/wikitext2_n128 \
    --node_rank 1 \
    --master_addr 10.120.72.45 \
    --layers 3 4 5 6 7 8 11 12 13 14 15 16 17 34 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 58    
