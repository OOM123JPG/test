import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
import argparse
import logging
import gc
import traceback
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)

# ==========================================
# 1. 定义 Tucker 分组推理组件 (基于您的旧代码优化)
# ==========================================
# class TuckerGroupLinear(nn.Module):
#     def __init__(self, core, factors):
#         super().__init__()
#         # 使用 register_buffer 确保 to(device) 时自动搬运
#         self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
#         self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        
#         with torch.no_grad():
#             temp_U_E = factors[0].to(torch.bfloat16)
#             temp_core = core.to(torch.bfloat16)
#             # 提前合并核心，维度变为 [E, R_O, R_I]
#             w_low_val = torch.einsum('er,roi->eoi', temp_U_E, temp_core).contiguous()
#             self.register_buffer("W_low", w_low_val)

class TuckerGroupLinear(nn.Module):
    def __init__(self, core, factors):
        super().__init__()
        self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
        self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        
        with torch.no_grad():
            temp_U_E = factors[0].to(torch.bfloat16)
            temp_core = core.to(torch.bfloat16)
            # 严格按照参考代码的维度索引合并：er (U_E) + rud (core) -> eud (merged)
            # 这里的 r 是压缩秩，u 是输出维度，d 是输入维度
            w_low_val = torch.einsum('er,rud->eud', temp_U_E, temp_core).contiguous()
            self.register_buffer("W_low", w_low_val)
            
    def forward(self, x, expert_indices):
        # x: [N, D_in], expert_indices: [N]
        x = torch.matmul(x, self.U_in)
        
        # 兼容 DeepSeek 的批量推理逻辑
        if expert_indices.dim() == 1 and (expert_indices == expert_indices[0]).all():
            core_weight = self.W_low[expert_indices[0]]
            x = torch.matmul(x, core_weight.transpose(-1, -2))
        else:
            current_cores = self.W_low[expert_indices]
            x = torch.bmm(x.unsqueeze(1), current_cores.transpose(1, 2)).squeeze(1)
            
        return torch.matmul(x, self.U_out.T)

class TuckerExpertWrapper(nn.Module):
    def __init__(self, group_module, local_idx):
        super().__init__()
        self.group_module = group_module
        self.local_idx = local_idx

    def forward(self, x):
        # 构造索引 Tensor 以适配 GroupLinear
        indices = torch.full((x.shape[0],), self.local_idx, dtype=torch.long, device=x.device)
        return self.group_module(x, indices)

# ==========================================
# 2. 单层在位替换函数
# ==========================================
@torch.no_grad()
def reconstruct_single_layer(layer, layer_idx, args):
    num_groups = 8
    experts_per_group = 32
    for g_idx in range(num_groups):
        decomp_file = os.path.join(args.whitening_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
        if not os.path.exists(decomp_file): continue
        
        group_data = torch.load(decomp_file, map_location='cpu', weights_only=True)
        
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            data = group_data[proj_name]
            # 创建共享的组模块
            group_module = TuckerGroupLinear(data['core'], data['factors'])
            
            for local_idx in range(experts_per_group):
                global_idx = g_idx * experts_per_group + local_idx
                if global_idx < len(layer.mlp.experts):
                    new_expert = TuckerExpertWrapper(group_module, local_idx)
                    # 替换原 Linear
                    setattr(layer.mlp.experts[global_idx], proj_name, new_expert)
    gc.collect()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs='+', default=range(3, 61))
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="10.120.72.45")
    return parser.parse_args()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    args = get_args()
    
    # 1. 环境与通信初始化
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '8,9,10,11,12,13,14,15'
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '29506'
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    main_npu_dev = "npu:0" 

    logging.info(f"==> Node {args.node_rank} 初始化 HCCL 通信...")
    dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2)
    torch.npu.set_device(main_npu_dev)

    # 2. 模型加载 (CPU)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map={"": "cpu"}
    )
    config = model.config

    # 3. Pipeline 切分
    all_layers = model.model.layers
    # mid_point = len(all_layers) // 2
    mid_point=27
    if args.node_rank == 0:
        model.model.layers = nn.ModuleList([all_layers[j] for j in range(0, mid_point)])
        start_idx = 0
    else:
        model.model.layers = nn.ModuleList([all_layers[j] for j in range(mid_point, len(all_layers))])
        start_idx = mid_point
    
    del all_layers
    gc.collect()

    # 4. 动态显存部署逻辑：自动避让 OOM
    model.model.embed_tokens.to(main_npu_dev)
    if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu_dev)

    logging.info(f"==> 开始动态 NPU 部署 (Node {args.node_rank})")
    logging.info("-" * 95)
    logging.info(f"{'Layer':<6} | {'Target':<8} | {'Type':<8} | {'Usage(GB)':<10} | {'Remain(GB)':<10}")
    logging.info("-" * 95)

    current_card_idx = 0
    last_active_dev = main_npu_dev

    for idx, layer in enumerate(model.model.layers):
        real_idx = start_idx + idx
        
        # 预手术：CPU 侧执行 Tucker 替换
        l_type = "Normal"
        if real_idx in args.layers:
            reconstruct_single_layer(layer, real_idx, args)
            l_type = "Tucker"
        
        success = False
        while current_card_idx < 8:
            target_dev = f"npu:{current_card_idx}"
            free_byte, _ = torch.npu.mem_get_info(target_dev)
            free_gb = free_byte / (1024**3)
            
            # 阈值：Normal 层约 22GB, Tucker 层约 5.5GB，额外留 8GB 冗余
            required = 22.0 if l_type == "Normal" else 5.5
            
            if free_gb > (required + 8.0):
                try:
                    layer.to(target_dev)
                    torch.npu.empty_cache()
                    new_free, _ = torch.npu.mem_get_info(target_dev)
                    usage_actual = free_gb - (new_free / 1024**3)
                    logging.info(f"L{real_idx:02d}    | {target_dev:<8} | {l_type:<8} | {usage_actual:>9.2f} | {new_free/1024**3:>9.2f}")
                    last_active_dev = target_dev
                    success = True
                    break 
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        current_card_idx += 1
                        torch.npu.empty_cache()
                        continue
                    else: raise e
            else:
                current_card_idx += 1
        
        if not success:
            logging.error(f"❌ OOM：8张NPU显存已耗尽，无法放置 L{real_idx}。请增加压缩比例！")
            sys.exit(1)

    # 结尾组件放在流转的最后一站
    model.model.norm.to(last_active_dev)
    if hasattr(model, "lm_head"):
        model.lm_head.to(last_active_dev)
    logging.info("-" * 95)

    # 5. 修复了 Pos_ids 和 Mask 的流式长文本生成
    max_new_tokens = 50
    
    try:
        with torch.no_grad():
            test_input = "什么是人工智能？请简短回答。"
            inputs = tokenizer(test_input, return_tensors="pt")
            # 维护完整的 Sequence ID 列表
            input_ids_list = inputs.input_ids[0].tolist()
            batch_size = 1

            logging.info("==> [无缓存模式] 启动生成...")
            for step in range(max_new_tokens):
                # 核心区别：每一轮都把 input_ids_list 全部转为 Tensor
                curr_seq_len = len(input_ids_list)
                input_to_run = torch.tensor([input_ids_list], dtype=torch.long)
                
                # Position IDs 必须是全量序列 [0, 1, ..., curr_seq_len-1]
                pos_ids = torch.arange(curr_seq_len).unsqueeze(0)
                
                # Mask 必须是全量的下三角 Causal Mask [1, 1, seq, seq]
                mask = torch.full((curr_seq_len, curr_seq_len), float("-inf"))
                mask = torch.triu(mask, diagonal=1).view(1, 1, curr_seq_len, curr_seq_len)
                mask = mask.to(torch.bfloat16)

                if args.node_rank == 0:
                    # --- Node 0 推理 ---
                    h = model.model.embed_tokens(input_to_run.to(main_npu_dev))
                    for i, layer in enumerate(model.model.layers):
                        dev = next(layer.parameters()).device
                        # 注意：use_cache=False, 不传入 past_key_value
                        outputs = layer(
                            h.to(dev), 
                            attention_mask=mask.to(dev), 
                            position_ids=pos_ids.to(dev), 
                            use_cache=False
                        )
                        h = outputs[0]
                    
                    # 发送全量 Sequence 的 Hidden States
                    dist.send(tensor=h.to(main_npu_dev).contiguous(), dst=1)
                    
                    # 接收最后一个 Token ID
                    new_id_dev = torch.zeros((1, 1), dtype=torch.long, device=main_npu_dev)
                    dist.recv(tensor=new_id_dev, src=1)
                    
                    new_id = new_id_dev.item()
                    input_ids_list.append(new_id)
                    
                    # 实时输出
                    print(tokenizer.decode([new_id]), end="", flush=True)
                    if new_id == tokenizer.eos_token_id: break

                else:
                    # --- Node 1 推理 ---
                    # 接收全量 Hidden States，形状随长度增长
                    r_shape = (batch_size, curr_seq_len, config.hidden_size)
                    h_recv = torch.zeros(r_shape, dtype=torch.bfloat16, device=main_npu_dev)
                    dist.recv(tensor=h_recv, src=0)
                    
                    h = h_recv
                    for i, layer in enumerate(model.model.layers):
                        dev = next(layer.parameters()).device
                        outputs = layer(
                            h.to(dev), 
                            attention_mask=mask.to(dev), 
                            position_ids=pos_ids.to(dev), 
                            use_cache=False
                        )
                        h = outputs[0]
                    
                    # 只需要取最后一个 Position 的 Logits 来预测
                    head_dev = next(model.lm_head.parameters()).device
                    h_final = model.model.norm(h.to(head_dev))
                    logits = model.lm_head(h_final)
                    
                    # 获取最新一个 token
                    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    # 发送回 Node 0
                    dist.send(tensor=next_token_id.to(main_npu_dev).contiguous(), dst=0)
                    
                    # Node 1 也同步列表以维持下一轮循环的 curr_seq_len
                    input_ids_list.append(next_token_id.item())
                    if next_token_id.item() == tokenizer.eos_token_id: break

            if args.node_rank == 0:
                logging.info("\n==> [DONE] 无缓存模式推理完成。")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
        
