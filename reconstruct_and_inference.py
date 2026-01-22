# 加kv cache
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
import argparse
import logging
import gc
import time
import traceback
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

# ==========================================
# 0. 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ==========================================
# 1. Tucker 组件
# ==========================================
class TuckerGroupLinear(nn.Module):
    def __init__(self, core, factors):
        super().__init__()
        self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
        self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        with torch.no_grad():
            # 权重合并：U_E * core -> W_low
            w_low_val = torch.einsum('er,rud->eud', factors[0].to(torch.bfloat16), core.to(torch.bfloat16)).contiguous()
            self.register_buffer("W_low", w_low_val)
            
    def forward(self, x, expert_indices):
        x = torch.matmul(x, self.U_in)
        if expert_indices.dim() == 1 and (expert_indices == expert_indices[0]).all():
            x = torch.matmul(x, self.W_low[expert_indices[0]].transpose(-1, -2))
        else:
            x = torch.bmm(x.unsqueeze(1), self.W_low[expert_indices].transpose(1, 2)).squeeze(1)
        return torch.matmul(x, self.U_out.T)

class TuckerExpertWrapper(nn.Module):
    def __init__(self, group_module, local_idx):
        super().__init__()
        self.group_module, self.local_idx = group_module, local_idx

    def forward(self, x):
        indices = torch.full((x.shape[0],), self.local_idx, dtype=torch.long, device=x.device)
        return self.group_module(x, indices)

# ==========================================
# 2. 推理辅助逻辑 (采样)
# ==========================================
def apply_sampling(logits, temperature, top_p, repetition_penalty, batch_ids):
    if repetition_penalty != 1.0:
        for i in range(logits.shape[0]):
            for token_id in set(batch_ids[i]):
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= repetition_penalty
                else:
                    logits[i, token_id] *= repetition_penalty
    if temperature > 0:
        logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(logits.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = float('-inf')
    return logits

# ==========================================
# 3. 核心：并行手术 + 顺序装箱流水线
# ==========================================
def pipeline_loading_and_surgery(model, start_idx, args):
    """
    CPU 多核并行执行 Tucker 变换，但主线程严格按物理层序搬运到 NPU。
    """
    num_layers = len(model.model.layers)
    num_groups, experts_per_group = 8, 32
    RESERVED, main_npu = 6.0, "npu:0" # 预留 6GB 缓冲区
    
    # --- 生产者：后台并行执行手术 ---
    def producer_task(idx_in_module, layer):
        real_idx = start_idx + idx_in_module
        try:
            if real_idx in args.layers:
                for g_idx in range(num_groups):
                    path = os.path.join(args.whitening_dir, f"layer_{real_idx}_group_{g_idx}.pt")
                    if not os.path.exists(path): continue
                    data = torch.load(path, map_location='cpu', weights_only=True)
                    for proj in ['gate_proj', 'up_proj', 'down_proj']:
                        gm = TuckerGroupLinear(data[proj]['core'], data[proj]['factors'])
                        for l_idx in range(experts_per_group):
                            g_exp = g_idx * experts_per_group + l_idx
                            if g_exp < len(layer.mlp.experts):
                                setattr(layer.mlp.experts[g_exp], proj, TuckerExpertWrapper(gm, l_idx))
            return True
        except Exception as e:
            logging.error(f"L{real_idx} 手术失败: {e}")
            return False

    # 提交所有手术任务到线程池
    executor = ThreadPoolExecutor(max_workers=8)
    futures = [executor.submit(producer_task, i, model.model.layers[i]) for i in range(num_layers)]

    # --- 消费者：主线程顺序分发 ---
    logging.info("-" * 105)
    logging.info(f"{'Layer':<6} | {'Target':<8} | {'Type':<8} | {'Est. Usage(GB)':<15} | {'Status':<8}")
    logging.info("-" * 105)

    curr_card = 0
    last_active = main_npu
    if args.node_rank == 0:
        model.model.embed_tokens.to(main_npu)
        if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu)

    for i in range(num_layers):
        real_idx = start_idx + i
        # 核心：等待当前层手术完成，确保顺序
        futures[i].result() 
        
        layer = model.model.layers[i]
        l_type = "Tucker" if real_idx in args.layers else "Normal"
        req = 22.0 if l_type == "Normal" else 5.5
        success = False
        
        while curr_card < 8:
            target = f"npu:{curr_card}"
            # 搬运前的显存状态
            free_before = torch.npu.mem_get_info(target)[0] / (1024**3)
            
            if free_before > (req + RESERVED):
                try:
                    layer.to(target)
                    torch.npu.empty_cache() # 清理搬运过程中的临时拷贝
                    
                    # 搬运后的显存状态
                    free_after = torch.npu.mem_get_info(target)[0] / (1024**3)
                    actual_usage = free_before - free_after # 这才是最真实的占用
                    
                    logging.info(f"L{real_idx:02d}    | {target:<8} | {l_type:<8} | {actual_usage:>14.2f} | SUCCESS")
                    last_active, success = target, True
                    break
                except:
                    curr_card += 1
            else:
                curr_card += 1
        
        if not success:
            layer.to("cpu")
            logging.warning(f"L{real_idx:02d}    | {'cpu':<8} | {l_type:<8} | {'-':>14} | OFFLOAD")

    if args.node_rank == 1:
        model.model.norm.to(last_active)
        if hasattr(model, "lm_head"): model.lm_head.to(last_active)

    logging.info("-" * 105)
    executor.shutdown(wait=True)

# ==========================================
# 4. 分布式推理循环 (无缓存重算模式)
# ==========================================
def run_no_cache_distributed(model, tokenizer, args, prompts):
    config = model.config
    batch_size = len(prompts)
    total_start_t = time.time()
    
    try:
        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            batch_ids = inputs.input_ids.tolist()
            
            for step in range(args.max_new_tokens):
                curr_ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
                seq_len = curr_ids_tensor.shape[1]
                pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
                mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1).view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1).to(torch.bfloat16)

                if args.node_rank == 0:
                    # Node 0 计算
                    h = model.model.embed_tokens(curr_ids_tensor.to("npu:0"))
                    for layer in model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
                    # 发送 Hidden States
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
                    # 接收新 Token
                    new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids_dev, src=1)
                    for i in range(batch_size): batch_ids[i].append(new_ids_dev[i].item())
                    
                    if args.stream: print(tokenizer.decode([batch_ids[0][-1]]), end="", flush=True)

                else:
                    # Node 1 计算
                    h_recv = torch.zeros((batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0)
                    
                    h = h_recv
                    for layer in model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
                    head_dev = next(model.lm_head.parameters()).device
                    logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
                    # 采样
                    if args.do_sample:
                        logits = apply_sampling(logits, args.temperature, args.top_p, args.repetition_penalty, batch_ids)
                        next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    else:
                        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
                    for i in range(batch_size): batch_ids[i].append(next_tokens[i].item())

                if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids): break
            
            if args.node_rank == 0:
                print(f"\n\n推理汇总:\n" + "="*50)
                for i, res in enumerate(batch_ids):
                    print(f"[{i+1}] {tokenizer.decode(res, skip_special_tokens=True)}\n")
                print(f"总耗时: {time.time()-total_start_t:.2f}s")
    except Exception: traceback.print_exc()



from transformers.cache_utils import DynamicCache

# def run_with_kv_cache_distributed(model, tokenizer, args, prompts):
#     config = model.config
#     batch_size = len(prompts)
    
#     # 1. 初始化分布式 KV Cache 对象
#     # 每个节点只会在自己持有的层索引处（如 Node 1 负责 29-61 层）写入缓存
#     past_key_values = DynamicCache()
    
#     # 编码 Prompt
#     inputs = tokenizer(prompts, return_tensors="pt", padding=True)
#     batch_ids = inputs.input_ids.tolist()
    
#     # 初始输入（Prefill 阶段是整个 Prompt）
#     curr_input_ids = inputs.input_ids.to("npu:0")
#     # 初始位置编码: 0, 1, 2, ..., seq_len-1
#     position_ids = torch.arange(curr_input_ids.shape[1]).unsqueeze(0).expand(batch_size, -1).to("npu:0")
    
#     total_start_t = time.time()

#     try:
#         with torch.no_grad():
#             for step in range(args.max_new_tokens):
#                 is_prefill = (step == 0)
#                 # 核心修复：在这里定义 total_len，确保 Node 0 和 Node 1 都能访问到
#                 total_len = len(batch_ids[0])
#                 seq_len = curr_input_ids.shape[1]

#                 # 2. 准备 Attention Mask
#                 if is_prefill:
#                     # Prefill 阶段: 标准因果三角掩码 (bs, 1, q_len, kv_len)
#                     mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1)
#                     mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1)
#                 else:
#                     # Decoding 阶段: 新 Token 关注过去所有 Token
#                     # 此时 total_len 是包含已生成的 token 的总长度
#                     mask = torch.zeros((batch_size, 1, 1, total_len))
                
#                 mask = mask.to(torch.bfloat16).to("npu:0")

#                 if args.node_rank == 0:
#                     # ================= Node 0 计算 (Layer 0 - 28) =================
#                     h = model.model.embed_tokens(curr_input_ids)
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         # 传入 past_key_value，层内部会自动根据其全局 .layer_idx 存储 KV
#                         layer_outputs = layer(
#                             h.to(dev), 
#                             attention_mask=mask.to(dev), 
#                             position_ids=position_ids.to(dev), 
#                             past_key_value=past_key_values,
#                             use_cache=True
#                         )
#                         h = layer_outputs[0]
                    
#                     # 发送中间 Hidden States 给 Node 1
#                     dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
#                     # 接收 Node 1 采样出的新 Token ID
#                     new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
#                     dist.recv(tensor=new_ids_dev, src=1)
                    
#                     # 更新本地序列记录
#                     next_token_list = new_ids_dev.cpu().numpy().tolist()
#                     for i in range(batch_size):
#                         batch_ids[i].append(next_token_list[i][0])
                    
#                     # 打印流式输出
#                     if args.stream:
#                         print(tokenizer.decode([batch_ids[0][-1]]), end="", flush=True)

#                     # 准备下一轮 Decoding 的输入
#                     curr_input_ids = new_ids_dev
#                     # 下一轮的位置 ID 就是当前更新后的总长度 - 1
#                     position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")

#                 else:
#                     # ================= Node 1 计算 (Layer 29 - 61) =================
#                     h_recv = torch.zeros((batch_size, seq_len, config.hidden_size), 
#                                        dtype=torch.bfloat16, device="npu:0")
#                     dist.recv(tensor=h_recv, src=0)
                    
#                     h = h_recv
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         layer_outputs = layer(
#                             h.to(dev), 
#                             attention_mask=mask.to(dev), 
#                             position_ids=position_ids.to(dev), 
#                             past_key_value=past_key_values,
#                             use_cache=True
#                         )
#                         h = layer_outputs[0]
                    
#                     # Node 1 负责最后的 Norm、Head 和采样
#                     head_dev = next(model.lm_head.parameters()).device
#                     logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
#                     # 采样逻辑
#                     if args.do_sample:
#                         logits = apply_sampling(logits, args.temperature, args.top_p, args.repetition_penalty, batch_ids)
#                         next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
#                     else:
#                         next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                    
#                     # 将采样结果发送回 Node 0
#                     dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
                    
#                     # Node 1 同步更新 batch_ids 以确保采样逻辑一致
#                     next_token_list = next_tokens.cpu().numpy().tolist()
#                     for i in range(batch_size):
#                         batch_ids[i].append(next_token_list[i][0])

#                     # 准备下一轮输入
#                     curr_input_ids = next_tokens.to("npu:0")
#                     position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")

#                 # 终止符检测
#                 if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids):
#                     break
            
#             # 推理结束汇总
#             if args.node_rank == 0:
#                 print(f"\n\n推理汇总:\n" + "="*50)
#                 for i, res in enumerate(batch_ids):
#                     print(f"[{i+1}] {tokenizer.decode(res, skip_special_tokens=True)}\n")
#                 print(f"总耗时: {time.time()-total_start_t:.2f}s")

#     except Exception:
#         traceback.print_exc()
        
def run_with_kv_cache_distributed(model, tokenizer, args, prompts):
    config = model.config
    batch_size = len(prompts)
    
    # 1. 强制设置左填充 (Left Padding)，这对生成任务至关重要
    # 确保所有序列的最后一个有效 token 都在同一列对齐
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # 记录每个序列的原始 Mask，后续动态更新
    # 形状: (batch_size, seq_len)
    current_padding_mask = inputs.attention_mask.to("npu:0") 
    batch_ids = inputs.input_ids.tolist()
    
    past_key_values = DynamicCache()
    curr_input_ids = inputs.input_ids.to("npu:0")
    
    # 生成初始 Position IDs，考虑到 Left Padding，需要减去 padding 的数量
    # 确保第一个有效 token 的位置从 0 或正确偏移开始
    position_ids = current_padding_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(current_padding_mask == 0, 1) # 临时填 1，后续会被 mask 挡住
    position_ids = position_ids.to("npu:0")

    total_start_t = time.time()

    try:
        with torch.no_grad():
            for step in range(args.max_new_tokens):
                is_prefill = (step == 0)
                total_len = curr_input_ids.shape[1] if is_prefill else len(batch_ids[0])
                q_len = curr_input_ids.shape[1]

                # 2. 核心：构建 4D Attention Mask
                # 官方代码要求形状: (bsz, 1, q_len, kv_seq_len)
                if is_prefill:
                    # Prefill: 结合因果掩码 + Padding 掩码
                    # 先生成因果三角掩码
                    causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
                    # 扩展到 4D 并加上 Padding Mask 的影响
                    # current_padding_mask 为 0 的位置设为 -inf
                    p_mask = current_padding_mask.view(batch_size, 1, 1, q_len)
                    mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - p_mask.to(torch.bfloat16)) * -10000.0
                else:
                    # Decoding: 只需 Padding 掩码（因为是逐个生成，不需要因果三角）
                    # 此时 KV 长度为 total_len，Q 长度为 1
                    mask = (1.0 - current_padding_mask.view(batch_size, 1, 1, total_len).to(torch.bfloat16)) * -10000.0
                
                mask = mask.to(torch.bfloat16)

                if args.node_rank == 0:
                    # --- Node 0 计算 ---
                    h = model.model.embed_tokens(curr_input_ids)
                    for layer in model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=position_ids.to(dev), 
                                  past_key_value=past_key_values, use_cache=True)[0]
                    
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids_dev, src=1)
                    
                    # 更新序列和 Mask
                    new_tokens = new_ids_dev.cpu().numpy().tolist()
                    for i in range(batch_size): batch_ids[i].append(new_tokens[i][0])
                    # 新生成的 token 永远不是 Padding
                    current_padding_mask = torch.cat([current_padding_mask, torch.ones((batch_size, 1), device="npu:0")], dim=-1)
                    
                    curr_input_ids = new_ids_dev
                    position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")
                    
                    if args.stream:
                        print(f"\rStep {step} 生成中...", end="", flush=True)

                else:
                    # --- Node 1 计算 ---
                    h_recv = torch.zeros((batch_size, q_len, config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0)
                    
                    h = h_recv
                    for layer in model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=position_ids.to(dev), 
                                  past_key_value=past_key_values, use_cache=True)[0]
                    
                    head_dev = next(model.lm_head.parameters()).device
                    logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
                    if args.do_sample:
                        logits = apply_sampling(logits, args.temperature, args.top_p, args.repetition_penalty, batch_ids)
                        next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    else:
                        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
                    
                    # 同步更新 Node 1 的 Mask 和序列
                    nt_list = next_tokens.cpu().numpy().tolist()
                    for i in range(batch_size): batch_ids[i].append(nt_list[i][0])
                    current_padding_mask = torch.cat([current_padding_mask, torch.ones((batch_size, 1), device="npu:0")], dim=-1)

                    curr_input_ids = next_tokens.to("npu:0")
                    position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")

                if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids):
                    break
            
            if args.node_rank == 0:
                print(f"\n\n推理汇总:\n" + "="*50)
                for i, res in enumerate(batch_ids):
                    # 跳过 Padding token 进行解码
                    out = tokenizer.decode(res, skip_special_tokens=True)
                    print(f"[{i+1}] {out}\n")
                print(f"总耗时: {time.time()-total_start_t:.2f}s")
    except Exception:
        traceback.print_exc()
                
# ==========================================
# 5. 主程序入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(3, 61)))
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="10.120.72.45")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    # 稳定性环境变量
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '29506'
    os.environ['HCCL_CONNECT_TIMEOUT'] = '7200'
    os.environ['HCCL_EXEC_TIMEOUT'] = '7200'
    os.environ['HCCL_TCP_KEEP_ALIVE_ENABLE'] = '1'
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'

    # 分布式初始化
    dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2, timeout=timedelta(seconds=7200))
    torch.npu.set_device("npu:0")

    # 模型加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
    
    # 负载均衡切分 (Node 0 负责 0-26, Node 1 负责 27-61)
    # mid = 29
    # all_l = model.model.layers
    # if args.node_rank == 0:
    #     model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid)]), 0
    # else:
    #     model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))]), mid
    
    mid = 29
    all_l = model.model.layers # 这是一个包含 0-61 层的列表
    if args.node_rank == 0:
        # Node 0 拿走 0-28 层，它们的 .layer_idx 属性依然是 0-28
        model.model.layers = nn.ModuleList([all_l[i] for i in range(mid)])
        start = 0
    else:
        # Node 1 拿走 29-61 层，它们的 .layer_idx 属性依然是 29-61
        model.model.layers = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))])
        start = mid
        
        
    del all_l
    gc.collect()

    # 执行重构后的顺序装箱流水线
    pipeline_loading_and_surgery(model, start, args)

    logging.info(f"Node {args.node_rank} 准备就绪，同步对端...")
    dist.barrier()
    logging.info("同步成功，启动推理。")

    prompts = ["什么是人工智能？","请写一首关于大海的诗。"]
    # run_no_cache_distributed(model, tokenizer, args, prompts)
    run_with_kv_cache_distributed(model, tokenizer, args, prompts)

if __name__ == "__main__":
    main()
# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch_npu
# import argparse
# import logging
# import gc
# import time
# import traceback
# from datetime import timedelta
# from concurrent.futures import ThreadPoolExecutor
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm

# # ==========================================
# # 0. 日志配置
# # ==========================================
# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s.%(msecs)03d - %(message)s', 
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# # ==========================================
# # 1. Tucker 组件
# # ==========================================
# class TuckerGroupLinear(nn.Module):
#     def __init__(self, core, factors):
#         super().__init__()
#         self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
#         self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
#         with torch.no_grad():
#             # 权重合并：U_E * core -> W_low
#             w_low_val = torch.einsum('er,rud->eud', factors[0].to(torch.bfloat16), core.to(torch.bfloat16)).contiguous()
#             self.register_buffer("W_low", w_low_val)
            
#     def forward(self, x, expert_indices):
#         x = torch.matmul(x, self.U_in)
#         if expert_indices.dim() == 1 and (expert_indices == expert_indices[0]).all():
#             x = torch.matmul(x, self.W_low[expert_indices[0]].transpose(-1, -2))
#         else:
#             x = torch.bmm(x.unsqueeze(1), self.W_low[expert_indices].transpose(1, 2)).squeeze(1)
#         return torch.matmul(x, self.U_out.T)

# class TuckerExpertWrapper(nn.Module):
#     def __init__(self, group_module, local_idx):
#         super().__init__()
#         self.group_module, self.local_idx = group_module, local_idx

#     def forward(self, x):
#         indices = torch.full((x.shape[0],), self.local_idx, dtype=torch.long, device=x.device)
#         return self.group_module(x, indices)

# # ==========================================
# # 2. 推理辅助逻辑 (采样)
# # ==========================================
# def apply_sampling(logits, temperature, top_p, repetition_penalty, batch_ids):
#     if repetition_penalty != 1.0:
#         for i in range(logits.shape[0]):
#             for token_id in set(batch_ids[i]):
#                 if logits[i, token_id] > 0:
#                     logits[i, token_id] /= repetition_penalty
#                 else:
#                     logits[i, token_id] *= repetition_penalty
#     if temperature > 0:
#         logits = logits / temperature
#     if top_p < 1.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
#         sorted_indices_to_remove = cumulative_probs > top_p
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0
#         for i in range(logits.shape[0]):
#             indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
#             logits[i, indices_to_remove] = float('-inf')
#     return logits

# # ==========================================
# # 3. 核心：并行手术 + 顺序装箱流水线
# # ==========================================
# def pipeline_loading_and_surgery(model, start_idx, args):
#     """
#     CPU 多核并行执行 Tucker 变换，但主线程严格按物理层序搬运到 NPU。
#     """
#     num_layers = len(model.model.layers)
#     num_groups, experts_per_group = 8, 32
#     RESERVED, main_npu = 6.0, "npu:0" # 预留 6GB 缓冲区
    
#     # --- 生产者：后台并行执行手术 ---
#     def producer_task(idx_in_module, layer):
#         real_idx = start_idx + idx_in_module
#         try:
#             if real_idx in args.layers:
#                 for g_idx in range(num_groups):
#                     path = os.path.join(args.whitening_dir, f"layer_{real_idx}_group_{g_idx}.pt")
#                     if not os.path.exists(path): continue
#                     data = torch.load(path, map_location='cpu', weights_only=True)
#                     for proj in ['gate_proj', 'up_proj', 'down_proj']:
#                         gm = TuckerGroupLinear(data[proj]['core'], data[proj]['factors'])
#                         for l_idx in range(experts_per_group):
#                             g_exp = g_idx * experts_per_group + l_idx
#                             if g_exp < len(layer.mlp.experts):
#                                 setattr(layer.mlp.experts[g_exp], proj, TuckerExpertWrapper(gm, l_idx))
#             return True
#         except Exception as e:
#             logging.error(f"L{real_idx} 手术失败: {e}")
#             return False

#     # 提交所有手术任务到线程池
#     executor = ThreadPoolExecutor(max_workers=8)
#     futures = [executor.submit(producer_task, i, model.model.layers[i]) for i in range(num_layers)]

#     # --- 消费者：主线程顺序分发 ---
#     logging.info("-" * 105)
#     logging.info(f"{'Layer':<6} | {'Target':<8} | {'Type':<8} | {'Est. Usage(GB)':<15} | {'Status':<8}")
#     logging.info("-" * 105)

#     curr_card = 0
#     last_active = main_npu
#     if args.node_rank == 0:
#         model.model.embed_tokens.to(main_npu)
#         if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu)

#     for i in range(num_layers):
#         real_idx = start_idx + i
#         # 核心：等待当前层手术完成，确保顺序
#         futures[i].result() 
        
#         layer = model.model.layers[i]
#         l_type = "Tucker" if real_idx in args.layers else "Normal"
#         req = 22.0 if l_type == "Normal" else 5.5
#         success = False
        
#         while curr_card < 8:
#             target = f"npu:{curr_card}"
#             free_mem = torch.npu.mem_get_info(target)[0] / (1024**3)
#             if free_mem > (req + RESERVED):
#                 try:
#                     layer.to(target)
#                     torch.npu.empty_cache()
#                     logging.info(f"L{real_idx:02d}    | {target:<8} | {l_type:<8} | {req:>14.2f} | SUCCESS")
#                     last_active, success = target, True
#                     break
#                 except:
#                     curr_card += 1
#             else:
#                 curr_card += 1
        
#         if not success:
#             layer.to("cpu")
#             logging.warning(f"L{real_idx:02d}    | {'cpu':<8} | {l_type:<8} | {'-':>14} | OFFLOAD")

#     if args.node_rank == 1:
#         model.model.norm.to(last_active)
#         if hasattr(model, "lm_head"): model.lm_head.to(last_active)

#     logging.info("-" * 105)
#     executor.shutdown(wait=True)

# # ==========================================
# # 4. 分布式推理循环 (无缓存重算模式)
# # ==========================================
# def run_no_cache_distributed(model, tokenizer, args, prompts):
#     config = model.config
#     batch_size = len(prompts)
#     total_start_t = time.time()
    
#     try:
#         with torch.no_grad():
#             inputs = tokenizer(prompts, return_tensors="pt", padding=True)
#             batch_ids = inputs.input_ids.tolist()
            
#             for step in range(args.max_new_tokens):
#                 curr_ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
#                 seq_len = curr_ids_tensor.shape[1]
#                 pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
#                 mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1).view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1).to(torch.bfloat16)

#                 if args.node_rank == 0:
#                     # Node 0 计算
#                     h = model.model.embed_tokens(curr_ids_tensor.to("npu:0"))
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
#                     # 发送 Hidden States
#                     dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
#                     # 接收新 Token
#                     new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
#                     dist.recv(tensor=new_ids_dev, src=1)
#                     for i in range(batch_size): batch_ids[i].append(new_ids_dev[i].item())
                    
#                     if args.stream: print(tokenizer.decode([batch_ids[0][-1]]), end="", flush=True)

#                 else:
#                     # Node 1 计算
#                     h_recv = torch.zeros((batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16, device="npu:0")
#                     dist.recv(tensor=h_recv, src=0)
                    
#                     h = h_recv
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
#                     head_dev = next(model.lm_head.parameters()).device
#                     logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
#                     # 采样
#                     if args.do_sample:
#                         logits = apply_sampling(logits, args.temperature, args.top_p, args.repetition_penalty, batch_ids)
#                         next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
#                     else:
#                         next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                    
#                     dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
#                     for i in range(batch_size): batch_ids[i].append(next_tokens[i].item())

#                 if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids): break
            
#             if args.node_rank == 0:
#                 print(f"\n\n推理汇总:\n" + "="*50)
#                 for i, res in enumerate(batch_ids):
#                     print(f"[{i+1}] {tokenizer.decode(res, skip_special_tokens=True)}\n")
#                 print(f"总耗时: {time.time()-total_start_t:.2f}s")
#     except Exception: traceback.print_exc()

# # ==========================================
# # 5. 主程序入口
# # ==========================================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--whitening_dir", type=str, required=True)
#     parser.add_argument("--layers", type=int, nargs='+', default=list(range(3, 61)))
#     parser.add_argument("--node_rank", type=int, default=0)
#     parser.add_argument("--master_addr", type=str, default="10.120.72.45")
#     parser.add_argument("--max_new_tokens", type=int, default=50)
#     parser.add_argument("--do_sample", action="store_true")
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--top_p", type=float, default=0.9)
#     parser.add_argument("--repetition_penalty", type=float, default=1.1)
#     parser.add_argument("--stream", action="store_true")
#     args = parser.parse_args()

#     # 稳定性环境变量
#     os.environ['MASTER_ADDR'] = args.master_addr
#     os.environ['MASTER_PORT'] = '29506'
#     os.environ['HCCL_CONNECT_TIMEOUT'] = '7200'
#     os.environ['HCCL_EXEC_TIMEOUT'] = '7200'
#     os.environ['HCCL_TCP_KEEP_ALIVE_ENABLE'] = '1'
#     os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'

#     # 分布式初始化
#     dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2, timeout=timedelta(seconds=7200))
#     torch.npu.set_device("npu:0")

#     # 模型加载
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
    
#     # 负载均衡切分 (Node 0 负责 0-26, Node 1 负责 27-61)
#     mid = 29
#     all_l = model.model.layers
#     if args.node_rank == 0:
#         model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid)]), 0
#     else:
#         model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))]), mid
#     del all_l
#     gc.collect()

#     # 执行重构后的顺序装箱流水线
#     pipeline_loading_and_surgery(model, start, args)

#     logging.info(f"Node {args.node_rank} 准备就绪，同步对端...")
#     dist.barrier()
#     logging.info("同步成功，启动推理。")

#     run_no_cache_distributed(model, tokenizer, args, ["什么是人工智能？"])

# if __name__ == "__main__":
#     main()

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch_npu
# import argparse
# import logging
# import gc
# import time
# import traceback
# from concurrent.futures import ThreadPoolExecutor
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm

# # 日志配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# # ==========================================
# # 1. Tucker 组件
# # ==========================================
# class TuckerGroupLinear(nn.Module):
#     def __init__(self, core, factors):
#         super().__init__()
#         self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
#         self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
#         with torch.no_grad():
#             # 索引校验: er (U_E) + rud (core) -> eud (merged)
#             w_low_val = torch.einsum('er,rud->eud', factors[0].to(torch.bfloat16), core.to(torch.bfloat16)).contiguous()
#             self.register_buffer("W_low", w_low_val)
            
#     def forward(self, x, expert_indices):
#         x = torch.matmul(x, self.U_in)
#         if expert_indices.dim() == 1 and (expert_indices == expert_indices[0]).all():
#             x = torch.matmul(x, self.W_low[expert_indices[0]].transpose(-1, -2))
#         else:
#             x = torch.bmm(x.unsqueeze(1), self.W_low[expert_indices].transpose(1, 2)).squeeze(1)
#         return torch.matmul(x, self.U_out.T)

# class TuckerExpertWrapper(nn.Module):
#     def __init__(self, group_module, local_idx):
#         super().__init__()
#         self.group_module, self.local_idx = group_module, local_idx
#     def forward(self, x):
#         indices = torch.full((x.shape[0],), self.local_idx, dtype=torch.long, device=x.device)
#         return self.group_module(x, indices)

# # ==========================================
# # 2. 核心功能函数 (并行装箱 + 显存分发)
# # ==========================================

# def init_resources(model_path):
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
#     # 架构加载到 CPU
#     model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
#     return tokenizer, model

# @torch.no_grad()
# def parallel_tucker_surgery(model, start_idx, args):
#     """ 利用多核 CPU 并行替换权重，显著提升启动速度 """
#     num_groups, experts_per_group = 8, 32
#     def process_layer(layer_data):
#         real_idx, layer = layer_data
#         if real_idx not in args.layers: return
#         for g_idx in range(num_groups):
#             path = os.path.join(args.whitening_dir, f"layer_{real_idx}_group_{g_idx}.pt")
#             if not os.path.exists(path): continue
#             data = torch.load(path, map_location='cpu', weights_only=True)
#             for proj in ['gate_proj', 'up_proj', 'down_proj']:
#                 gm = TuckerGroupLinear(data[proj]['core'], data[proj]['factors'])
#                 for l_idx in range(experts_per_group):
#                     g_exp = g_idx * experts_per_group + l_idx
#                     if g_exp < len(layer.mlp.experts):
#                         setattr(layer.mlp.experts[g_exp], proj, TuckerExpertWrapper(gm, l_idx))
    
#     layers = [(start_idx + i, l) for i, l in enumerate(model.model.layers)]
#     logging.info(f"==> 启动并行装箱手术 (Workers=8)...")
#     with ThreadPoolExecutor(max_workers=8) as ex:
#         list(tqdm(ex.map(process_layer, layers), total=len(layers), desc="Parallel Surgery"))

# def distribute_logic(model, start_idx, args):
#     """ 显存分发：预留8G，溢出顺延至下一张卡或 CPU """
#     RESERVED, main_npu = 8.0, "npu:0"
#     model.model.embed_tokens.to(main_npu)
#     if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu)
    
#     curr_card, last_active = 0, main_npu
#     logging.info("-" * 105)
#     logging.info(f"{'Layer':<6} | {'Target':<8} | {'Type':<8} | {'Usage(GB)':<10} | {'Status':<8}")
#     logging.info("-" * 105)

#     for idx, layer in enumerate(model.model.layers):
#         real_idx = start_idx + idx
#         l_type = "Tucker" if real_idx in args.layers else "Normal"
#         req = 22.0 if l_type == "Normal" else 5.5
#         success = False
#         while curr_card < 8:
#             target = f"npu:{curr_card}"
#             if (torch.npu.mem_get_info(target)[0] / 1024**3) > (req + RESERVED):
#                 try:
#                     layer.to(target)
#                     torch.npu.empty_cache()
#                     usage = req # 粗略估算显示
#                     logging.info(f"L{real_idx:02d}    | {target:<8} | {l_type:<8} | {usage:>9.2f} | SUCCESS")
#                     last_active, success = target, True
#                     break
#                 except: curr_card += 1
#             else: curr_card += 1
#         if not success:
#             layer.to("cpu")
#             logging.warning(f"L{real_idx:02d}    | {'cpu':<8} | {l_type:<8} | {'-':>9} | OFFLOAD")

#     model.model.norm.to(last_active)
#     if hasattr(model, "lm_head"): model.lm_head.to(last_active)
#     logging.info("-" * 105)
#     return last_active

# # ==========================================
# # 3. 核心推理：全量重算模式 (禁用 KV Cache)
# # ==========================================

# def run_no_cache_distributed(model, tokenizer, args, prompts):
#     """ 禁用 KV Cache 的全量重算推理：最稳定、bf16 传输加速 """
#     config = model.config
#     batch_size = len(prompts)
#     total_start_t = time.time()
    
#     try:
#         with torch.no_grad():
#             inputs = tokenizer(prompts, return_tensors="pt", padding=True)
#             batch_ids = inputs.input_ids.tolist()
            
#             logging.info(f"==> [无缓存模式] 启动推理, bf16 跨机传输...")
            
#             for step in range(args.max_new_tokens):
#                 step_start_t = time.time()
                
#                 # 每一轮构造全量张量
#                 curr_ids_tensor = torch.tensor(batch_ids, dtype=torch.long)
#                 seq_len = curr_ids_tensor.shape[1]
                
#                 # 构造全量位置和因果掩码
#                 pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
#                 mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), 1).view(1, 1, seq_len, seq_len).expand(batch_size, 1, -1, -1).to(torch.bfloat16)

#                 if args.node_rank == 0:
#                     # --- Node 0 ---
#                     h = model.model.embed_tokens(curr_ids_tensor.to("npu:0"))
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         # 显式禁用缓存
#                         h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
#                     # 发送全量 Hidden States，转 bf16 极大缓解网络压力
#                     dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
#                     # 接收最后一个 Token ID
#                     new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
#                     dist.recv(tensor=new_ids_dev, src=1)
                    
#                     for i in range(batch_size): batch_ids[i].append(new_ids_dev[i].item())
#                     if args.stream:
#                         print(tokenizer.decode([batch_ids[0][-1]]), end="", flush=True)

#                 else:
#                     # --- Node 1 ---
#                     # 接收 bf16 Buffer
#                     r_shape = (batch_size, seq_len, config.hidden_size)
#                     h_recv = torch.zeros(r_shape, dtype=torch.bfloat16, device="npu:0")
#                     dist.recv(tensor=h_recv, src=0)
                    
#                     h = h_recv
#                     for layer in model.model.layers:
#                         dev = next(layer.parameters()).device
#                         h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), use_cache=False)[0]
                    
#                     # 采样最新 Token
#                     head_dev = next(model.lm_head.parameters()).device
#                     logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
#                     if args.temperature > 0:
#                         probs = torch.softmax(logits / args.temperature, dim=-1)
#                         next_tokens = torch.multinomial(probs, num_samples=1)
#                     else:
#                         next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                    
#                     dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
#                     for i in range(batch_size): batch_ids[i].append(next_tokens[i].item())

#                 if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids): break

#             if args.node_rank == 0:
#                 print("\n\n" + "="*50 + "\n推理汇总:\n" + "="*50)
#                 for i, res in enumerate(batch_ids):
#                     print(f"[{i+1}] {tokenizer.decode(res, skip_special_tokens=True)}\n")
#                 print(f"总耗时: {time.time()-total_start_t:.2f}s")

#     except Exception: traceback.print_exc()

# # ==========================================
# # 4. 执行入口
# # ==========================================

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--whitening_dir", type=str, required=True)
#     parser.add_argument("--layers", type=int, nargs='+', default=range(3, 61))
#     parser.add_argument("--node_rank", type=int, default=0)
#     parser.add_argument("--master_addr", type=str, default="10.120.72.45")
#     parser.add_argument("--max_new_tokens", type=int, default=50)
#     parser.add_argument("--temperature", type=float, default=0.7)
#     parser.add_argument("--stream", action="store_true")
#     return parser.parse_args()

# def main():
#     args = get_args()
#     os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = args.master_addr, '29506'
#     os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'

#     dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2)
#     torch.npu.set_device("npu:0")

#     tokenizer, model = init_resources(args.model_path)
    
#     # 负载切分：Node 0 (0-26层)，Node 1 (27-61层)
#     mid = 27
#     all_l = model.model.layers
#     if args.node_rank == 0:
#         model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid)]), 0
#     else:
#         model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))]), mid
#     del all_l
#     gc.collect()

#     # 1. 并行装箱手术 (CPU 多核加速)
#     parallel_tucker_surgery(model, start, args)
#     # 2. 显存动态分发
#     distribute_logic(model, start, args)

#     # 3. 稳健推理 (无缓存全量重算)
#     prompts = ["什么是人工智能？"]
#     run_no_cache_distributed(model, tokenizer, args, prompts)

# if __name__ == "__main__":
#     main()





# import os
# import sys
# import gc
# import logging
# import argparse

# # 必须先 import torch 和 torch_npu
# import torch
# import torch_npu

# # 针对 910B 的关键优化配置
# # jit_compile=False 表示使用二进制算子，避免在线编译导致的 FlashAttention 算子缺失或死锁
# torch.npu.set_compile_mode(jit_compile=False) 

# # 继续原有的导入
# from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
# import torch.nn as nn
# import torch.distributed as dist


# # ==============================================================================
# # 1. 定义 Tucker 专家层
# # ==============================================================================

# class TuckerDecomposedExpert(nn.Module):
#     def __init__(self, proj_name, core, factors, device="cpu"):
#         super().__init__()
#         self.proj_name = proj_name
#         # 强制 bf16 确保性能，并移动到对应设备
#         self.u_in = nn.Parameter(factors[1].to(torch.bfloat16).to(device))
#         self.core = nn.Parameter(core.to(torch.bfloat16).to(device))
#         self.u_out = nn.Parameter(factors[2].to(torch.bfloat16).to(device))

#     def forward(self, x, expert_idx_in_group):
#         if x.shape[0] == 0:
#             return x
#         # 1. 输入投影
#         x = torch.matmul(x, self.u_in)
#         # 2. 核心交互 (Batch MatMul)
#         core_slices = self.core[expert_idx_in_group]
#         x = torch.bmm(x.unsqueeze(1), core_slices).squeeze(1)
#         # 3. 输出投影
#         x = torch.matmul(x, self.u_out)
#         return x

# # ==============================================================================
# # 2. 动态替换与局部层装载逻辑 (针对 PP2 优化)
# # ==============================================================================

# def apply_tucker_to_model_partial(model, decomp_dir, active_layers, device="cpu"):
#     """
#     只处理当前 Rank 负责的层范围，节省内存
#     """
#     n_group = 8
#     experts_per_group = 32

#     for layer_idx in active_layers:
#         layer = model.model.layers[layer_idx]
#         if not hasattr(layer.mlp, 'experts'): continue
        
#         # 检查该层文件
#         check_file = os.path.join(decomp_dir, f"layer_{layer_idx}_group_0.pt")
#         if not os.path.exists(check_file):
#             continue

#         logging.info(f"Applying Tucker to Layer {layer_idx}...")
#         for g_idx in range(n_group):
#             decomp_path = os.path.join(decomp_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
#             group_weights = torch.load(decomp_path, map_location="cpu")
            
#             for exp_in_g in range(experts_per_group):
#                 abs_exp_idx = g_idx * experts_per_group + exp_in_g
#                 expert = layer.mlp.experts[abs_exp_idx]
#                 for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
#                     data = group_weights[proj_name]
#                     tucker_mod = TuckerDecomposedExpert(proj_name, data['core'], data['factors'], device)
#                     setattr(expert, proj_name, tucker_mod)
#             del group_weights
#             gc.collect()
#     return model

# # ==============================================================================
# # 3. 主函数 (PP=2 核心逻辑)
# # ==============================================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--decomp_dir", type=str, required=True)
#     parser.add_argument("--local_rank", type=int, default=-1)
#     args = parser.parse_args()

#     # 初始化分布式
#     if args.local_rank != -1:
#         torch.npu.set_device(args.local_rank)
#         dist.init_process_group(backend="hccl")
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#     else:
#         rank, world_size = 0, 1
#         torch.npu.set_device(0)

#     # 1. 确定流水线切分 (PP=2)
#     # 机器0 (Rank 0-7) 处理 0-30 层; 机器1 (Rank 8-15) 处理 31-61 层
#     total_layers = 62
#     mid = 31
#     if rank < 8:
#         active_range = range(0, mid) 
#         neighbor_rank = rank + 8  # 机器0的卡对应发给机器1的卡
#         is_first_stage = True
#     else:
#         active_range = range(mid, total_layers)
#         neighbor_rank = rank - 8
#         is_first_stage = False

#     logging.info(f"Rank {rank}: Handling layers {list(active_range)}")

#     # 2. 使用 Meta Device 创建模型框架 (不占显存)
#     config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
#     with torch.device("meta"):
#         model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16)

#     # 3. 执行局部层重构
#     # 将模型搬运到 NPU 之前，只针对 active_range 进行权重替换
#     model = apply_tucker_to_model_partial(model, args.decomp_dir, active_range, device="cpu")

#     # 将负责的层移动到 NPU
#     for i in active_range:
#         model.model.layers[i].to(torch.npu.current_device())
    
#     # Embedding 和 Norm 需要根据阶段归属
#     if is_first_stage:
#         model.model.embed_tokens.to(torch.npu.current_device())
#     else:
#         model.model.norm.to(torch.npu.current_device())
#         model.lm_head.to(torch.npu.current_device())

#     logging.info(f"Rank {rank} Partial Model Loaded.")

#     # 4. 推理演示 (单步 Handshake)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
#     # 简单的流水线执行流程
#     with torch.no_grad():
#         if is_first_stage:
#             # 机器0：执行
#             text = "Explain the Tucker decomposition."
#             inputs = tokenizer(text, return_tensors="pt").to(torch.npu.current_device())
#             hidden_states = model.model.embed_tokens(inputs['input_ids'])
            
#             # 跑前一半层
#             for i in active_range:
#                 hidden_states = model.model.layers[i](hidden_states)[0]
            
#             # 发送给机器1
#             # 注意：DeepSeek-V3 hidden_size=7168
#             if rank == 0: # 仅由主卡发送，或各卡发送自己的 TP 部分
#                 dist.send(tensor=hidden_states.float().cpu(), dst=8)
#                 logging.info("Rank 0: Handover data sent to Rank 8.")
        
#         else:
#             # 机器1：接收
#             if rank == 8:
#                 # 预分配接收缓存 (根据输入 SeqLen 调整)
#                 recv_buf = torch.empty([1, 6, 7168], dtype=torch.float32) 
#                 dist.recv(tensor=recv_buf, src=0)
#                 hidden_states = recv_buf.to(torch.bfloat16).to(torch.npu.current_device())
#                 logging.info("Rank 8: Data received from Rank 0.")
#             else:
#                 # 其它卡等待同步 (如果做了 TP 需要广播，这里简化处理)
#                 hidden_states = torch.zeros([1, 6, 7168], dtype=torch.bfloat16).to(torch.npu.current_device())

#             # 跑后一半层
#             for i in active_range:
#                 hidden_states = model.model.layers[i](hidden_states)[0]
            
#             # 输出结果
#             if rank == 8:
#                 logits = model.lm_head(model.model.norm(hidden_states))
#                 next_token = torch.argmax(logits[:, -1, :], dim=-1)
#                 print(f"Rank 8 Prediction Token ID: {next_token.item()}")
#                 print(f"Decoded: {tokenizer.decode(next_token)}")

# if __name__ == "__main__":
#     main()



# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch_npu
# import argparse
# import logging
# import gc
# import traceback
# from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# # 设置日志
# logging.basicConfig(level=logging.INFO)

# # ==========================================
# # 1. 定义 Tucker 分组推理组件 (基于您的旧代码优化)
# # ==========================================
# # class TuckerGroupLinear(nn.Module):
# #     def __init__(self, core, factors):
# #         super().__init__()
# #         # 使用 register_buffer 确保 to(device) 时自动搬运
# #         self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
# #         self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        
# #         with torch.no_grad():
# #             temp_U_E = factors[0].to(torch.bfloat16)
# #             temp_core = core.to(torch.bfloat16)
# #             # 提前合并核心，维度变为 [E, R_O, R_I]
# #             w_low_val = torch.einsum('er,roi->eoi', temp_U_E, temp_core).contiguous()
# #             self.register_buffer("W_low", w_low_val)

# class TuckerGroupLinear(nn.Module):
#     def __init__(self, core, factors):
#         super().__init__()
#         self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
#         self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        
#         with torch.no_grad():
#             temp_U_E = factors[0].to(torch.bfloat16)
#             temp_core = core.to(torch.bfloat16)
#             # 严格按照参考代码的维度索引合并：er (U_E) + rud (core) -> eud (merged)
#             # 这里的 r 是压缩秩，u 是输出维度，d 是输入维度
#             w_low_val = torch.einsum('er,rud->eud', temp_U_E, temp_core).contiguous()
#             self.register_buffer("W_low", w_low_val)
            
#     def forward(self, x, expert_indices):
#         # x: [N, D_in], expert_indices: [N]
#         x = torch.matmul(x, self.U_in)
        
#         # 兼容 DeepSeek 的批量推理逻辑
#         if expert_indices.dim() == 1 and (expert_indices == expert_indices[0]).all():
#             core_weight = self.W_low[expert_indices[0]]
#             x = torch.matmul(x, core_weight.transpose(-1, -2))
#         else:
#             current_cores = self.W_low[expert_indices]
#             x = torch.bmm(x.unsqueeze(1), current_cores.transpose(1, 2)).squeeze(1)
            
#         return torch.matmul(x, self.U_out.T)

# class TuckerExpertWrapper(nn.Module):
#     def __init__(self, group_module, local_idx):
#         super().__init__()
#         self.group_module = group_module
#         self.local_idx = local_idx

#     def forward(self, x):
#         # 构造索引 Tensor 以适配 GroupLinear
#         indices = torch.full((x.shape[0],), self.local_idx, dtype=torch.long, device=x.device)
#         return self.group_module(x, indices)

# # ==========================================
# # 2. 单层在位替换函数
# # ==========================================
# @torch.no_grad()
# def reconstruct_single_layer(layer, layer_idx, args):
#     num_groups = 8
#     experts_per_group = 32
#     for g_idx in range(num_groups):
#         decomp_file = os.path.join(args.whitening_dir, f"layer_{layer_idx}_group_{g_idx}.pt")
#         if not os.path.exists(decomp_file): continue
        
#         group_data = torch.load(decomp_file, map_location='cpu', weights_only=True)
        
#         for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
#             data = group_data[proj_name]
#             # 创建共享的组模块
#             group_module = TuckerGroupLinear(data['core'], data['factors'])
            
#             for local_idx in range(experts_per_group):
#                 global_idx = g_idx * experts_per_group + local_idx
#                 if global_idx < len(layer.mlp.experts):
#                     new_expert = TuckerExpertWrapper(group_module, local_idx)
#                     # 替换原 Linear
#                     setattr(layer.mlp.experts[global_idx], proj_name, new_expert)
#     gc.collect()

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--whitening_dir", type=str, required=True)
#     parser.add_argument("--layers", type=int, nargs='+', default=range(3, 61))
#     parser.add_argument("--node_rank", type=int, default=0)
#     parser.add_argument("--master_addr", type=str, default="10.120.72.45")
#     return parser.parse_args()

# # ==========================================
# # 3. 主程序
# # ==========================================
# def main():
#     args = get_args()
    
#     # 1. 环境与通信初始化
#     os.environ['MASTER_ADDR'] = args.master_addr
#     os.environ['MASTER_PORT'] = '29506'
#     os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'
    
#     main_npu_dev = "npu:0" 

#     logging.info(f"==> Node {args.node_rank} 初始化 HCCL 通信...")
#     dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2)
#     torch.npu.set_device(main_npu_dev)

#     # 2. 模型加载 (CPU)
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
#         low_cpu_mem_usage=True, device_map={"": "cpu"}
#     )
#     config = model.config

#     # 3. Pipeline 切分
#     all_layers = model.model.layers
#     # mid_point = len(all_layers) // 2
#     mid_point=27
#     if args.node_rank == 0:
#         model.model.layers = nn.ModuleList([all_layers[j] for j in range(0, mid_point)])
#         start_idx = 0
#     else:
#         model.model.layers = nn.ModuleList([all_layers[j] for j in range(mid_point, len(all_layers))])
#         start_idx = mid_point
    
#     del all_layers
#     gc.collect()

#     # 4. 动态显存部署逻辑：自动避让 OOM
#     model.model.embed_tokens.to(main_npu_dev)
#     if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu_dev)

#     logging.info(f"==> 开始动态 NPU 部署 (Node {args.node_rank})")
#     logging.info("-" * 95)
#     logging.info(f"{'Layer':<6} | {'Target':<8} | {'Type':<8} | {'Usage(GB)':<10} | {'Remain(GB)':<10}")
#     logging.info("-" * 95)

#     current_card_idx = 0
#     last_active_dev = main_npu_dev

#     for idx, layer in enumerate(model.model.layers):
#         real_idx = start_idx + idx
        
#         # 预手术：CPU 侧执行 Tucker 替换
#         l_type = "Normal"
#         if real_idx in args.layers:
#             reconstruct_single_layer(layer, real_idx, args)
#             l_type = "Tucker"
        
#         success = False
#         while current_card_idx < 8:
#             target_dev = f"npu:{current_card_idx}"
#             free_byte, _ = torch.npu.mem_get_info(target_dev)
#             free_gb = free_byte / (1024**3)
            
#             # 阈值：Normal 层约 22GB, Tucker 层约 5.5GB，额外留 8GB 冗余
#             required = 22.0 if l_type == "Normal" else 5.5
            
#             if free_gb > (required + 8.0):
#                 try:
#                     layer.to(target_dev)
#                     torch.npu.empty_cache()
#                     new_free, _ = torch.npu.mem_get_info(target_dev)
#                     usage_actual = free_gb - (new_free / 1024**3)
#                     logging.info(f"L{real_idx:02d}    | {target_dev:<8} | {l_type:<8} | {usage_actual:>9.2f} | {new_free/1024**3:>9.2f}")
#                     last_active_dev = target_dev
#                     success = True
#                     break 
#                 except RuntimeError as e:
#                     if "out of memory" in str(e).lower():
#                         current_card_idx += 1
#                         torch.npu.empty_cache()
#                         continue
#                     else: raise e
#             else:
#                 current_card_idx += 1
        
#         if not success:
#             logging.error(f"❌ OOM：8张NPU显存已耗尽，无法放置 L{real_idx}。请增加压缩比例！")
#             sys.exit(1)

#     # 结尾组件放在流转的最后一站
#     model.model.norm.to(last_active_dev)
#     if hasattr(model, "lm_head"):
#         model.lm_head.to(last_active_dev)
#     logging.info("-" * 95)

#     # 5. 修复了 Pos_ids 和 Mask 的流式长文本生成
#     max_new_tokens = 50
    
#     try:
#         with torch.no_grad():
#             test_input = "什么是人工智能？请简短回答。"
#             inputs = tokenizer(test_input, return_tensors="pt")
#             # 维护完整的 Sequence ID 列表
#             input_ids_list = inputs.input_ids[0].tolist()
#             batch_size = 1

#             logging.info("==> [无缓存模式] 启动生成...")
#             for step in range(max_new_tokens):
#                 # 核心区别：每一轮都把 input_ids_list 全部转为 Tensor
#                 curr_seq_len = len(input_ids_list)
#                 input_to_run = torch.tensor([input_ids_list], dtype=torch.long)
                
#                 # Position IDs 必须是全量序列 [0, 1, ..., curr_seq_len-1]
#                 pos_ids = torch.arange(curr_seq_len).unsqueeze(0)
                
#                 # Mask 必须是全量的下三角 Causal Mask [1, 1, seq, seq]
#                 mask = torch.full((curr_seq_len, curr_seq_len), float("-inf"))
#                 mask = torch.triu(mask, diagonal=1).view(1, 1, curr_seq_len, curr_seq_len)
#                 mask = mask.to(torch.bfloat16)

#                 if args.node_rank == 0:
#                     # --- Node 0 推理 ---
#                     h = model.model.embed_tokens(input_to_run.to(main_npu_dev))
#                     for i, layer in enumerate(model.model.layers):
#                         dev = next(layer.parameters()).device
#                         # 注意：use_cache=False, 不传入 past_key_value
#                         outputs = layer(
#                             h.to(dev), 
#                             attention_mask=mask.to(dev), 
#                             position_ids=pos_ids.to(dev), 
#                             use_cache=False
#                         )
#                         h = outputs[0]
                    
#                     # 发送全量 Sequence 的 Hidden States
#                     dist.send(tensor=h.to(main_npu_dev).contiguous(), dst=1)
                    
#                     # 接收最后一个 Token ID
#                     new_id_dev = torch.zeros((1, 1), dtype=torch.long, device=main_npu_dev)
#                     dist.recv(tensor=new_id_dev, src=1)
                    
#                     new_id = new_id_dev.item()
#                     input_ids_list.append(new_id)
                    
#                     # 实时输出
#                     print(tokenizer.decode([new_id]), end="", flush=True)
#                     if new_id == tokenizer.eos_token_id: break

#                 else:
#                     # --- Node 1 推理 ---
#                     # 接收全量 Hidden States，形状随长度增长
#                     r_shape = (batch_size, curr_seq_len, config.hidden_size)
#                     h_recv = torch.zeros(r_shape, dtype=torch.bfloat16, device=main_npu_dev)
#                     dist.recv(tensor=h_recv, src=0)
                    
#                     h = h_recv
#                     for i, layer in enumerate(model.model.layers):
#                         dev = next(layer.parameters()).device
#                         outputs = layer(
#                             h.to(dev), 
#                             attention_mask=mask.to(dev), 
#                             position_ids=pos_ids.to(dev), 
#                             use_cache=False
#                         )
#                         h = outputs[0]
                    
#                     # 只需要取最后一个 Position 的 Logits 来预测
#                     head_dev = next(model.lm_head.parameters()).device
#                     h_final = model.model.norm(h.to(head_dev))
#                     logits = model.lm_head(h_final)
                    
#                     # 获取最新一个 token
#                     next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
#                     # 发送回 Node 0
#                     dist.send(tensor=next_token_id.to(main_npu_dev).contiguous(), dst=0)
                    
#                     # Node 1 也同步列表以维持下一轮循环的 curr_seq_len
#                     input_ids_list.append(next_token_id.item())
#                     if next_token_id.item() == tokenizer.eos_token_id: break

#             if args.node_rank == 0:
#                 logging.info("\n==> [DONE] 无缓存模式推理完成。")

#     except Exception:
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()

