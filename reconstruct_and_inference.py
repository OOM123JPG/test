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
