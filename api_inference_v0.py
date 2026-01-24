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
import uuid
import asyncio
import traceback
from datetime import timedelta
from typing import List, Dict, Any, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# ==========================================
# 0. 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Tucker 组件
# ==========================================
class TuckerGroupLinear(nn.Module):
    def __init__(self, core, factors):
        super().__init__()
        self.register_buffer("U_out", factors[1].to(torch.bfloat16).contiguous())
        self.register_buffer("U_in", factors[2].to(torch.bfloat16).contiguous())
        with torch.no_grad():
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
# 2. 采样辅助
# ==========================================
def apply_sampling(logits, temperature, top_p, repetition_penalty, batch_ids):
    if repetition_penalty != 1.0:
        for i in range(logits.shape[0]):
            for token_id in set(batch_ids[i]):
                if logits[i, token_id] > 0: logits[i, token_id] /= repetition_penalty
                else: logits[i, token_id] *= repetition_penalty
    if temperature > 0: logits = logits / temperature
    else: return logits
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(logits.shape[0]):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = float('-inf')
    return logits

# ==========================================
# 3. 分布式引擎
# ==========================================
class DistributedInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.req_queue = asyncio.Queue()
        self.results = {}
        self.events = {}
        self.is_running = True
        self.start_idx = 0

    def setup(self):
        dist.init_process_group(backend='hccl', rank=self.args.node_rank, world_size=2, timeout=timedelta(seconds=7200))
        torch.npu.set_device("npu:0")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        full_model = AutoModelForCausalLM.from_pretrained(self.args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
        
        mid = 29
        all_l = full_model.model.layers
        if self.args.node_rank == 0:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid)])
            self.start_idx = 0
        else:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))])
            self.start_idx = mid
        self.model = full_model
        
        self._pipeline_surgery_and_loading()
        dist.barrier()
        logger.info(f"Node {self.args.node_rank} Ready.")

    def _pipeline_surgery_and_loading(self):
        num_layers = len(self.model.model.layers)
        RESERVED, main_npu = 12, "npu:0"
        
        def surgery_task(idx_in_module, layer):
            real_idx = self.start_idx + idx_in_module
            if hasattr(layer.mlp, "experts") and real_idx in self.args.layers:
                for g_idx in range(8):
                    path = os.path.join(self.args.whitening_dir, f"layer_{real_idx}_group_{g_idx}.pt")
                    if os.path.exists(path):
                        data = torch.load(path, map_location='cpu', weights_only=True)
                        for proj in ['gate_proj', 'up_proj', 'down_proj']:
                            gm = TuckerGroupLinear(data[proj]['core'], data[proj]['factors'])
                            for l_idx in range(32):
                                g_exp = g_idx * 32 + l_idx
                                if g_exp < len(layer.mlp.experts):
                                    setattr(layer.mlp.experts[g_exp], proj, TuckerExpertWrapper(gm, l_idx))
            return True

        executor = ThreadPoolExecutor(max_workers=8)
        futures = [executor.submit(surgery_task, i, self.model.model.layers[i]) for i in range(num_layers)]

        curr_card = 0
        last_active = main_npu
        if self.args.node_rank == 0: self.model.model.embed_tokens.to(main_npu)
        
        logger.info(f"Node {self.args.node_rank} 每一层装箱详情:")
        for i in range(num_layers):
            real_idx = self.start_idx + i
            futures[i].result()
            layer = self.model.model.layers[i]
            
            # 计算这一层占用的字节数
            layer_mem = sum(p.numel() * p.element_size() for p in layer.parameters()) + \
                        sum(b.numel() * b.element_size() for b in layer.buffers())
            req_gb = layer_mem / (1024**3)
            
            success = False
            while curr_card < 8:
                target = f"npu:{curr_card}"
                free_gb = torch.npu.mem_get_info(target)[0] / (1024**3)
                if free_gb > (req_gb + RESERVED):
                    try:
                        layer.to(target); torch.npu.empty_cache()
                        # 更新当前卡剩余显存
                        post_free_gb = torch.npu.mem_get_info(target)[0] / (1024**3)
                        logger.info(f" - Layer {real_idx:02d}: {target} | Size: {req_gb:.3f} GB | NPU Free: {post_free_gb:.3f} GB")
                        last_active, success = target, True
                        break
                    except: curr_card += 1
                else: curr_card += 1
            if not success: logger.warning(f" - Layer {real_idx:02d}: FAILED to load on NPU, staying on CPU")

        if self.args.node_rank == 1:
            self.model.model.norm.to(last_active)
            self.model.lm_head.to(last_active)
            
        total_params = sum(p.numel() for p in self.model.parameters())
        total_buffers = sum(b.numel() for b in self.model.buffers())
        node_total = (total_params + total_buffers) / 1e9
        
        logger.info(f"========================================")
        logger.info(f"Node {self.args.node_rank} 参数统计:")
        logger.info(f" - 模型参数量 (Trainable): {total_params / 1e9:.3f} B")
        logger.info(f" - 缓存数据量 (Tucker Buffers): {total_buffers / 1e9:.3f} B")
        logger.info(f" - 节点显存总负载: {node_total:.3f} B")
        logger.info(f"========================================")
        
        executor.shutdown(wait=True); gc.collect(); torch.npu.empty_cache()

    async def run_loop(self):
        while self.is_running:
            active_reqs = []
            if self.args.node_rank == 0:
                try:
                    req = await asyncio.wait_for(self.req_queue.get(), timeout=0.1)
                    active_reqs.append(req)
                    while not self.req_queue.empty() and len(active_reqs) < 16: 
                        active_reqs.append(self.req_queue.get_nowait())
                except asyncio.TimeoutError:
                    dist.broadcast(torch.tensor([0], dtype=torch.long, device="npu:0"), src=0); continue

                dist.broadcast(torch.tensor([len(active_reqs)], dtype=torch.long, device="npu:0"), src=0)
                p_args = active_reqs[0]
                task_code = 1 if p_args['type'] == 'loglikelihood' else 0
                params = torch.tensor([task_code, p_args['max_tokens'], p_args['temp'], p_args['top_p'], p_args.get('cont_len', 0)], device="npu:0", dtype=torch.float32)
                dist.broadcast(params, src=0)
                
                inputs = self.tokenizer([r['prompt'] for r in active_reqs], return_tensors="pt", padding=True).to("npu:0")
                dist.broadcast(torch.tensor([inputs.input_ids.shape[1]], dtype=torch.long, device="npu:0"), src=0)
                dist.broadcast(inputs.input_ids, src=0)
                input_ids = inputs.input_ids
            else:
                signal = torch.zeros([1], dtype=torch.long, device="npu:0")
                dist.broadcast(signal, src=0)
                if signal.item() == 0: continue
                batch_size = int(signal.item())
                params = torch.zeros([5], device="npu:0")
                dist.broadcast(params, src=0)
                task_code, p_max, p_temp, p_top, p_cont = params.tolist()
                p_args = {'max_tokens': int(p_max), 'temp': p_temp, 'top_p': p_top, 'cont_len': int(p_cont)}
                seq_len_t = torch.zeros([1], dtype=torch.long, device="npu:0")
                dist.broadcast(seq_len_t, src=0)
                input_ids = torch.zeros((batch_size, int(seq_len_t.item())), dtype=torch.long, device="npu:0")
                dist.broadcast(input_ids, src=0)

            if task_code == 1: results = self._execute_loglikelihood_sync(input_ids, len(input_ids), p_args)
            else: results = self._execute_inference_sync(input_ids, len(input_ids), p_args)
            
            if self.args.node_rank == 0:
                for i, req in enumerate(active_reqs):
                    rid = req['id']; self.results[rid] = results[i]
                    if rid in self.events: self.events[rid].set()

    # def _execute_loglikelihood_sync(self, input_ids, batch_size, p_args):
    #     cont_len = int(p_args['cont_len'])
    #     with torch.no_grad():
    #         q_len = input_ids.shape[1]
    #         padding_mask = (input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
    #         causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
    #         mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - padding_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
            
    #         if self.args.node_rank == 0:
    #             h = self.model.model.embed_tokens(input_ids)
    #             for layer in self.model.model.layers: 
    #                 h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
    #             dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                
    #             # 接收批次结果
    #             # 注意：这里接收的是一个扁平化的 logprobs 矩阵 [batch, q_len-1]
    #             logprobs_recv = torch.zeros((batch_size, q_len - 1), device="npu:0", dtype=torch.float32)
    #             dist.recv(tensor=logprobs_recv, src=1)
    #             return logprobs_recv.tolist() # 返回给 API 解析
    #         else:
    #             h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
    #             dist.recv(tensor=h_recv, src=0); h = h_recv
    #             for layer in self.model.model.layers: 
    #                 h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
    #             logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))
    #             logprobs = F.log_softmax(logits, dim=-1)
                
    #             # 提取目标的 logprobs
    #             target_ids = input_ids[:, 1:].unsqueeze(-1).to(logprobs.device)
    #             relevant_logprobs = logprobs[:, :-1, :].gather(-1, target_ids).squeeze(-1) # [batch, q_len-1]
                
    #             dist.send(tensor=relevant_logprobs.to("npu:0").to(torch.float32).contiguous(), dst=0)
    #             return relevant_logprobs.tolist()
    
    # 修改 loglikelihood 接口 
    def _execute_loglikelihood_sync(self, input_ids, batch_size, p_args):
        with torch.no_grad():
            q_len = input_ids.shape[1]
            padding_mask = (input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
            causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
            mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - padding_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
            
            if self.args.node_rank == 0:
                h = self.model.model.embed_tokens(input_ids)
                for layer in self.model.model.layers: 
                    h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
                dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                
                # 接收批次结果：[batch, q_len]
                logprobs_recv = torch.zeros((batch_size, q_len), device="npu:0", dtype=torch.float32)
                dist.recv(tensor=logprobs_recv, src=1)
                return logprobs_recv.tolist()
            else:
                h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                dist.recv(tensor=h_recv, src=0); h = h_recv
                for layer in self.model.model.layers: 
                    h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
                
                logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))
                logprobs = F.log_softmax(logits.float(), dim=-1)
                
                target_ids = input_ids[:, 1:].unsqueeze(-1).to(logprobs.device)
                relevant_logprobs = logprobs[:, :-1, :].gather(-1, target_ids).squeeze(-1) 
                
                # 修复处：使用 new_zeros 确保设备严格一致
                padding_val = relevant_logprobs.new_zeros((batch_size, 1))
                final_logprobs = torch.cat([padding_val, relevant_logprobs], dim=-1)
                
                dist.send(tensor=final_logprobs.to("npu:0").to(torch.float32).contiguous(), dst=0)
                return final_logprobs.tolist()

    def _execute_inference_sync(self, input_ids, batch_size, p_args):
        batch_ids = input_ids.tolist(); past_key_values = DynamicCache(); curr_input_ids = input_ids
        current_mask = (curr_input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
        position_ids = current_mask.cumsum(-1) - 1
        position_ids.masked_fill_(current_mask == 0, 0)
        finished, max_gen = [False] * batch_size, int(p_args['max_tokens'])
        with torch.no_grad():
            for step in range(max_gen):
                is_prefill = (step == 0)
                total_len = curr_input_ids.shape[1] if is_prefill else len(batch_ids[0])
                q_len = curr_input_ids.shape[1]
                if is_prefill:
                    causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
                    mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - current_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
                else: mask = (1.0 - current_mask.view(batch_size, 1, 1, total_len).to(torch.bfloat16)) * -10000.0
                if self.args.node_rank == 0:
                    h = self.model.model.embed_tokens(curr_input_ids)
                    for layer in self.model.model.layers: h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), position_ids=position_ids.to(next(layer.parameters()).device), past_key_value=past_key_values, use_cache=True)[0]
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids_dev, src=1); curr_input_ids = new_ids_dev
                else:
                    h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0); h = h_recv
                    for layer in self.model.model.layers: h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), position_ids=position_ids.to(next(layer.parameters()).device), past_key_value=past_key_values, use_cache=True)[0]
                    logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))[:, -1, :]
                    logits = apply_sampling(logits, p_args['temp'], p_args['top_p'], 1.1, batch_ids)
                    next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1) if p_args['temp'] > 0 else torch.argmax(logits, -1, True)
                    dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0); curr_input_ids = next_tokens.to("npu:0")
                nt_list = curr_input_ids.squeeze(-1).tolist()
                for i in range(batch_size):
                    if not finished[i]:
                        batch_ids[i].append(nt_list[i])
                        if nt_list[i] == self.tokenizer.eos_token_id: finished[i] = True
                if all(finished): break
                current_mask = torch.cat([current_mask, torch.ones((batch_size, 1), device="npu:0")], dim=-1)
                position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_ids]

# ==========================================
# 4. API 路由
# ==========================================
app = FastAPI()
class ChatCompletionRequest(BaseModel): model: str = "deepseek-v3-tucker"; messages: List[Dict]; max_tokens: int = 128; temperature: float = 0.7; top_p: float = 0.9
class CompletionRequest(BaseModel): model: str = "deepseek-v3-tucker"; prompt: Union[str, List[str]]; max_tokens: int = 128; temperature: float = 0.0; top_p: float = 1.0; logprobs: Optional[int] = None; echo: bool = False

# @app.post("/v1/chat/completions")
# async def chat_completions(request: ChatCompletionRequest):
#     req_id = str(uuid.uuid4()); event = asyncio.Event(); engine.events[req_id] = event
#     formatted_prompt = engine.tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
#     await engine.req_queue.put({"type": "chat", "id": req_id, "prompt": formatted_prompt, "max_tokens": request.max_tokens, "temp": request.temperature, "top_p": request.top_p})
#     await event.wait(); res = engine.results.pop(req_id); del engine.events[req_id]
#     return {"id": req_id, "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": res}, "index": 0, "finish_reason": "stop"}]}

# @app.post("/v1/completions")
# async def completions(request: CompletionRequest):
#     req_id_base = str(uuid.uuid4())
#     prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
#     is_ll = (request.logprobs is not None and request.logprobs > 0)
    
#     events, results_ids = [], []
#     for i, p in enumerate(prompts):
#         rid = f"{req_id_base}-{i}"
#         ev = asyncio.Event()
#         engine.events[rid] = ev
#         events.append(ev)
#         results_ids.append(rid)
#         # 默认 cont_len 取全长度以兼容 loglikelihood
#         await engine.req_queue.put({
#             "type": "loglikelihood" if is_ll else "chat", 
#             "id": rid, 
#             "prompt": p, 
#             "max_tokens": request.max_tokens, 
#             "temp": request.temperature, 
#             "top_p": request.top_p, 
#             "cont_len": 2048 # 默认上限
#         })
    
#     # 等待所有并发请求完成
#     await asyncio.gather(*(ev.wait() for ev in events))
    
#     choices = []
#     for i, rid in enumerate(results_ids):
#         # 从结果池中获取推理结果
#         res = engine.results.pop(rid)
#         del engine.events[rid]
        
#         if is_ll:
#             # 这里的 res 是 List[float]，代表序列中每个 token 的 logprobs
#             # 为了修复 max() 报错，top_logprobs 必须包含实际的数值字典
#             choices.append({
#                 "text": prompts[i] if request.echo else "", 
#                 "index": i, 
#                 "logprobs": {
#                     "token_logprobs": res, 
#                     "tokens": ["<token>"] * len(res), 
#                     # 将每一个 token 的概率放入字典，防止客户端 max(top.values()) 报错
#                     "top_logprobs": [{"<token>": val} for val in res] 
#                 }, 
#                 "finish_reason": "stop"
#             })
#         else:
#             # 普通生成任务返回文本结果
#             choices.append({
#                 "text": res, 
#                 "index": i, 
#                 "finish_reason": "stop"
#             })
            
#     return {
#         "id": req_id_base, 
#         "object": "text_completion", 
#         "choices": choices
#     }

# 修改 chat_completions 接口
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    req_id = str(uuid.uuid4()); event = asyncio.Event(); engine.events[req_id] = event

    formatted_prompt = ""
    for msg in request.messages:
        formatted_prompt += f"{msg['role']}: {msg['content']}\n"
    # 如果你连 "user: " 这种前缀都不要，可以直接取最后一条：
    # formatted_prompt = request.messages[-1]['content']
    
    await engine.req_queue.put({
        "type": "chat", 
        "id": req_id, 
        "prompt": formatted_prompt, 
        "max_tokens": request.max_tokens, 
        "temp": request.temperature, 
        "top_p": request.top_p
    })
    await event.wait(); res = engine.results.pop(req_id); del engine.events[req_id]
    
    return {
        "id": req_id, 
        "object": "chat.completion", 
        "choices": [{
            "message": {"role": "assistant", "content": res}, 
            "index": 0, 
            "finish_reason": "stop"
        }]
    }
    
    
# 修改 completions 接口
@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    req_id_base = str(uuid.uuid4())
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    is_ll = (request.logprobs is not None and request.logprobs > 0)
    
    # 1. 明确 Tokenizer 编码后的设备 (建议保持 CPU 以处理字符串转换，但在比较时需注意)
    encoded = engine.tokenizer(prompts, padding=True, return_tensors="pt")
    # 保持在 CPU 上用于 convert_ids_to_tokens 这种 CPU 密集型操作
    all_input_ids = encoded.input_ids 

    events, results_ids = [], []
    for i, p in enumerate(prompts):
        rid = f"{req_id_base}-{i}"
        ev = asyncio.Event(); engine.events[rid] = ev
        events.append(ev); results_ids.append(rid)
        await engine.req_queue.put({
            "type": "loglikelihood" if is_ll else "chat", 
            "id": rid, "prompt": p, 
            "max_tokens": request.max_tokens, 
            "temp": request.temperature, "top_p": request.top_p, 
            "cont_len": 2048
        })
    
    await asyncio.gather(*(ev.wait() for ev in events))
    
    choices = []
    for i, rid in enumerate(results_ids):
        res = engine.results.pop(rid)
        del engine.events[rid]
        
        if is_ll:
            token_ids = all_input_ids[i] # 位于 CPU
            tokens_str = engine.tokenizer.convert_ids_to_tokens(token_ids)
            
            # 2. 这里的逻辑在 CPU 执行，不会报错，但为了严谨，
            # 如果 pad_token_id 很大或涉及复杂 Tensor 操作，建议加上 .to(token_ids.device)
            pad_id = engine.tokenizer.pad_token_id
            
            # 找到第一个非 pad token
            # 直接在 CPU 向量化计算
            first_valid = (token_ids != pad_id).long().argmax().item()
            
            valid_tokens = tokens_str[first_valid:]
            valid_logprobs = res[first_valid:]
            
            top_lp = [{t: lp} for t, lp in zip(valid_tokens, valid_logprobs)]

            choices.append({
                "text": prompts[i] if request.echo else "", 
                "index": i, 
                "logprobs": {
                    "token_logprobs": valid_logprobs, 
                    "tokens": valid_tokens, 
                    "top_logprobs": top_lp
                }, 
                "finish_reason": "stop"
            })
        else:
            choices.append({"text": res, "index": i, "finish_reason": "stop"})
            
    return {"id": req_id_base, "object": "text_completion", "choices": choices}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True); parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--node_rank", type=int, default=0); parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888); parser.add_argument("--layers", type=int, nargs='+', default=list(range(3, 61)))
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr; os.environ['MASTER_PORT'] = '29506'
    os.environ['HCCL_CONNECT_TIMEOUT'] = '7200'; os.environ['HCCL_EXEC_TIMEOUT'] = '7200'
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'
    global engine; engine = DistributedInferenceEngine(args); engine.setup()
    if args.node_rank == 0:
        import threading
        def run_e(): loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); loop.run_until_complete(engine.run_loop())
        threading.Thread(target=run_e, daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else: asyncio.run(engine.run_loop())

if __name__ == "__main__": main()



# 3个修改的版本