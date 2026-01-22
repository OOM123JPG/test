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
# 2. 采样辅助函数
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
# 3. 分布式引擎 (集成脚本版的 pipeline_loading)
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

    def setup(self):
        dist.init_process_group(backend='hccl', rank=self.args.node_rank, world_size=2, timeout=timedelta(days=36500))
        torch.npu.set_device("npu:0")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # 1. 初始加载到 CPU
        full_model = AutoModelForCausalLM.from_pretrained(self.args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
        
        # 2. 物理层切分 (0-28 对 29-61)
        mid = 29
        all_l = full_model.model.layers
        if self.args.node_rank == 0:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid)])
            self.start_idx = 0
        else:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))])
            self.start_idx = mid
        
        self.model = full_model
        # 3. 执行脚本版的高效手术与装箱逻辑
        self._pipeline_surgery_and_loading()
        
        dist.barrier()
        logger.info(f"Node {self.args.node_rank} Ready.")

    def _pipeline_surgery_and_loading(self):
        """完全移植你脚本中的 pipeline_loading_and_surgery 逻辑"""
        num_layers = len(self.model.model.layers)
        RESERVED, main_npu = 6.0, "npu:0"
        
        # 并行手术生产者
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

        # 顺序装箱消费者
        curr_card = 0
        last_active = main_npu
        if self.args.node_rank == 0:
            self.model.model.embed_tokens.to(main_npu)
        
        for i in range(num_layers):
            real_idx = self.start_idx + i
            futures[i].result() # 等待手术完成
            
            layer = self.model.model.layers[i]
            # 根据是否是 Tucker 层预估显存
            req_gb = 5.5 if real_idx in self.args.layers else 22.0
            
            success = False
            while curr_card < 8:
                target = f"npu:{curr_card}"
                free_gb = torch.npu.mem_get_info(target)[0] / (1024**3)
                if free_gb > (req_gb + RESERVED):
                    try:
                        layer.to(target)
                        torch.npu.empty_cache()
                        last_active, success = target, True
                        logger.info(f"Layer {real_idx:02d} -> {target} (Est {req_gb}GB) SUCCESS")
                        break
                    except: curr_card += 1
                else: curr_card += 1
            
            if not success:
                logger.warning(f"Layer {real_idx:02d} -> OFFLOAD TO CPU")

        if self.args.node_rank == 1:
            self.model.model.norm.to(last_active)
            self.model.lm_head.to(last_active)
            
        executor.shutdown(wait=True)
        gc.collect(); torch.npu.empty_cache()

    async def run_loop(self):
        while self.is_running:
            active_reqs = []
            if self.args.node_rank == 0:
                try:
                    req = await asyncio.wait_for(self.req_queue.get(), timeout=0.001)
                    active_reqs.append(req)
                    while not self.req_queue.empty() and len(active_reqs) < 8:
                        active_reqs.append(self.req_queue.get_nowait())
                except asyncio.TimeoutError:
                    dist.broadcast(torch.tensor([0], dtype=torch.long, device="npu:0"), src=0)
                    continue

                batch_size = len(active_reqs)
                logger.info(f"🚀 Packing {batch_size} requests. Type: {active_reqs[0]['type']}. Queue: {self.req_queue.qsize()}")
                
                dist.broadcast(torch.tensor([batch_size], dtype=torch.long, device="npu:0"), src=0)
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
                task_code = int(params[0].item())
                p_args = {'max_tokens': int(params[1].item()), 'temp': params[2].item(), 'top_p': params[3].item(), 'cont_len': int(params[4].item())}
                seq_len_t = torch.zeros([1], dtype=torch.long, device="npu:0")
                dist.broadcast(seq_len_t, src=0)
                input_ids = torch.zeros((batch_size, int(seq_len_t.item())), dtype=torch.long, device="npu:0")
                dist.broadcast(input_ids, src=0)

            if task_code == 1: results = self._execute_loglikelihood_sync(input_ids, batch_size, p_args)
            else: results = self._execute_inference_sync(input_ids, batch_size, p_args)
            
            if self.args.node_rank == 0:
                for i, req in enumerate(active_reqs):
                    rid = req['id']; self.results[rid] = results[i]
                    if rid in self.events: self.events[rid].set()

    def _execute_loglikelihood_sync(self, input_ids, batch_size, p_args):
        cont_len = int(p_args['cont_len'])
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
                logprobs_result = torch.zeros(batch_size, device="npu:0", dtype=torch.float32)
                dist.recv(tensor=logprobs_result, src=1)
                return logprobs_result.tolist()
            else:
                h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                dist.recv(tensor=h_recv, src=0)
                h = h_recv
                for layer in self.model.model.layers:
                    h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
                logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))
                logprobs = F.log_softmax(logits, dim=-1)
                target_ids = input_ids[:, 1:].unsqueeze(-1).to(logprobs.device)
                relevant_logprobs = logprobs[:, :-1, :].gather(-1, target_ids).squeeze(-1)
                batch_res = [relevant_logprobs[i, -cont_len:].sum().item() for i in range(batch_size)]
                dist.send(tensor=torch.tensor(batch_res, device="npu:0", dtype=torch.float32), dst=0)
                return batch_res

    def _execute_inference_sync(self, input_ids, batch_size, p_args):
        batch_ids = input_ids.tolist()
        past_key_values = DynamicCache()
        curr_input_ids = input_ids
        current_padding_mask = (curr_input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
        position_ids = current_padding_mask.cumsum(-1) - 1
        position_ids.masked_fill_(current_padding_mask == 0, 0)
        finished, max_gen = [False] * batch_size, int(p_args['max_tokens'])
        
        with torch.no_grad():
            for step in range(max_gen):
                is_prefill = (step == 0)
                total_len = curr_input_ids.shape[1] if is_prefill else len(batch_ids[0])
                q_len = curr_input_ids.shape[1]
                
                if is_prefill:
                    causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
                    mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - current_padding_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
                else:
                    mask = (1.0 - current_padding_mask.view(batch_size, 1, 1, total_len).to(torch.bfloat16)) * -10000.0

                if self.args.node_rank == 0:
                    h = self.model.model.embed_tokens(curr_input_ids)
                    for layer in self.model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=position_ids.to(dev), past_key_value=past_key_values, use_cache=True)[0]
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids_dev, src=1)
                    curr_input_ids = new_ids_dev
                else:
                    h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0)
                    h = h_recv
                    for layer in self.model.model.layers:
                        dev = next(layer.parameters()).device
                        h = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=position_ids.to(dev), past_key_value=past_key_values, use_cache=True)[0]
                    logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))[:, -1, :]
                    logits = apply_sampling(logits, p_args['temp'], p_args['top_p'], 1.1, batch_ids)
                    next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1) if p_args['temp'] > 0 else torch.argmax(logits, -1, True)
                    dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
                    curr_input_ids = next_tokens.to("npu:0")

                nt_list = curr_input_ids.squeeze(-1).tolist()
                for i in range(batch_size):
                    if not finished[i]:
                        batch_ids[i].append(nt_list[i])
                        if nt_list[i] == self.tokenizer.eos_token_id: finished[i] = True
                if all(finished): break
                current_padding_mask = torch.cat([current_padding_mask, torch.ones((batch_size, 1), device="npu:0")], dim=-1)
                position_ids = torch.full((batch_size, 1), len(batch_ids[0]) - 1, dtype=torch.long, device="npu:0")
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch_ids]

# ==========================================
# 4. API 路由
# ==========================================
app = FastAPI()
class ChatMessage(BaseModel): role: str; content: str
class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v3-tucker"; messages: List[ChatMessage]
    max_tokens: int = 128; temperature: float = 0.7; top_p: float = 0.9
class CompletionRequest(BaseModel):
    model: str = "deepseek-v3-tucker"; prompt: Union[str, List[str]]
    max_tokens: int = 128; temperature: float = 0.7; top_p: float = 0.9; logprobs: Optional[int] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    req_id = str(uuid.uuid4()); event = asyncio.Event(); engine.events[req_id] = event
    content = request.messages[-1].content
    await engine.req_queue.put({"type": "chat", "id": req_id, "prompt": content, "max_tokens": request.max_tokens, "temp": request.temperature, "top_p": request.top_p})
    await event.wait(); res = engine.results.pop(req_id); del engine.events[req_id]
    return {"id": req_id, "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": res}, "index": 0, "finish_reason": "stop"}]}

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    req_id = str(uuid.uuid4()); event = asyncio.Event(); engine.events[req_id] = event
    p = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    is_ll = (request.logprobs is not None and request.logprobs > 0)
    await engine.req_queue.put({"type": "loglikelihood" if is_ll else "chat", "id": req_id, "prompt": p, "max_tokens": request.max_tokens, "temp": request.temperature, "top_p": request.top_p, "cont_len": request.max_tokens})
    await event.wait(); res = engine.results.pop(req_id); del engine.events[req_id]
    if is_ll:
        return {"id": req_id, "choices": [{"text": "", "index": 0, "logprobs": {"token_logprobs": [res], "tokens": ["<continuation>"], "top_logprobs": [{}]}, "finish_reason": "stop"}]}
    return {"id": req_id, "choices": [{"text": res, "index": 0, "finish_reason": "stop"}]}

# ==========================================
# 5. 主入口
# ==========================================
engine = None
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="10.120.72.45")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(3, 61)))
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr; os.environ['MASTER_PORT'] = '29506'
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['HCCL_CONNECT_TIMEOUT'] = '0'
    os.environ['HCCL_EXEC_TIMEOUT'] = '0'
    os.environ['HCCL_TCP_KEEP_ALIVE_ENABLE'] = '1'
    
    global engine; engine = DistributedInferenceEngine(args); engine.setup()
    if args.node_rank == 0:
        import threading
        def run_e():
            loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); loop.run_until_complete(engine.run_loop())
        threading.Thread(target=run_e, daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else: asyncio.run(engine.run_loop())

if __name__ == "__main__": main()
