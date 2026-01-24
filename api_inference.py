import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch_npu
import argparse
import logging
import gc
import queue
import threading
import asyncio
import uuid
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
# 0. 日志与全局配置
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 全局变量用于跨线程通信
task_queue = queue.Queue() # FastAPI -> Inference Thread
main_loop = None           # 存储 FastAPI 的事件循环

# ==========================================
# 1. Tucker 组件 (保持不变)
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
    if temperature > 0: 
        logits = logits / temperature
    else: 
        return logits
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
# 3. 分布式推理引擎 (核心重构)
# ==========================================
class DistributedInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.start_idx = 0
        self.results_map = {} # 用于暂存结果的字典

    def setup(self):
        logger.info(f"=== [Node {self.args.node_rank}] 正在初始化分布式环境 ===")
        # 设置环境变量（增加一些调试用的超时设置）
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        
        try:
            logger.info(f"正在尝试连接到 Master: {self.args.master_addr}:29506 ...")
            dist.init_process_group(
                backend='hccl', 
                rank=self.args.node_rank, 
                world_size=2, 
                timeout=timedelta(seconds=7200)
            )
            logger.info("✓ 分布式环境初始化成功 (HCCL Handshake Done)")
        except Exception as e:
            logger.error(f"✗ 分布式初始化失败: {e}")
            sys.exit(1)

        torch.npu.set_device("npu:0")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        logger.info("正在从 CPU 加载模型骨架...")
        full_model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            device_map={"": "cpu"}
        )
        
        # 这里的 mid 根据你的需求改成了 34
        mid = 34
        all_l = full_model.model.layers
        if self.args.node_rank == 0:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid)])
            self.start_idx = 0
            logger.info(f"Node 0 分配层级: 0 - {mid-1}")
        else:
            full_model.model.layers = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))])
            self.start_idx = mid
            logger.info(f"Node 1 分配层级: {mid} - {len(all_l)-1}")
            
        self.model = full_model
        
        # 进入搬运环节
        self._pipeline_surgery_and_loading()
        
        dist.barrier()
        logger.info(f"=== [Node {self.args.node_rank}] 模型加载完毕，服务就绪 ===")

    def _pipeline_surgery_and_loading(self):
        """执行 Tucker 替换、模型搬运并打印详细参数统计"""
        num_layers = len(self.model.model.layers)
        RESERVED, main_npu = 8, "npu:0" 
        
        logger.info("--- Step 1: 执行 Tucker Expert 替换 (Surgery) ---")
        with ThreadPoolExecutor(max_workers=8) as executor:
            def surgery_task(idx_in_module):
                layer = self.model.model.layers[idx_in_module]
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
            list(executor.map(surgery_task, range(num_layers)))

        logger.info("--- Step 2: 模型搬运至 NPU & 显存装箱 ---")
        curr_card = 0
        if self.args.node_rank == 0:
            self.model.model.embed_tokens.to(main_npu)
            logger.info(f" [Node 0] Embedding -> {main_npu}")

        for i in range(num_layers):
            real_idx = self.start_idx + i
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
                        layer.to(target)
                        logger.info(f" - Layer {real_idx:02d}: {target} | Size: {req_gb:.3f} GB | Free: {free_gb - req_gb:.2f} GB")
                        success = True
                        break
                    except Exception as e:
                        logger.warning(f" NPU:{curr_card} 搬运失败，尝试下一张卡. Err: {e}")
                        curr_card += 1
                else:
                    curr_card += 1
            if not success:
                logger.error(f" !!! 关键错误: Layer {real_idx} 无法加载到任何 NPU，将留在 CPU !!!")

        if self.args.node_rank == 1:
            # 找到 Node 1 最后一张有空间的卡放 norm 和 head
            final_target = f"npu:{min(curr_card, 7)}"
            self.model.model.norm.to(final_target)
            self.model.lm_head.to(final_target)
            logger.info(f" [Node 1] Norm & LM_Head -> {final_target}")

        # --- 这里是你要的参数量打印 ---
        logger.info("==================================================")
        total_params = sum(p.numel() for p in self.model.parameters())
        total_buffers = sum(b.numel() for b in self.model.buffers())
        node_total_billions = (total_params + total_buffers) / 1e9
        
        logger.info(f"Node {self.args.node_rank} 参数统计摘要:")
        logger.info(f" - 模型权重参数量: {total_params / 1e9:.3f} B")
        logger.info(f" - Tucker 缓存量: {total_buffers / 1e9:.3f} B")
        logger.info(f" - 节点总显存负荷: {node_total_billions:.3f} B (参数总数)")
        logger.info("==================================================")
            
        torch.npu.empty_cache()
        gc.collect()

    def run_inference_loop(self):
        """核心推理循环：此函数在独立线程运行"""
        logger.info(f"Starting Inference Loop on Node {self.args.node_rank}")
        while True:
            active_reqs = []
            if self.args.node_rank == 0:
                # Rank 0 尝试从队列获取任务
                try:
                    # 阻塞 0.1s 检查是否有新请求
                    req = task_queue.get(timeout=0.1)
                    active_reqs.append(req)
                    # 尝试 Batching (最多16个)
                    while not task_queue.empty() and len(active_reqs) < 16:
                        active_reqs.append(task_queue.get_nowait())
                except queue.Empty:
                    # 无任务，通知 Rank 1 继续等待
                    dist.broadcast(torch.tensor([0], dtype=torch.long, device="npu:0"), src=0)
                    continue

                # 发送 Batch Size 给 Rank 1
                dist.broadcast(torch.tensor([len(active_reqs)], dtype=torch.long, device="npu:0"), src=0)
                
                # 同步参数
                p_args = active_reqs[0]
                task_code = 1 if p_args['type'] == 'loglikelihood' else 0
                params = torch.tensor([task_code, p_args['max_tokens'], p_args['temp'], p_args['top_p'], 0], device="npu:0", dtype=torch.float32)
                dist.broadcast(params, src=0)
                
                # Tokenize 并同步 Input IDs
                inputs = self.tokenizer([r['prompt'] for r in active_reqs], return_tensors="pt", padding=True).to("npu:0")
                dist.broadcast(torch.tensor([inputs.input_ids.shape[1]], dtype=torch.long, device="npu:0"), src=0)
                dist.broadcast(inputs.input_ids, src=0)
                input_ids = inputs.input_ids
            else:
                # Rank 1 等待指令
                signal = torch.zeros([1], dtype=torch.long, device="npu:0")
                dist.broadcast(signal, src=0)
                if signal.item() == 0: continue
                
                batch_size = int(signal.item())
                params = torch.zeros([5], device="npu:0")
                dist.broadcast(params, src=0)
                task_code, p_max, p_temp, p_top, _ = params.tolist()
                p_args = {'max_tokens': int(p_max), 'temp': p_temp, 'top_p': p_top}
                
                seq_len_t = torch.zeros([1], dtype=torch.long, device="npu:0")
                dist.broadcast(seq_len_t, src=0)
                input_ids = torch.zeros((batch_size, int(seq_len_t.item())), dtype=torch.long, device="npu:0")
                dist.broadcast(input_ids, src=0)

            # 执行具体任务
            if task_code == 1:
                results = self._execute_loglikelihood_sync(input_ids, len(input_ids))
            else:
                results = self._execute_inference_sync(input_ids, len(input_ids), p_args)
            
            # Rank 0 分发结果并唤醒 API 线程
            if self.args.node_rank == 0:
                for i, req in enumerate(active_reqs):
                    rid = req['id']
                    event = req['event']
                    self.results_map[rid] = results[i]
                    # 线程安全地唤醒 asyncio 事件
                    main_loop.call_soon_threadsafe(event.set)

    def _execute_loglikelihood_sync(self, input_ids, batch_size):
        with torch.no_grad():
            q_len = input_ids.shape[1]
            # 准备 Attention Mask
            padding_mask = (input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
            causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
            mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - padding_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
            
            if self.args.node_rank == 0:
                h = self.model.model.embed_tokens(input_ids)
                for layer in self.model.model.layers:
                    h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
                # P2P 发送到 Node 1
                dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                
                # 接收最终的 logprobs [batch, q_len]
                logprobs_recv = torch.zeros((batch_size, q_len), device="npu:0", dtype=torch.float32)
                dist.recv(tensor=logprobs_recv, src=1)
                return logprobs_recv.tolist()
            else:
                h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                dist.recv(tensor=h_recv, src=0)
                h = h_recv
                for layer in self.model.model.layers:
                    h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), use_cache=False)[0]
                
                logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))
                logprobs = F.log_softmax(logits.float(), dim=-1) # [batch, q_len, vocab]
                
                # 提取目标 token 的概率：input_ids 的 [1:] 对应 logits 的 [:-1]
                target_ids = input_ids[:, 1:].unsqueeze(-1).to(logprobs.device)
                relevant_logprobs = logprobs[:, :-1, :].gather(-1, target_ids).squeeze(-1) 
                
                # 补齐第一个 token (无前驱概率)
                padding_val = relevant_logprobs.new_zeros((batch_size, 1))
                final_logprobs = torch.cat([padding_val, relevant_logprobs], dim=-1)
                
                dist.send(tensor=final_logprobs.to("npu:0").to(torch.float32).contiguous(), dst=0)
                return final_logprobs.tolist()

    def _execute_inference_sync(self, input_ids, batch_size, p_args):
        batch_ids = input_ids.tolist()
        past_key_values = DynamicCache()
        curr_input_ids = input_ids
        
        current_mask = (curr_input_ids != self.tokenizer.pad_token_id).long().to("npu:0")
        position_ids = current_mask.cumsum(-1) - 1
        position_ids.masked_fill_(current_mask == 0, 0)
        
        finished = [False] * batch_size
        max_gen = int(p_args['max_tokens'])
        
        with torch.no_grad():
            for step in range(max_gen):
                is_prefill = (step == 0)
                q_len = curr_input_ids.shape[1]
                total_len = len(batch_ids[0])
                
                if is_prefill:
                    causal_mask = torch.triu(torch.full((q_len, q_len), float("-inf"), device="npu:0"), 1)
                    mask = causal_mask.view(1, 1, q_len, q_len) + (1.0 - current_mask.view(batch_size, 1, 1, q_len).to(torch.bfloat16)) * -10000.0
                else:
                    mask = (1.0 - current_mask.view(batch_size, 1, 1, total_len).to(torch.bfloat16)) * -10000.0
                
                if self.args.node_rank == 0:
                    h = self.model.model.embed_tokens(curr_input_ids)
                    for layer in self.model.model.layers:
                        h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), position_ids=position_ids.to(next(layer.parameters()).device), past_key_value=past_key_values, use_cache=True)[0]
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
                    new_ids_dev = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids_dev, src=1)
                    curr_input_ids = new_ids_dev
                else:
                    h_recv = torch.zeros((batch_size, q_len, self.model.config.hidden_size), dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0)
                    h = h_recv
                    for layer in self.model.model.layers:
                        h = layer(h.to(next(layer.parameters()).device), attention_mask=mask.to(next(layer.parameters()).device), position_ids=position_ids.to(next(layer.parameters()).device), past_key_value=past_key_values, use_cache=True)[0]
                    
                    logits = self.model.lm_head(self.model.model.norm(h.to(next(self.model.lm_head.parameters()).device)))[:, -1, :]
                    logits = apply_sampling(logits, p_args['temp'], p_args['top_p'], 1.1, batch_ids)
                    next_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1) if p_args['temp'] > 0 else torch.argmax(logits, -1, True)
                    
                    dist.send(tensor=next_tokens.to("npu:0").contiguous(), dst=0)
                    curr_input_ids = next_tokens.to("npu:0")

                # 更新状态
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

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v3-tucker"
    messages: List[Dict]
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class CompletionRequest(BaseModel):
    model: str = "deepseek-v3-tucker"
    prompt: Union[str, List[str]]
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: Optional[int] = None
    echo: bool = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    req_id = str(uuid.uuid4())
    event = asyncio.Event()
    
    # 格式化 Prompt
    formatted_prompt = ""
    for msg in request.messages:
        formatted_prompt += f"{msg['role']}: {msg['content']}\n"
    formatted_prompt += "assistant: "

    task_queue.put({
        "type": "chat", "id": req_id, "event": event, "prompt": formatted_prompt,
        "max_tokens": request.max_tokens, "temp": request.temperature, "top_p": request.top_p
    })

    await event.wait()
    res = engine.results_map.pop(req_id)
    return {
        "id": req_id, "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": res}, "index": 0, "finish_reason": "stop"}]
    }

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    req_id_base = str(uuid.uuid4())
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    is_ll = (request.logprobs is not None and request.logprobs > 0)
    
    # 预先编码以便获取 Token 字符串
    encoded = engine.tokenizer(prompts, padding=True, return_tensors="pt")
    all_input_ids = encoded.input_ids

    events, results_ids = [], []
    for i, p in enumerate(prompts):
        rid = f"{req_id_base}-{i}"
        ev = asyncio.Event()
        events.append(ev)
        results_ids.append(rid)
        task_queue.put({
            "type": "loglikelihood" if is_ll else "chat", "id": rid, "event": ev, "prompt": p,
            "max_tokens": request.max_tokens, "temp": request.temperature, "top_p": request.top_p
        })
    
    await asyncio.gather(*(ev.wait() for ev in events))
    
    choices = []
    for i, rid in enumerate(results_ids):
        res = engine.results_map.pop(rid)
        if is_ll:
            token_ids = all_input_ids[i]
            tokens_str = engine.tokenizer.convert_ids_to_tokens(token_ids)
            pad_id = engine.tokenizer.pad_token_id
            first_valid = (token_ids != pad_id).long().argmax().item()
            
            valid_tokens = tokens_str[first_valid:]
            valid_logprobs = res[first_valid:]
            top_lp = [{t: lp} for t, lp in zip(valid_tokens, valid_logprobs)]

            choices.append({
                "text": prompts[i] if request.echo else "",
                "message": {"role": "assistant", "content": ""}, # 兼容性补充
                "index": i, 
                "logprobs": {
                    "token_logprobs": valid_logprobs, 
                    "tokens": valid_tokens, 
                    "top_logprobs": top_lp
                }, 
                "finish_reason": "stop"
            })
        else:
            choices.append({
                "text": res, "index": i, "finish_reason": "stop",
                "message": {"role": "assistant", "content": res}
            })
            
    return {"id": req_id_base, "object": "text_completion", "choices": choices}

# ==========================================
# 5. 启动入口
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--layers", type=int, nargs='+', default=list(range(3, 61)))
    args = parser.parse_args()

    # 环境变量配置
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = '29506'
    os.environ['HCCL_CONNECT_TIMEOUT'] = '7200'
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'max_split_size_mb:128'

    global engine, main_loop
    engine = DistributedInferenceEngine(args)
    engine.setup()

    if args.node_rank == 0:
        # 获取当前 asyncio 事件循环
        main_loop = asyncio.get_event_loop()
        # 启动推理线程
        threading.Thread(target=engine.run_inference_loop, daemon=True).start()
        # 启动 API 服务
        uvicorn.run(app, host="0.0.0.0", port=args.port, loop="asyncio")
    else:
        # Rank 1 直接同步运行推理循环
        engine.run_inference_loop()

if __name__ == "__main__":
    main()