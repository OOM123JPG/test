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
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(message)s', datefmt='%H:%M:%S')

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
# 2. 核心功能函数 (并行装箱 + 显存分发)
# ==========================================

def init_resources(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map={"": "cpu"})
    return tokenizer, model

@torch.no_grad()
def parallel_tucker_surgery(model, start_idx, args):
    num_groups, experts_per_group = 8, 32
    def process_layer(layer_data):
        real_idx, layer = layer_data
        if real_idx not in args.layers: return
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
    
    layers = [(start_idx + i, l) for i, l in enumerate(model.model.layers)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(tqdm(ex.map(process_layer, layers), total=len(layers), desc="Parallel Surgery"))

def distribute_logic(model, start_idx, args):
    RESERVED, main_npu = 8.0, "npu:0"
    model.model.embed_tokens.to(main_npu)
    if hasattr(model.model, "rotary_emb"): model.model.rotary_emb.to(main_npu)
    curr_card, last_active = 0, main_npu
    for idx, layer in enumerate(model.model.layers):
        real_idx = start_idx + idx
        req = 22.0 if real_idx not in args.layers else 5.5
        success = False
        while curr_card < 8:
            target = f"npu:{curr_card}"
            if (torch.npu.mem_get_info(target)[0] / 1024**3) > (req + RESERVED):
                try:
                    layer.to(target)
                    last_active, success = target, True
                    break
                except: curr_card += 1
            else: curr_card += 1
        if not success: layer.to("cpu")
    model.model.norm.to(last_active)
    if hasattr(model, "lm_head"): model.lm_head.to(last_active)
    return last_active

# ==========================================
# 3. 终极推理：bf16 跨机 + 修正 KV Cache
# ==========================================

def run_kv_cache_distributed(model, tokenizer, args, prompts):
    config = model.config
    batch_size = len(prompts)
    past_key_values = [None] * len(model.model.layers)
    
    try:
        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            batch_ids = inputs.input_ids.tolist()
            init_len = inputs.input_ids.shape[1]
            
            logging.info(f"==> KV Cache 模式启动 (bf16 通信)")
            
            for step in range(args.max_new_tokens):
                t_step = time.time()
                # A. 确定输入与位置
                if step == 0:
                    curr_ids = torch.tensor(batch_ids, dtype=torch.long)
                    pos_ids = torch.arange(init_len).unsqueeze(0).expand(batch_size, -1)
                    mask = torch.triu(torch.full((init_len, init_len), float("-inf")), 1).view(1, 1, init_len, init_len).expand(batch_size, 1, -1, -1).to(torch.bfloat16)
                else:
                    curr_ids = torch.tensor([[ids[-1]] for ids in batch_ids], dtype=torch.long)
                    pos_ids = torch.tensor([[init_len + step - 1]], dtype=torch.long).expand(batch_size, 1)
                    mask = torch.zeros((batch_size, 1, 1, 1)).to(torch.bfloat16)

                # B. 执行推理
                if args.node_rank == 0:
                    h = model.model.embed_tokens(curr_ids.to("npu:0"))
                    for i, layer in enumerate(model.model.layers):
                        dev = next(layer.parameters()).device
                        h, past_key_values[i] = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), past_key_value=past_key_values[i], use_cache=True)
                    
                    # 关键点：转 bf16 后发送
                    dist.send(tensor=h.to("npu:0").to(torch.bfloat16).contiguous(), dst=1)
                    
                    new_ids = torch.zeros((batch_size, 1), dtype=torch.long, device="npu:0")
                    dist.recv(tensor=new_ids, src=1)
                    for i in range(batch_size): batch_ids[i].append(new_ids[i].item())
                    if args.stream: print(tokenizer.decode([batch_ids[0][-1]]), end="", flush=True)

                else:
                    # Node 1 接收 (bf16 Buffer)
                    r_shape = (batch_size, curr_ids.shape[1], config.hidden_size)
                    h_recv = torch.zeros(r_shape, dtype=torch.bfloat16, device="npu:0")
                    dist.recv(tensor=h_recv, src=0)
                    
                    h = h_recv
                    for i, layer in enumerate(model.model.layers):
                        dev = next(layer.parameters()).device
                        h, past_key_values[i] = layer(h.to(dev), attention_mask=mask.to(dev), position_ids=pos_ids.to(dev), past_key_value=past_key_values[i], use_cache=True)
                    
                    head_dev = next(model.lm_head.parameters()).device
                    logits = model.lm_head(model.model.norm(h.to(head_dev)))[:, -1, :]
                    
                    next_tk = torch.argmax(logits, dim=-1, keepdim=True) if args.temperature == 0 else torch.multinomial(torch.softmax(logits/args.temperature, -1), 1)
                    dist.send(tensor=next_tk.to("npu:0").contiguous(), dst=0)
                    for i in range(batch_size): batch_ids[i].append(next_tk[i].item())

                if all(ids[-1] == tokenizer.eos_token_id for ids in batch_ids): break

            if args.node_rank == 0:
                print("\n" + "="*50 + "\n结果汇总:\n" + "\n".join([f"[{i+1}] {tokenizer.decode(res, skip_special_tokens=True)}" for i, res in enumerate(batch_ids)]))

    except Exception: traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--whitening_dir", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs='+', default=range(3, 61))
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="10.120.72.45")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = args.master_addr, '29506'
    dist.init_process_group(backend='hccl', rank=args.node_rank, world_size=2)
    torch.npu.set_device("npu:0")

    tokenizer, model = init_resources(args.model_path)
    mid = 27
    all_l = model.model.layers
    if args.node_rank == 0:
        model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid)]), 0
    else:
        model.model.layers, start = nn.ModuleList([all_l[i] for i in range(mid, len(all_l))]), mid
    del all_l
    gc.collect()

    parallel_tucker_surgery(model, start, args)
    distribute_logic(model, start, args)
    
    prompts = ["什么是人工智能？"]
    run_kv_cache_distributed(model, tokenizer, args, prompts)

if __name__ == "__main__":
    main()
