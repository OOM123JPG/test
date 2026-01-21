import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from typing import List, Optional
from vllm.distributed import (get_tensor_model_parallel_rank, 
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

# 假设 DistributedTuckerLinear 已在同文件或上方定义
class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config=None, prefix="", layer_idx=-1):
        super().__init__()
        # 1. 获取分布式状态
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        # 2. 基础配置加载
        self.config = config
        self.layer_idx = layer_idx
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        
        # 3. Tucker 激活判定
        self.use_tucker_cfg = getattr(config, "use_tucker", False)
        self.tucker_layers = getattr(config, "tucker_layers", [])
        self.is_tucker_active = (self.use_tucker_cfg and self.layer_idx in self.tucker_layers)

        # 4. 路由 Gate 初始化
        from vllm.model_executor.layers.linear import ReplicatedLinear
        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False,
            quant_config=None, prefix=f"{prefix}.gate",
        )

        # 5. Tucker 专家加载逻辑 (核心)
        self.tucker_experts = nn.ModuleDict()
        if self.is_tucker_active:
            tucker_path = getattr(config, "tucker_path", "/nfs-share/wx1463835/tdmoe/output/decomp_results")
            
            # 双机 TP8 环境下，根据 rank 计算分片
            # 假设 256 专家分 8 组，每组 32 个
            experts_per_node = 256 // 1  # 如果是 EP1, 则每节点看到全量组
            my_group_ids = range(0, 8) # 这里建议加载所有组，forward 时按 mask 过滤

            # 错峰加载，防止 16 个进程同时冲击 NFS
            time.sleep(self.tp_rank * 2)
            
            if self.tp_rank == 0:
                print(f"[TUCKER] Layer {self.layer_idx} 启动错峰加载...")

            for g_id in my_group_ids:
                f_path = os.path.join(tucker_path, f"layer_{self.layer_idx}_group_{g_id}.pt")
                if os.path.exists(f_path):
                    if self.tp_rank == 0:
                        print(f"[TUCKER] Rank {self.tp_rank} 加载 Group {g_id}")
                    # 必须在 CPU 加载
                    data = torch.load(f_path, map_location='cpu')
                    self.tucker_experts[str(g_id)] = nn.ModuleDict({
                        'gate': DistributedTuckerLinear(data['gate_proj'], "gate_up", self.tp_size, self.tp_rank),
                        'up':   DistributedTuckerLinear(data['up_proj'],   "gate_up", self.tp_size, self.tp_rank),
                        'down': DistributedTuckerLinear(data['down_proj'], "down",    self.tp_size, self.tp_rank),
                    })
                    del data
            
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def forward_tucker(self, hidden_states, router_logits):
        # 1. 路由计算
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 2. Token 分发
        dispatched_tokens = hidden_states.repeat_interleave(self.top_k, dim=0)
        indices = selected_experts.flatten()
        local_output = torch.zeros_like(dispatched_tokens)
        
        # 3. Tucker 重建计算
        hit_count = 0
        for g_id_str, group in self.tucker_experts.items():
            g_id = int(g_id_str)
            mask = (indices // 32 == g_id)
            if mask.any():
                token_inp = dispatched_tokens[mask]
                inner_ids = indices[mask] % 32
                hit_count += token_inp.shape[0]
                
                gate_out = group['gate'](token_inp, inner_ids)
                up_out = group['up'](token_inp, inner_ids)
                inter = F.silu(gate_out) * up_out
                local_output[mask] = group['down'](inter, inner_ids)

        # 4. 聚合输出
        combined_output = (local_output.view(hidden_states.shape[0], self.top_k, -1) 
                           * routing_weights.unsqueeze(-1)).sum(dim=1)
        return combined_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        if self.is_tucker_active:
            return self.forward_tucker(hidden_states, router_logits)
        # 非 Tucker 层走原有的 FusedMoE 逻辑...
