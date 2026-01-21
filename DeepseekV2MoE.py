import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from typing import List, Optional
from vllm.distributed import (get_tensor_model_parallel_rank, 
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config=None, prefix="", layer_idx=-1):
        super().__init__()
        # 1. 自动获取分布式状态 (TP8 PP4 模式下 tp_size=8)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        self.config = config
        self.layer_idx = layer_idx
        self.top_k = config.num_experts_per_tok
        
        # 2. Tucker 激活逻辑
        self.use_tucker_cfg = getattr(config, "use_tucker", False)
        self.tucker_layers = getattr(config, "tucker_layers", [])
        self.is_tucker_active = (self.use_tucker_cfg and self.layer_idx in self.tucker_layers)

        # 3. 路由 Gate 初始化 (使用 vLLM 标准层)
        from vllm.model_executor.layers.linear import ReplicatedLinear
        self.gate = ReplicatedLinear(
            config.hidden_size, config.n_routed_experts, bias=False,
            quant_config=None, prefix=f"{prefix}.gate",
        )

        # 4. Tucker 权重异步错峰加载
        self.tucker_experts = nn.ModuleDict()
        if self.is_tucker_active:
            tucker_path = getattr(config, "tucker_path", "/nfs-share/wx1463835/tdmoe/output/decomp_results")
            
            # 每张卡加载属于自己的 Tucker 专家组 (DeepSeek-V3 256专家分8组)
            # 在 TP8 模式下，我们让所有卡都加载 8 个组的 .pt，但在 forward 时通过 TP 分片计算
            my_group_ids = range(8) 

            # 错峰加载：防止 16 个进程瞬间卡死 NFS
            # Node 0 (rank 0-7) 先加载，Node 1 (rank 8-15) 稍后
            time.sleep(self.tp_rank * 3)
            
            if self.tp_rank == 0:
                print(f"[TUCKER] Layer {self.layer_idx} 正在从 {tucker_path} 错峰加载...")

            for g_id in my_group_ids:
                f_path = os.path.join(tucker_path, f"layer_{self.layer_idx}_group_{g_id}.pt")
                if os.path.exists(f_path):
                    # 必须在 CPU 加载，防止分布式初始化前抢占 NPU 内存
                    data = torch.load(f_path, map_location='cpu')
                    
                    # 实例化自定义的分布式 Tucker 线性层
                    self.tucker_experts[str(g_id)] = nn.ModuleDict({
                        'gate': DistributedTuckerLinear(data['gate_proj'], "gate_up", self.tp_size, self.tp_rank),
                        'up':   DistributedTuckerLinear(data['up_proj'],   "gate_up", self.tp_size, self.tp_rank),
                        'down': DistributedTuckerLinear(data['down_proj'], "down",    self.tp_size, self.tp_rank),
                    })
                    del data
                    if self.tp_rank % 8 == 0:
                        print(f"[TUCKER] Rank {self.tp_rank} 成功加载 Group {g_id}")
            
            # 加载完毕后强制同步，防止 PP 流水线过早启动
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    def forward_tucker(self, hidden_states, router_logits):
        # 1. 路由与 Top-K
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 2. Token 分发逻辑
        dispatched_tokens = hidden_states.repeat_interleave(self.top_k, dim=0)
        indices = selected_experts.flatten()
        local_output = torch.zeros_like(dispatched_tokens)
        
        # 3. 专家组遍历与 Tucker 重构
        for g_id_str, group in self.tucker_experts.items():
            g_id = int(g_id_str)
            mask = (indices // 32 == g_id)
            if mask.any():
                token_inp = dispatched_tokens[mask]
                inner_ids = indices[mask] % 32 # 获取 0-31 的内部索引
                
                # 计算分解后的中间结果
                gate_out = group['gate'](token_inp, inner_ids)
                up_out = group['up'](token_inp, inner_ids)
                inter = F.silu(gate_out) * up_out
                local_output[mask] = group['down'](inter, inner_ids)

        # 4. 结果聚合
        combined_output = (local_output.view(hidden_states.shape[0], self.top_k, -1) 
                           * routing_weights.unsqueeze(-1)).sum(dim=1)
        return combined_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 核心入口
        router_logits, _ = self.gate(hidden_states)
        if self.is_tucker_active:
            return self.forward_tucker(hidden_states, router_logits)
        
        # ... (原有逻辑保持不变)
