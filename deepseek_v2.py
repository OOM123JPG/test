from vllm.distributed import tensor_model_parallel_all_reduce, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

class DistributedTuckerLinear(nn.Module):
    def __init__(self, data, proj_type="gate_up", tp_size=1, tp_rank=0):
        super().__init__()
        self.proj_type = proj_type
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # 只有在确认为当前设备后才移动到 NPU
        device = torch.npu.current_device()
        
        self.register_buffer('U_E', data['factors'][0].to(torch.bfloat16).to(f"npu:{device}"))
        self.register_buffer('core', data['core'].to(torch.bfloat16).to(f"npu:{device}"))

        u_h_raw = data['factors'][1] 
        u_i_raw = data['factors'][2] 
        r_h, r_i = u_h_raw.shape[1], u_i_raw.shape[1]
        r_h_shard, r_i_shard = r_h // tp_size, r_i // tp_size

        # 真正的分片加载逻辑
        if proj_type == "gate_up":
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to(f"npu:{device}").contiguous())
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to(f"npu:{device}").contiguous())
        else:
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to(f"npu:{device}").contiguous())
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to(f"npu:{device}").contiguous())

    def forward(self, x, expert_inner_indices):
        ue = self.U_E[expert_inner_indices]
        r_h, r_i = self.core.shape[1], self.core.shape[2]
        combined_core = torch.matmul(ue, self.core.view(self.core.shape[0], -1)).view(-1, r_h, r_i)
        
        r_h_shard, r_i_shard = r_h // self.tp_size, r_i // self.tp_size

        if self.proj_type == "gate_up":
            x_low = torch.matmul(x, self.U_I)
            core_shard = combined_core[:, :, self.tp_rank * r_i_shard : (self.tp_rank + 1) * r_i_shard]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard.transpose(1, 2)).squeeze(1)
            mid = tensor_model_parallel_all_reduce(mid)
            mid_shard = mid[:, self.tp_rank * r_h_shard : (self.tp_rank + 1) * r_h_shard]
            return torch.matmul(mid_shard, self.U_H.T)
        else:
            x_low = torch.matmul(x, self.U_H)
            core_shard = combined_core[:, self.tp_rank * r_h_shard : (self.tp_rank + 1) * r_h_shard, :]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard).squeeze(1)
            mid = tensor_model_parallel_all_reduce(mid)
            mid_shard = mid[:, self.tp_rank * r_i_shard : (self.tp_rank + 1) * r_i_shard]
            return torch.matmul(mid_shard, self.U_I.T)



class DeepseekV2MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config=None, prefix="", layer_idx=-1):
        super().__init__()
        # 1. 确保并行状态已初始化
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # 2. 基础配置
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # 3. Tucker 状态
        self.layer_idx = layer_idx
        self.is_tucker_active = (getattr(config, "use_tucker", False) and 
                                 self.layer_idx in getattr(config, "tucker_layers", []))

        # 4. 加载专家逻辑
        self.tucker_experts = nn.ModuleDict()
        if self.is_tucker_active:
            tucker_path = getattr(config, "tucker_path", "/nfs-share/wx1463835/tdmoe/output/decomp_results")
            
            # 确定当前卡负责的专家组
            groups_per_ep = 8 // self.ep_size
            my_group_ids = range(self.ep_rank * groups_per_ep, (self.ep_rank + 1) * groups_per_ep)

            # 错峰加载防止 NFS 崩溃
            time.sleep(self.tp_rank * 5) # 稍微缩短间隔
            
            for g_id in my_group_ids:
                f_path = os.path.join(tucker_path, f"layer_{self.layer_idx}_group_{g_id}.pt")
                if os.path.exists(f_path):
                    # 必须 map_location='cpu'
                    data = torch.load(f_path, map_location='cpu')
                    # 修正变量名为 self.tp_size 和 self.tp_rank
                    self.tucker_experts[str(g_id)] = nn.ModuleDict({
                        'gate': DistributedTuckerLinear(data['gate_proj'], "gate_up", self.tp_size, self.tp_rank),
                        'up':   DistributedTuckerLinear(data['up_proj'],   "gate_up", self.tp_size, self.tp_rank),
                        'down': DistributedTuckerLinear(data['down_proj'], "down",    self.tp_size, self.tp_rank),
                    })
                    del data
                    if self.tp_rank == 0:
                        print(f"[TUCKER] Layer {self.layer_idx} Group {g_id} 加载成功")
                else:
                    if self.tp_rank == 0:
                        print(f"[TUCKER] 警告: 找不到文件 {f_path}")}")

    def forward_tucker(self, hidden_states, router_logits):
        import torch.nn.functional as F
        # 1. 获取路由权重
        try:
            res = self.experts.select_experts(router_logits)
            routing_weights, selected_experts = res if isinstance(res, tuple) else (res.topk_weights, res.topk_indices)
        except:
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(hidden_states.dtype)

        # 2. 单机模拟分发 (多机请恢复 moe_dispatch)
        dispatched_tokens = hidden_states.repeat_interleave(self.top_k, dim=0) 
        indices = selected_experts.flatten()
        local_output = torch.zeros_like(dispatched_tokens)
        
        # 3. Tucker 计算
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

        if self.tp_rank == 0:
            print(f"[DEBUG Layer {self.layer_idx}] 命中Tokens: {hit_count}")

        # 4. 聚合
        combined_output = (local_output.view(hidden_states.shape[0], self.top_k, -1) 
                           * routing_weights.unsqueeze(-1)).sum(dim=1)
        return combined_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)

        if self.is_tucker_active:
            final_hidden_states = self.forward_tucker(hidden_states, router_logits)
            shared_output = self.shared_experts(hidden_states) if self.shared_experts else None
        else:
            fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
            shared_output, final_hidden_states = fused_moe_out

        if self.shared_experts is not None and shared_output is not None:
            final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)
