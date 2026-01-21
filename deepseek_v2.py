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
    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        layer_idx: int = -1,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe
        self.top_k = config.num_experts_per_tok

        # --- Tucker 状态初始化 ---
        self.layer_idx = layer_idx
        self.use_tucker_cfg = getattr(config, "use_tucker", False)
        self.tucker_layers = getattr(config, "tucker_layers", [])
        self.is_tucker_active = (self.use_tucker_cfg and self.layer_idx in self.tucker_layers)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        ) if getattr(config, "topk_method", None) == "noaux_tc" else None

        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_local_physical_experts = (self.n_routed_experts + self.n_redundant_experts) // self.ep_size

        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=1.0,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

        # --- Tucker 权重异步加载 (TP4 兼容版) ---
        self.tucker_experts = nn.ModuleDict()
        if self.is_tucker_active:
            # --- 核心改进：分时错峰加载 ---
            # 根据 rank 错开时间，每张卡间隔 10 秒（给 NFS 缓冲时间）
            wait_time = self.tp_rank * 10 
            if self.tp_rank == 0:
                print(f"[TUCKER] 为了减轻 NFS 压力，采用错峰加载模式...")
            
            time.sleep(wait_time) 
            
            for g_id in my_group_ids:
                f_path = os.path.join(tucker_path, f"layer_{self.layer_idx}_group_{g_id}.pt")
                if os.path.exists(f_path):
                    print(f"[TUCKER Rank {self.tp_rank}] 正在读取: {f_path}")
                    # 必须使用 map_location='cpu'，先在 CPU 完成分片，再上 NPU
                    data = torch.load(f_path, map_location='cpu')
                    
                    self.tucker_experts[str(g_id)] = nn.ModuleDict({
                        'gate': DistributedTuckerLinear(data['gate_proj'], "gate_up", curr_tp_size, curr_tp_rank),
                        'up':   DistributedTuckerLinear(data['up_proj'],   "gate_up", curr_tp_size, curr_tp_rank),
                        'down': DistributedTuckerLinear(data['down_proj'], "down",    curr_tp_size, curr_tp_rank),
                    })
                    del data
                    if self.tp_rank == 0:
                        print(f"[TUCKER] 成功发现并分片加载: {f_path}")
                else:
                    if self.tp_rank == 0:
                        print(f"[TUCKER] 警告: 找不到文件 {f_path}")

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
