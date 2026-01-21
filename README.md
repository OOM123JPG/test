# test
```
from vllm.distributed import (get_tensor_model_parallel_rank, 
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

class DistributedTuckerLinear(nn.Module):
    def __init__(self, data, proj_type="gate_up", tp_size=1, tp_rank=0):
        super().__init__()
        self.proj_type = proj_type
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        
        # 1. 加载因子和 Core
        self.register_buffer('U_E', data['factors'][0].to(torch.bfloat16).to("npu"))
        self.register_buffer('core', data['core'].to(torch.bfloat16).to("npu"))

        # 2. 真正的分片加载逻辑
        u_h_raw = data['factors'][1]
        u_i_raw = data['factors'][2]
        
        r_h, r_i = u_h_raw.shape[1], u_i_raw.shape[1]
        r_h_shard, r_i_shard = r_h // tp_size, r_i // tp_size

        if proj_type == "gate_up":
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to("npu").contiguous())
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to("npu").contiguous())
        else:
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to("npu").contiguous())
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to("npu").contiguous())

    def forward(self, x, expert_inner_indices):
        # ... 这里的计算逻辑保持不变，但使用 self.tp_size / self.tp_rank ...
        ue = self.U_E[expert_inner_indices]
        r_h, r_i = self.core.shape[1], self.core.shape[2]
        combined_core = torch.matmul(ue, self.core.view(self.core.shape[0], -1)).view(-1, r_h, r_i)

        r_h_shard, r_i_shard = r_h // self.tp_size, r_i // self.tp_size

        if self.proj_type == "gate_up":
            x_low = torch.matmul(x, self.U_I)
            core_shard = combined_core[:, :, self.tp_rank * r_i_shard : (self.tp_rank + 1) * r_i_shard]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard.transpose(1, 2)).squeeze(1)
            mid = tensor_model_parallel_all_reduce(mid) # 这个函数在 forward 里调是安全的
            mid_shard = mid[:, self.tp_rank * r_h_shard : (self.tp_rank + 1) * r_h_shard]
            return torch.matmul(mid_shard, self.U_H.T)
        else:
            # Down 逻辑同理
            x_low = torch.matmul(x, self.U_H)
            core_shard = combined_core[:, self.tp_rank * r_h_shard : (self.tp_rank + 1) * r_h_shard, :]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard).squeeze(1)
            mid = tensor_model_parallel_all_reduce(mid)
            mid_shard = mid[:, self.tp_rank * r_i_shard : (self.tp_rank + 1) * r_i_shard]
            return torch.matmul(mid_shard, self.U_I.T)
```

# DeepseekV2MoE.__init__
```
# --- Tucker 权重异步加载与 TP4 分片配置 ---
        if self.is_tucker_active:
            # 1. 获取基础路径与分布式状态
            tucker_path = getattr(config, "tucker_path", "/home/GZGKD001/tmp/yanhong/tdmoe_deepseek/output/decomp_results")
            
            # 安全获取 TP/EP 状态 (此时 initialize_model_parallel 必须已完成)
            try:
                current_tp_size = get_tensor_model_parallel_world_size()
                current_tp_rank = get_tensor_model_parallel_rank()
                # 兼容单机测试：如果获取失败(size=0)，强制设为 1
                current_tp_size = max(1, current_tp_size)
            except Exception:
                current_tp_size, current_tp_rank = 1, 0

            # 计算本卡需要加载的专家组 (EP 逻辑)
            ep_size_val = max(1, self.ep_size)
            groups_per_ep = 8 // ep_size_val
            my_group_ids = range(self.ep_rank * groups_per_ep, (self.ep_rank + 1) * groups_per_ep)
            
            self.tucker_experts = nn.ModuleDict()

            if self.tp_rank == 0:
                print(f"\n[TUCKER INIT] Layer {self.layer_idx} | TP_Size: {current_tp_size} | TP_Rank: {current_tp_rank}")
                print(f"[TUCKER INIT] 计划加载专家组: {list(my_group_ids)}")

            # 2. 遍历并加载权重
            for g_id in my_group_ids:
                f_path = os.path.join(tucker_path, f"layer_{self.layer_idx}_group_{g_id}.pt")
                
                if os.path.exists(f_path):
                    if self.tp_rank == 0:
                        print(f"[TUCKER INIT] 正在分片加载 Group {g_id}: {f_path}")
                    
                    # 必须 map_location='cpu'，防止全量瞬间挤爆 NPU 0
                    data = torch.load(f_path, map_location='cpu')
                    
                    # 实例化 Tucker 专家，并显式传入 tp 信息
                    # 内部 DistributedTuckerLinear 会处理 CPU -> NPU 的分片
                    self.tucker_experts[str(g_id)] = nn.ModuleDict({
                        'gate': DistributedTuckerLinear(
                            data['gate_proj'], "gate_up", 
                            tp_size=current_tp_size, tp_rank=current_tp_rank
                        ),
                        'up': DistributedTuckerLinear(
                            data['up_proj'], "gate_up", 
                            tp_size=current_tp_size, tp_rank=current_tp_rank
                        ),
                        'down': DistributedTuckerLinear(
                            data['down_proj'], "down", 
                            tp_size=current_tp_size, tp_rank=current_tp_rank
                        ),
                    })
                    
                    # 加载完成后立即从内存删除 data 对象，释放 CPU 内存
                    del data
                else:
                    if self.tp_rank == 0:
                        print(f"[TUCKER INIT] 警告：找不到文件 {f_path}，该专家组将无法运行")

            if self.tp_rank == 0:
                print(f"[TUCKER INIT] Layer {self.layer_idx} 加载完成，实际命中组数: {len(self.tucker_experts)}\n")
```

torchrun --nproc_per_node=4            

检查 test_tucker_layer.py：确认 initialize_model_parallel 在 DeepseekV2MoE 之前运行。
