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
# 在 DeepseekV2MoE.__init__ 内部
if self.is_tucker_active:
    # 此时 initialize_model_parallel 已经在脚本里调过了，这里获取是安全的
    current_tp_size = get_tensor_model_parallel_world_size()
    current_tp_rank = get_tensor_model_parallel_rank()

    for g_id in my_group_ids:
        # ...
        if os.path.exists(f_path):
            data = torch.load(f_path, map_location='cpu')
            self.tucker_experts[str(g_id)] = nn.ModuleDict({
                'gate': DistributedTuckerLinear(data['gate_proj'], "gate_up", 
                                               tp_size=current_tp_size, tp_rank=current_tp_rank),
                'up':   DistributedTuckerLinear(data['up_proj'], "gate_up", 
                                               tp_size=current_tp_size, tp_rank=current_tp_rank),
                'down': DistributedTuckerLinear(data['down_proj'], "down", 
                                               tp_size=current_tp_size, tp_rank=current_tp_rank),
            })
```

torchrun --nproc_per_node=4            

检查 test_tucker_layer.py：确认 initialize_model_parallel 在 DeepseekV2MoE 之前运行。
