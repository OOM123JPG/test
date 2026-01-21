# test
```
from vllm.distributed import (get_tensor_model_parallel_rank, 
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

class DistributedTuckerLinear(nn.Module):
    def __init__(self, data, proj_type="gate_up"):
        super().__init__()
        self.proj_type = proj_type
        
        # 必须在初始化模型前已经调过 initialize_model_parallel
        tp_size = get_tensor_model_parallel_world_size() 
        tp_rank = get_tensor_model_parallel_rank()
        
        # 1. 加载因子 U_E 和 Core (CPU -> NPU)
        self.register_buffer('U_E', data['factors'][0].to(torch.bfloat16).to("npu"))
        self.register_buffer('core', data['core'].to(torch.bfloat16).to("npu"))

        # 2. 真正的分片加载：先在 CPU 切分，再上 NPU
        u_h_raw = data['factors'][1] # 此时在 CPU
        u_i_raw = data['factors'][2] # 此时在 CPU
        
        r_h, r_i = u_h_raw.shape[1], u_i_raw.shape[1]
        r_h_shard, r_i_shard = r_h // tp_size, r_i // tp_size

        # 只把属于自己的切片移到 NPU
        if proj_type == "gate_up":
            # U_I 和 U_H 都按 Rank 维度切分
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to("npu").contiguous())
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to("npu").contiguous())
        else:
            self.register_buffer('U_H', u_h_raw[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard].to(torch.bfloat16).to("npu").contiguous())
            self.register_buffer('U_I', u_i_raw[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard].to(torch.bfloat16).to("npu").contiguous())

    def forward(self, x, expert_inner_indices):
        # 这里的计算逻辑基本正确
        # combined_core: [batch, r_h, r_i]
        ue = self.U_E[expert_inner_indices]
        r_h, r_i = self.core.shape[1], self.core.shape[2]
        combined_core = torch.matmul(ue, self.core.view(self.core.shape[0], -1)).view(-1, r_h, r_i)

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        r_h_shard, r_i_shard = r_h // tp_size, r_i // tp_size

        if self.proj_type == "gate_up":
            # 1. 映射到低维: x[D] * U_I[D, r_i/4] -> x_low[r_i/4]
            x_low = torch.matmul(x, self.U_I)
            # 2. 与 Core 乘: 
            core_shard = combined_core[:, :, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard.transpose(1, 2)).squeeze(1)
            # 3. TP 同步
            mid = tensor_model_parallel_all_reduce(mid)
            # 4. 映射回高维
            mid_shard = mid[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard]
            return torch.matmul(mid_shard, self.U_H.T)
        else:
            # Down 投影逻辑同理
            x_low = torch.matmul(x, self.U_H)
            core_shard = combined_core[:, tp_rank * r_h_shard : (tp_rank + 1) * r_h_shard, :]
            mid = torch.bmm(x_low.unsqueeze(1), core_shard).squeeze(1)
            mid = tensor_model_parallel_all_reduce(mid)
            mid_shard = mid[:, tp_rank * r_i_shard : (tp_rank + 1) * r_i_shard]
            return torch.matmul(mid_shard, self.U_I.T)
```
torchrun --nproc_per_node=4            
