#!/usr/bin/env python3
# toy_tp_lowrank.py
import os
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter, nn, ops
from mindspore.ops import operations as P

RUN_VERSION = "toy_tp_lowrank_v1_col_parallel_outdim_allgather"

DTYPE = ms.float16
NP_DTYPE = np.float16


# ----------------------------
# 0) Context/device must be first
# ----------------------------
def setup_context_and_device():
    # If msrun sets DEVICE_ID, use it; otherwise map by RANK_ID
    if os.getenv("DEVICE_ID") is not None:
        device_id = int(os.getenv("DEVICE_ID"))
    else:
        rid = int(os.getenv("RANK_ID", "0"))
        device_id = rid  # within ASCEND_VISIBLE_DEVICES

    context.set_context(mode=context.GRAPH_MODE)
    ms.set_device("Ascend", device_id=device_id)
    return device_id


# ----------------------------
# 1) Dist init
# ----------------------------
def init_dist():
    from mindspore.communication import init, get_rank, get_group_size
    from mindspore import ParallelMode

    init()
    rank = get_rank()
    world = get_group_size()

    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,  # We are manually doing TP logic
        gradients_mean=False,
        device_num=world,
        global_rank=rank
    )
    return rank, world


# ----------------------------
# 2) Tuple Broadcast helpers
# ----------------------------
def bcast_tensor_from_rank0(t: Tensor):
    bcast = P.Broadcast(root_rank=0)
    (t_out,) = bcast((t,))   # tuple in/out
    return t_out


def bcast_numpy_from_rank0(arr: np.ndarray, rank: int):
    # rank0 has real data; others pass empty same-shape
    t = Tensor(arr)
    t = bcast_tensor_from_rank0(t)
    return t.asnumpy()


# ----------------------------
# 3) AllGather helper that concatenates on LAST dim
#    P.AllGather concatenates on dim0, so we add a leading dim=1
#    then reshape/transposes to [B, out]
# ----------------------------
class AllGatherLastDim(nn.Cell):
    def __init__(self):
        super().__init__()
        self.expand = ops.ExpandDims()
        self.allgather = P.AllGather()  # returns Tensor
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        # x: [B, shard]
        x = self.expand(x, 0)            # [1,B,shard]
        x = self.allgather(x)            # [tp,B,shard]  (gather along dim0)
        x = self.transpose(x, (1, 0, 2)) # [B,tp,shard]
        b, tp, shard = x.shape
        x = self.reshape(x, (b, tp * shard))  # [B,out]
        return x


# ----------------------------
# 4) Dense column-parallel linear (out-dim sharded)
#    W: [out, in] ; each rank stores W_shard: [out/TP, in]
# ----------------------------
class DenseLinearColTP(nn.Cell):
    def __init__(self, W_shard: np.ndarray):
        super().__init__()
        self.W = Parameter(Tensor(W_shard, dtype=DTYPE), requires_grad=False)  # [out_shard,in]
        self.matmul = ops.MatMul(transpose_b=True)  # x:[B,in], W:[out,in] -> x @ W^T = [B,out]

    def construct(self, x):
        return self.matmul(x, self.W)  # [B,out_shard]


# ----------------------------
# 5) LowRank column-parallel linear (out-dim sharded on U)
#    W ≈ U diag(S) V^T
#    - shard U: [out/TP, r]
#    - replicate S:[r], V:[in,r]
#    y_shard = (x @ V) * S @ U_shard^T
# ----------------------------
class LowRankLinearColTP(nn.Cell):
    def __init__(self, U_shard: np.ndarray, S: np.ndarray, V: np.ndarray):
        super().__init__()
        self.U = Parameter(Tensor(U_shard, dtype=DTYPE), requires_grad=False)  # [out_shard,r]
        self.S = Parameter(Tensor(S, dtype=DTYPE), requires_grad=False)        # [r]
        self.V = Parameter(Tensor(V, dtype=DTYPE), requires_grad=False)        # [in,r]

        self.matmul = ops.MatMul(transpose_b=False)
        self.matmul_Ut = ops.MatMul(transpose_b=True)  # z:[B,r], U:[out_shard,r] -> z @ U^T
        self.mul = ops.Mul()

    def construct(self, x):
        z = self.matmul(x, self.V)          # [B,r]
        z = self.mul(z, self.S)             # [B,r]
        y = self.matmul_Ut(z, self.U)       # [B,out_shard]
        return y


# ----------------------------
# 6) SVD (numpy)
# ----------------------------
def svd_factor(W: np.ndarray, r: int):
    # W: [out,in]
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)
    U_r = U[:, :r].astype(NP_DTYPE)         # [out,r]
    S_r = S[:r].astype(NP_DTYPE)            # [r]
    V_r = Vt[:r, :].T.astype(NP_DTYPE)      # [in,r]
    return U_r, S_r, V_r


# ----------------------------
# 7) Main: verify TP correctness
# ----------------------------
def main():
    device_id = setup_context_and_device()
    rank, tp = init_dist()
    assert tp == 2, f"this demo expects tp=2, but got world_size={tp}"

    ms.set_seed(0)
    np.random.seed(0)

    print("RUN_VERSION =", RUN_VERSION)
    print("ASCEND_VISIBLE_DEVICES =", os.getenv("ASCEND_VISIBLE_DEVICES"),
          "DEVICE_ID =", os.getenv("DEVICE_ID"),
          "RANK_ID =", os.getenv("RANK_ID"),
          "resolved_device_id =", device_id)

    # ---- shapes ----
    B = 16
    in_dim = 256
    out_dim = 512
    r = 64  # change to min(out,in) for near-exact (sanity), e.g., r=256 here

    assert out_dim % tp == 0
    out_shard = out_dim // tp

    # ---- rank0 creates W and x, broadcast to others ----
    if rank == 0:
        W = (np.random.randn(out_dim, in_dim) * 0.02).astype(NP_DTYPE)
        x = (np.random.randn(B, in_dim) * 0.1).astype(NP_DTYPE)
    else:
        W = np.empty((out_dim, in_dim), dtype=NP_DTYPE)
        x = np.empty((B, in_dim), dtype=NP_DTYPE)

    W = bcast_numpy_from_rank0(W, rank)
    x = bcast_numpy_from_rank0(x, rank)

    # ---- build SVD factors (each rank can compute; deterministic) ----
    U, S, V = svd_factor(W, r)

    # ---- shard weights for TP ----
    # Dense baseline shard: W_shard = W[rank*out_shard : (rank+1)*out_shard, :]
    W_sh = W[rank*out_shard:(rank+1)*out_shard, :]

    # Lowrank shard: U_shard = U[rank*out_shard : (rank+1)*out_shard, :]
    U_sh = U[rank*out_shard:(rank+1)*out_shard, :]

    # ---- build networks ----
    dense_tp = DenseLinearColTP(W_sh)
    lowrank_tp = LowRankLinearColTP(U_sh, S, V)
    gather_last = AllGatherLastDim()

    x_t = Tensor(x, dtype=DTYPE)

    # ---- forward ----
    y_dense_sh = dense_tp(x_t)                # [B,out_shard]
    y_lr_sh = lowrank_tp(x_t)                 # [B,out_shard]

    y_dense = gather_last(y_dense_sh)         # [B,out_dim]
    y_lr = gather_last(y_lr_sh)               # [B,out_dim]

    # ---- error ----
    y_dense_np = y_dense.asnumpy().astype(np.float32)
    y_lr_np = y_lr.asnumpy().astype(np.float32)

    diff = y_lr_np - y_dense_np
    rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(y_dense_np) + 1e-8))
    max_abs = float(np.max(np.abs(diff)))

    print(f"[rank {rank}/{tp}] r={r} rel_err={rel_err:.6f}, max_abs={max_abs:.6f}")


if __name__ == "__main__":
    main()
