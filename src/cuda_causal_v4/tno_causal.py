import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

tno_causal_cuda = load(
    name="tno_causal_v4",
    sources=["src/cuda_causal_v4/tno_lower.cpp", "src/cuda_causal_v4/tno_lower.cu"],
    verbose=True,
)


class TnoCausal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T, x):
        # T: n, d
        # x: b, n, d
        b, n, d = x.shape
        ctx.b = b
        ctx.d = d
        ctx.n = n
        ctx.save_for_backward(T, x)
        y = torch.empty(
            (b, n, d), device=x.device, memory_format=torch.contiguous_format
        )
        tno_causal_cuda.forward(b, d, n, T, x, y)

        return y

    @staticmethod
    def backward(ctx, gy):
        # gy: b, n, d
        b = ctx.b
        d = ctx.d
        n = ctx.n
        T, x = ctx.saved_tensors
        gT = torch.empty((b, n, d), device=gy.device)
        gx = torch.empty((b, n, d), device=gy.device)
        tno_causal_cuda.backward(b, d, n, T, x, gy, gT, gx)
        gT = torch.sum(gT, dim=0)

        return gT, gx


class TnoCausalV4(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        """_summary_

        Args:
            x (Tensor): b, n, d;
            t (Tensor): n, d;
                        t0, t1, ..., t(n-1);

        Returns:
            o: b, n, d;
        """

        return TnoCausal.apply(t, x)
