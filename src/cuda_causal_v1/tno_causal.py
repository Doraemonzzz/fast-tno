import torch
import torch.nn as nn

from torch.utils.cpp_extension import load
tno_causal_cuda = load(
    name="tno_causal", 
    sources=["src/cuda_causal_v1/tno_lower.cpp", "src/cuda_causal_v1/tno_lower.cu"],
    verbose=True, 
    # extra_cuda_cflags=['--use_fast_math', '--extra-device-vectorization'], 
    # extra_cflags=['/wd4624']
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
        # must add contiguous, or the memory may be wrong!!!
        # n, d -> d, n
        T = T.transpose(1, 0).contiguous()
        # b, n, d -> b, d, n
        x = x.transpose(2, 1).contiguous()
        ctx.save_for_backward(T, x)
        y = torch.empty((b, d, n), device=x.device, memory_format=torch.contiguous_format)
        tno_causal_cuda.forward(b, d, n, T, x, y)
        # b, d, n -> b, n, d
        y = y.transpose(2, 1).contiguous()

        return y

    @staticmethod
    def backward(ctx, gy):
        b = ctx.b
        d = ctx.d
        n = ctx.n
        T, x = ctx.saved_tensors
        gT = torch.empty((b, d, n), device=gy.device)
        gx = torch.empty((b, d, n), device=gy.device)
        # b, n, d -> b, d, n
        gy = gy.transpose(2, 1).contiguous()
        tno_causal_cuda.backward(b, d, n, T, x, gy, gT, gx)
        gT = torch.sum(gT, dim=0)
        # d, n -> n, d
        gT = gT.transpose(1, 0).contiguous()
        # b, d, n -> b, n, d
        gx = gx.transpose(2, 1).contiguous()

        return gT, gx
    
class TnoCausalV1(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, t):
        """_summary_

        Args:
            x (Tensor): b, n, d;
            t (Tensor): b, d;
                        t0, t1, ..., t(n-1);

        Returns:
            o: b, n, d;
        """
        
        return TnoCausal.apply(t, x)