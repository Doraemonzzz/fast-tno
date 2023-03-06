import torch
import torch.nn as nn

from torch.utils.cpp_extension import load

# from fftconv import fftconv_fwd, fftconv_bwd

tno_causal_cuda = load(
    name="tno_causal_v12",
    sources=["src/fftconv/fftconv.cpp", "src/fftconv/fftconv_cuda.cu"],
    extra_include_paths=["/usr/local/cuda-11.2/targets/x86_64-linux/include/cuda/std/detail/libcxx/include",
                         "/usr/local/cuda-11.2/targets/x86_64-linux/include/thrust/detail/complex"],
    verbose=True,
)

class TnoCausal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T, x, dropout_mask=None, gelu=True, force_fp16_output=False,
                output_hbl_layout=False, v=None, head_dim=1, q=None, fftfp16=False, k_rev=None):
        # T_fft: n, d
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
        T_fft = torch.fft.rfft(T, dim=-1)
        
        ctx.save_for_backward(x, T_fft, dropout_mask, v, q)
        ctx.output_hbl_layout = output_hbl_layout
        ctx.head_dim = head_dim
        ctx.gelu = gelu
        ctx.fftfp16 = fftfp16
        ctx.has_k_rev = k_rev is not None
        y = fftconv_fwd(x, T_fft, D, v, head_dim, q, dropout_mask, gelu, False, False, fft_size, force_fp16_output, output_hbl_layout, fftfp16)
        # b, d, n -> b, n, d
        y = y.transpose(2, 1).contiguous()

        return y

    @staticmethod
    def backward(ctx, gy):
        x, T_fft, dropout_mask, v, q = ctx.saved_tensors
        seqlen = x.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        dx, dT_fft, dv, dq = fftconv_bwd(dout, x, T_fft, v, ctx.head_dim, q, dropout_mask, ctx.gelu, False, False, fft_size,
                                   ctx.output_hbl_layout, ctx.fftfp16)
        dT = torch.fft.irfft(dT_fft, n=fft_size)[..., :seqlen]
        dk_rev = None
        dv = None
        # d, n -> n, d
        dT = dT.transpose(1, 0).contiguous()
        # b, d, n -> b, n, d
        dx = dx.transpose(2, 1).contiguous()
        
        return dT, dx, None, None, None, None, dv if v is not None else None, None, dq if q is not None else None, None, dk_rev


class TnoCausalV11(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        """_summary_

        Args:
            x (Tensor): b, n, d;
            t (Tensor): n, d;
                        t0, t_1, ..., t_(n-1);

        Returns:
            o: b, n, d;
        """

        return TnoCausal.apply(t, x)
