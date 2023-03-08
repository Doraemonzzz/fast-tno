import torch
import torch.nn as nn

# from torch.utils.cpp_extension import load

import math

import os

# mathdx_dir = os.environ['mathdx_dir']

from fftconv import fftconv_fwd, fftconv_bwd

# ninja build does not work unless include_dirs are abs path
# this_dir = os.path.dirname(os.path.abspath(__file__))
# cuda_dir = os.environ["CUDA_HOME"]

# cub_dir = os.environ["CUB_HOME"]

# tno_causal_cuda = load(
#     name="tno_causal_v12",
#     sources=["src/fftconv/fftconv.cpp", "src/fftconv/fftconv_cuda.cu"],
#     # extra_include_paths=[f"{mathdx_dir}/mathdx/22.02/include"],
#     extra_include_paths=[os.path.join(this_dir, 'mathdx/22.02/include'),
#                          os.path.join(cuda_dir, "targets/x86_64-linux/include")
#                         #  f"{cub_dir}",
#                         #  os.path.join(this_dir, 'cub-1.17.2')
#                          ],
#     extra_cflags=['-g', '-march=native', '-funroll-loops'],
#     extra_cuda_cflags=['-O3', '--threads', '4', '-lineinfo', '--use_fast_math', '-std=c++17', '-arch=compute_70'],
#     # extra_include_paths=["/usr/local/cuda-11.2/targets/x86_64-linux/include/cuda/std/detail/libcxx/include",
#     #                      "/usr/local/cuda-11.2/targets/x86_64-linux/include/thrust/detail/complex"],
#     verbose=True,
# )

class FFTConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, k, D=None, dropout_mask=None, gelu=True, force_fp16_output=False,
                output_hbl_layout=False, v=None, head_dim=1, q=None, fftfp16=False, k_rev=None):
        # k: n, d
        # u: b, n, d
        # n, d -> d, n
        k = k.transpose(1, 0).contiguous()
        # b, n, d -> b, d, n
        u = u.transpose(2, 1).contiguous()
        
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        k_f = torch.fft.rfft(k, n=fft_size)
        
        ctx.save_for_backward(u, k_f, D, dropout_mask, v, q)
        ctx.output_hbl_layout = output_hbl_layout
        ctx.head_dim = head_dim
        ctx.gelu = gelu
        ctx.fftfp16 = fftfp16
        ctx.has_k_rev = k_rev is not None
        out = fftconv_fwd(u, k_f, D, v, head_dim, q, dropout_mask, gelu, False, False, fft_size, force_fp16_output, output_hbl_layout, fftfp16)
        # b, d, n -> b, n, d
        out = out.transpose(2, 1).contiguous()
        
        return out

    @staticmethod
    def backward(ctx, dout):
        # b, n, d -> b, d, n
        dout = dout.transpose(2, 1).contiguous()
        
        u, k_f, D, dropout_mask, v, q = ctx.saved_tensors
        seqlen = u.shape[-1]
        fft_size = max(2 * 2 ** int(math.ceil(math.log2(seqlen))), 16)
        du, dk_f, dD, dv, dq = fftconv_bwd(dout, u, k_f, D, v, ctx.head_dim, q, dropout_mask, ctx.gelu, False, False, fft_size,
                                   ctx.output_hbl_layout, ctx.fftfp16)
        dk = torch.fft.irfft(dk_f, n=fft_size, norm='forward')[..., :seqlen]
        dk_rev = (None if not ctx.has_k_rev
                  else torch.fft.irfft(dk_f.conj(), n=fft_size, norm='forward')[..., :seqlen])
        # d, n -> n, d
        dk = dk.transpose(1, 0).contiguous()
        # b, d, n -> b, n, d
        du = du.transpose(2, 1).contiguous()

        return du, dk, dD, None, None, None, None, dv if v is not None else None, None, dq if q is not None else None, None, dk_rev

class TnoCausalV12(nn.Module):
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

        return FFTConvFunc.apply(x, t)
