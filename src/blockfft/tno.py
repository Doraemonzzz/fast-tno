import torch
import torch.nn as nn
from .block_fft import BlockFFT


class TnoBlockFFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_fft = BlockFFT(learn_dft_matrices=False)

    def forward(self, x, t, m=None):
        """_summary_

        Args:
            x (Tensor): b, n, d;
            t (Tensor): 2 * max(n, m), d;
                        t0, t1, ..., t(n-1), t0, t-(n-1), ... , t-1;
            m (Tensor): output dimension;

        Returns:
            o: b, m, d;
        """
        n = x.shape[1]
        if m == None:
            m = n
        l = max(n, m)
        t = t.unsqueeze(0)
        x = x.transpose(-1, -2)
        t = t.transpose(-1, -2)
        x_fft = self.block_fft(x.to(torch.complex64), N=2 * l)
        t_fft = self.block_fft(t.to(torch.complex64), N=2 * l)
        o_fft = x_fft * t_fft
        o = self.block_fft(o_fft, N=2 * l, forward=False).real[..., :l]
        o = o.transpose(-1, -2)

        return o
