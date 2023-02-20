import torch
import torch.nn as nn

class TnoFft(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, t, m):
        """_summary_

        Args:
            x (Tensor): b, n, d;
            t (Tensor): 1, 2 * max(n, m) - 2, d; 
                        t0, t1, ..., t(n-1), t0, t-(n-1), ... , t-1;
            m (Tensor): output dimension;

        Returns:
            o: b, m, d;
        """
        n = x.shape[1]
        l = max(n, m)
        x_fft = torch.fft.rfft(x, 2 * l, dim=-2)
        t_fft = torch.fft.rfft(t, 2 * l, dim=-2)
        o_fft = x_fft * t_fft
        o = torch.fft.irfft(o_fft, 2 * l, dim=-2)[:, :m]

        return o