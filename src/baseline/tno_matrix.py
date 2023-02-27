import torch
import torch.nn as nn

class TnoMatrix(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, t):
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
        zero = t[:, 0, None]
        pos = t[:, 1: n]
        neg = t[:, n + 1:]
        c = torch.cat([zero, pos], dim=-2)
        r = torch.cat([zero, neg.flip(1)], dim=-2)
        vals = torch.cat([r, c[:, 1:].flip(1)], dim=-2)
        n = c.shape[-2]
        shape = n, n
        i, j = torch.ones(n, n).nonzero().T
        t_matrix = vals[:, j - i].reshape(n, n, -1)

        o = torch.einsum('n m d, b m d -> b n d', t_matrix, x)
        
        return o