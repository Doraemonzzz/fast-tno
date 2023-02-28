import torch
import torch.nn as nn

##### to be fixed
class TnoNaive(nn.Module):
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
        o = torch.zeros_like(x).to(x)
        b, n, d = x.shape
        for i in range(b):
            for j in range(d):
                for u in range(n):
                    d = 0
                    for v in range(u + 1):
                        d += t[v][j] * x[i][j][v]
                    t[u][j] = d

        return o
