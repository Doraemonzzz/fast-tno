import torch

from tno_fft_naive import TnoFft
from tno_naive import TnoNaive

b = 2
n = 100
d = 512

x = torch.randn(b, n, d)
t_zero = torch.randn(1, 1, d)
t_pos = torch.randn(1, n - 1, d)
t_neg = torch.randn(1, n - 1, d)
t = torch.cat([t_zero, t_pos, t_zero, t_neg.flip(1)], dim=1)

tno_fft = TnoFft()
tno_naive = TnoNaive()

output = []
print(torch.norm(tno_fft(x, t, n) - tno_naive(x, t)))