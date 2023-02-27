import torch

from src import TnoMatrix, TnoFFT, TnoCausalV1

b = 1
n = 2
d = 3

x = torch.randn(b, n, d).cuda()
t_zero = torch.randn(1, d)
t_pos = torch.randn(n - 1, d)
t_neg = torch.zeros(n - 1, d)
t = torch.cat([t_zero, t_pos, t_zero, t_neg.flip(0)], dim=0).unsqueeze(0).cuda()
t_causal = torch.cat([t_zero, t_pos], dim=0).cuda()

tno_fft = TnoFFT().cuda()
tno_matrix = TnoMatrix().cuda()
tno_causal_v1 = TnoCausalV1().cuda()

x1 = tno_fft(x, t, n)
x2 = tno_causal_v1(x, t_causal)
x3 = tno_matrix(x, t)

print(torch.norm(x1 - x2))
print(torch.norm(x1 - x3))