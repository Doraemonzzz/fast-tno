import torch
from torch.autograd import gradcheck

from src import TnoMatrix, TnoFFT, TnoCausalV1

b = 1
n = 2
d = 3

x = torch.randn(b, n, d).cuda().requires_grad_()
y = torch.randn(b, n, d).cuda().requires_grad_()
t_zero = torch.randn(1, d)
t_pos = torch.randn(n - 1, d)
t_neg = torch.zeros(n - 1, d)
t = torch.cat([t_zero, t_pos, t_zero, t_neg.flip(0)], dim=0).unsqueeze(0).cuda().requires_grad_()
t_causal = torch.cat([t_zero, t_pos], dim=0).cuda().requires_grad_()


tno_fft = TnoFFT().cuda()
tno_matrix = TnoMatrix().cuda()
tno_causal_v1 = TnoCausalV1().cuda()

y1 = tno_fft(x, t, n)
y2 = tno_matrix(x, t)
y3 = tno_causal_v1(x, t_causal)

gradcheck(tno_causal_v1, (x, t_causal))

# y1.backward(y)
# y2.backward(y)
# y3.backward(y)

print(torch.norm(y1 - y2))
print(torch.norm(y1 - y3))

print(torch.norm(y1.grad - y2.grad))

