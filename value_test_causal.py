import torch
from torch.autograd import gradcheck

from src import TnoMatrix, TnoFFT, TnoCausalV1, TnoCausalV2, TnoCausalV3


def get_model_name(model):
    name = str(type(model)).split(".")[-1].split("'")[0]

    return name


###### data initialize
b = 2
n = 16
d = 32

t_zero = torch.randn(1, d)
t_pos = torch.randn(n - 1, d)
# t_neg = torch.zeros(n - 1, d)
# t = torch.cat([t_zero, t_pos, t_zero, t_neg.flip(0)], dim=0).cuda().requires_grad_()
t = torch.cat([t_zero, t_pos], dim=0).cuda().requires_grad_()
x = torch.randn(b, n, d).cuda().requires_grad_()

###### model initialize
models = [
    TnoFFT().cuda(),
    TnoMatrix(causal=True).cuda(),
    TnoCausalV1().cuda(),
    TnoCausalV2().cuda(),
    TnoCausalV3().cuda(),
]


##### forward test
y_res = []
for model in models:
    y_res.append(model(x, t))

print("Forward test:")
n = len(y_res)
for i in range(1, n):
    print(
        f"Differ Norm between {get_model_name(models[i])} and {get_model_name(models[0])}: {torch.norm(y_res[i] - y_res[0]).item()}"
    )

##### backward test
t_grad_res = []
x_grad_res = []
for i in range(n):
    if t.grad != None:
        t.grad.data.zero_()
    if x.grad != None:
        x.grad.data.zero_()

    loss = (y_res[i] ** 2).sum()
    loss.backward()

    t_grad_res.append(t.grad.data.clone())
    x_grad_res.append(x.grad.data.clone())

print("Backward test:")
print("T grad test:")
for i in range(1, n):
    print(
        f"Differ Norm between {get_model_name(models[i])} and {get_model_name(models[0])}: {torch.norm(t_grad_res[i] - t_grad_res[0]).item()}"
    )

print("x grad test:")
for i in range(1, n):
    print(
        f"Differ Norm between {get_model_name(models[i])} and {get_model_name(models[0])}: {torch.norm(x_grad_res[i] - x_grad_res[0]).item()}"
    )
