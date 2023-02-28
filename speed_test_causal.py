import torch
from torch.autograd import gradcheck

from src import TnoMatrix, TnoFFT, TnoCausalV1


def get_model_name(model):
    name = str(type(model)).split(".")[-2].split("'")[0]

    return name


def speed_test(b, n, d, file="tmp"):
    ###### data initialize
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
    ]

    ###### warmup
    for _ in range(10):
        models[0](x, t)

    ##### forward test
    print("Forward test:")
    y_res = []
    for model in models:
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            y_res.append(model(x, t))
        print(f'{get_model_name(model)} forward\n', prof.key_averages(group_by_stack_n=5).table(
                sort_by='self_cuda_time_total', row_limit=5))

    ##### backward test
    print("Backward test:")
    n = len(y_res)
    for i in range(n):
        if t.grad != None:
            t.grad.data.zero_()
        if x.grad != None:
            x.grad.data.zero_()

        loss = (y_res[i] ** 2).sum()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            loss.backward()
            
        print(f'{get_model_name(models[i])} backward\n', prof.key_averages(group_by_stack_n=5).table(
                sort_by='self_cuda_time_total', row_limit=5))
        
# speed_test(2, 16, 32)
speed_test(b=8, n=2048, d=64)