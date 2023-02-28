import torch
import sys
from torch.autograd import gradcheck

from src import TnoMatrix, TnoFFT, TnoCausalV1, TnoCausalV2, TnoCausalV3


def get_model_name(model):
    name = str(type(model)).split(".")[-1].split("'")[0]

    return name


def speed_test(b, n, d):
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
        TnoCausalV2().cuda(),
        TnoCausalV3().cuda(),
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
        print(
            f"{get_model_name(model)} forward\n",
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=5
            ),
        )

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

        print(
            f"{get_model_name(models[i])} backward\n",
            prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=5
            ),
        )


# speed_test(2, 16, 32)
b = 8
d = 64
for n in [128, 256, 512, 1024, 2048, 4096]:
    fb = open(f"log/n_test_{n}.log", "w")
    sys.stdout = fb
    speed_test(b, n, d)
    fb.close()
