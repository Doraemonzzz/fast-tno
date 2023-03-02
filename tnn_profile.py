import torch
from tnn_pytorch import TnnLayer
from torch.profiler import profile, record_function

def get_model_name(model):
    name = str(type(model)).split(".")[-1].split("'")[0]

    return name

# batch size
b = 2
# number of head
h = 1
# sequce length
n = 10
# embedding size
e = 4
# rpe embedding size
d = 16

print("======Start Test Tno=====")
x = torch.rand(b, n, e).cuda()
model = TnnLayer(
    dim=e,
    num_heads=1,
    rpe_embedding=d,
    glu_dim=e,
).cuda()

print(model)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    # with record_function("TnnLayer"):
    model(x)
print(
    f"{get_model_name(model)} forward\n",
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=10
    ),
)
