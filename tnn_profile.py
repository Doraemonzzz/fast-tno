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
n = 2048
# embedding size
e = 512
# rpe embedding size
d = 32

print("======Start Test Tno=====")
x = torch.rand(b, n, e).cuda()
model = TnnLayer(
    dim=e,
    num_heads=1,
    rpe_embedding=d,
    glu_dim=e,
    norm_type="layernorm",
).cuda()

print(model)

with torch.autograd.profiler.profile(use_cuda=True, profile_memory=True, record_shapes=True, with_stack=True) as prof:
    # with record_function("Gtu"):
        # model(x)
    y = model(x)
print(
    f"{get_model_name(model)} forward\n",
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=10
    ),
)

# prof.export_chrome_trace("trace.json")
# prof.export_stacks("cuda.json", "self_cuda_time_total")

loss = (y ** 2).sum()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    loss.backward()

print(
    f"{get_model_name(model)} backward\n",
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=5
    ),
)