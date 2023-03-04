#include <torch/extension.h>

void forward_cuda(int b, int d, int n, float* T, float* x, float* y);
void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx);

void forward(int64_t b, int64_t d, int64_t n, torch::Tensor &T, torch::Tensor &x, torch::Tensor &y) {
    forward_cuda(b, d, n, T.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>());
}

void backward(int64_t b, int64_t d, int64_t n, torch::Tensor &T, torch::Tensor &x, torch::Tensor &gy, torch::Tensor &gT, torch::Tensor &gx) {
    backward_cuda(b, d, n, T.data_ptr<float>(), x.data_ptr<float>(), gy.data_ptr<float>(), gT.data_ptr<float>(), gx.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "tno forward");
    m.def("backward", &backward, "tno backward");
}

TORCH_LIBRARY(tno_causal_v10, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}