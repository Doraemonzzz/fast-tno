#include<stdio.h>

template<typename F>
__global__ void lower_kernel(const int b, const int d, const int n, const F* T, const F* x, F* y) {
    /**
    input:
        T: d, n, [t0, t1, ..., t_(n-1)]
        x: b, d, n

    output:
        y: b, d, n
    **/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * d) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) {
        return;
    }
    int b_ = idx / d;
    int d_ = idx % d;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s = 0;
    for (int j = 0; j <= i; j++) {
        s += T[t_offset + i - j] * x[x_offset + j];
    }
    y[x_offset + i] = s;
}

template<typename F>
__global__ void backward_kernel(const int b, const int d, const int n, const F* T, const F* x, const F* gy, F* gT, F* gx) {
    /**
    input:
        T: d, n, [t0, t1, ..., t_(n-1)]
        x: b, d, n
        gy: b, d, n

    output:
        gT: b, d, n
        gx: b, d, n
    **/
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= b * d) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) {
        return;
    }
    int b_ = idx / d;
    int d_ = idx % d;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s_x = 0;
    F s_T = 0;
    for (int j = 0; j < n - i; j++) {
        s_x += T[t_offset + j] * gy[x_offset + i + j];
        s_T += x[x_offset + j] * gy[x_offset + i + j];
    }
    gx[x_offset + i] = s_x;
    gT[x_offset + i] = s_T;
}

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    int c = 32;
    dim3 DimGrid((b * d + c - 1) / c, (n + c - 1) / c);
    dim3 DimBlock(c, c);
    lower_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    int c = 32;
    dim3 DimGrid((b * d + c - 1) / c, (n + c - 1) / c);
    dim3 DimBlock(c, c);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}