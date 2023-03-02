#include<stdio.h>
#define N 32
#define D 32

template<typename F>
__global__ void lower_kernel(const int b, const int d, const int n, const F* T, const F* x, F* y) {
    /**
    input:
        T: n, d [t0, t1, ..., t_(n-1)]
        x: b, n, d

    output:
        y: b, n, d
    **/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_ >= d) {
        return;
    }
    int t_offset = d_;

    for (int b_ = 0; b_ < b; b_++) {
        F s = 0;
        int x_offset = b_ * d * n + d_;
        for (int j = 0; j <= i; j++) {
            s += T[t_offset + (i - j) * d] * x[x_offset + j * d];
        }
        y[x_offset + i * d] = s;
    }
}

template<typename F>
__global__ void backward_kernel(const int b, const int d, const int n, const F* T, const F* x, const F* gy, F* gT, F* gx) {
    /**
    input:
        T: n, d, [t0, t1, ..., t_(n-1)]
        x: b, n, d
        gy: b, n, d

    output:
        gT: b, n, d
        gx: b, n, d
    **/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_ >= d) {
        return;
    }
    int t_offset = d_;

    for (int b_ = 0; b_ < b; b_++) {
        int x_offset = b_ * d * n + d_;
        F s_x = 0;
        F s_T = 0;
        for (int j = 0; j < n - i; j++) {
            s_x += T[t_offset + j * d] * gy[x_offset + (i + j) * d];
            s_T += x[x_offset + j * d] * gy[x_offset + (i + j) * d];
        }
        gx[x_offset + i * d] = s_x;
        gT[x_offset + i * d] = s_T;
    }
}

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D);
    dim3 DimBlock(N, D);
    lower_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D);
    dim3 DimBlock(N, D);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}