#include<stdio.h>
#define B 2
#define N 64
#define D 8

template<typename F>
__global__ void lower_kernel(const int b, const int d, const int n, const F* T, const F* x, F* y) {
    /**
    input:
        T: n, d [t0, t1, ..., t_(n-1)]
        x: b, n, d

    output:
        y: b, n, d
    **/
    int b_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_ >= b) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.z * blockDim.z + threadIdx.z;
    if (d_ >= d) {
        return;
    }
    int t_offset = d_;
    int x_offset = b_ * d * n + d_;

    // forward
    __shared__ F T_shared[N][D];
    __shared__ F x_shared[B][N][D];
    T_shared[i % N][d_ % D] = T[t_offset + i * d];
    x_shared[b_ % B][i % N][d_ % D] = x[x_offset + i * d];
    __syncthreads();

    int b1 = (b_ / B) * B;
    int i1 = (i / N) * N;
    int d1 = (d_ / D) * D;

    F s = 0;
    for (int j = 0; j <= i; j++) {
        if ((i1 <= i - j && i - j < i1 + N) && (d1 <= d_ && d_ < d1 + D)) {
            if ((b1 <= b_ && b_ < b1 + B) && (i1 <= j && j < i1 + N)) {
                s += T_shared[(i - j) % N][d_ % D] * x_shared[b_ % B][j % N][d_ % D];
            } else {
                s += T_shared[(i - j) % N][d_ % D] * x[x_offset + j * d];
            }
        } else {
            if ((b1 <= b_ && b_ < b1 + B) && (i1 <= j && j < i1 + N)) {
                s += T[t_offset + (i - j) * d] * x_shared[b_ % B][j % N][d_ % D];
            } else {
                s += T[t_offset + (i - j) * d] * x[x_offset + j * d];
            }
        }
    }

    y[x_offset + i * d] = s;
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
    int b_ = blockIdx.x * blockDim.x + threadIdx.x;
    if (b_ >= b) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.z * blockDim.z + threadIdx.z;
    if (d_ >= d) {
        return;
    }
    int t_offset = d_;
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

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    dim3 DimGrid((b + B - 1) / B, (n + N - 1) / N, (d + D - 1) / D);
    dim3 DimBlock(B, N, D);
    lower_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid((b + B - 1) / B, (n + N - 1) / N, (d + D - 1) / D);
    dim3 DimBlock(B, N, D);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}