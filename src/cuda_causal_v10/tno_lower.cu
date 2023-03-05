#include<stdio.h>
#define B 2
#define N 16
#define D 16

template<typename F>
__global__ void forward_kernel(
    const int b, 
    const int d, 
    const int n, 
    const F* T, 
    const F* x, 
    F* y
) {
    /**
    input:
        T: d, n, [t_(-(n-1)), ..., t_(-1), t0, t_1, ..., t_(n-1)]
        x: b, d, n

    output:
        y: b, d, n
    **/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_ >= d) {
        return;
    }
    int b_ = blockIdx.z * blockDim.z + threadIdx.z;
    if (b_ >= b) {
        return;
    }

    int l = (n + N - 1) / N;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s_y = 0;
    for (int j_block = 0; j_block < l; j_block++) {
        // j < n, 0 <= i - j < n
        // i - n < j <= min(i, n - 1)
        for (int j_thread = 0; j_thread < N; j_thread++) {
            int j = j_block * N + j_thread;
            if (j >= n) {
                break;
            }
            int k = i - j;
            if (k >= n || k < 0) {
                break;
            }
            s_y += T[t_offset + k] * x[x_offset + j];
        }
    }

    y[x_offset + i] = s_y;
}

template<typename F>
__global__ void backward_kernel(
    const int b, 
    const int d, 
    const int n, 
    // const int s,
    // const int e,
    const F* T, 
    const F* x, 
    const F* gy, 
    F* gT, 
    F* gx
) {
    /**
    input:
        T: d, n, [t0, t_1, ..., t_(n-1)]
        x: b, d, n, [x0, x_1, ..., x_(n-1)]
        gy: b, d, n

    output:
        gT: b, d, n
        gx: b, d, n
    **/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    int d_ = blockIdx.y * blockDim.y + threadIdx.y;
    if (d_ >= d) {
        return;
    }
    int b_ = blockIdx.z * blockDim.z + threadIdx.z;
    if (b_ >= b) {
        return;
    }
    // number of block
    int l = (n + N - 1) / N;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s_x = 0;
    F s_T = 0;
    for (int j_block = 0; j_block < l; j_block++) {
        for (int j_thread = 0; j_thread < N; j_thread++) {
            int j = j_block * N + j_thread;
            if (j >= n) {
                break;
            }
            int k = j - i;
            if (k >= n || k < 0) {
                continue;
            }

            s_x += T[t_offset + k] * gy[x_offset + j];
            s_T += x[x_offset + k] * gy[x_offset + j];
        }
    }

    gx[x_offset + i] = s_x;
    gT[x_offset + i] = s_T;
}

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    forward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}