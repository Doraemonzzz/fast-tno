#include<stdio.h>
#define B 2
#define N 32
#define D 16

template<typename F>
__global__ void forward_kernel(
    const int b, 
    const int d, 
    const int n, 
    const int s,
    const int e,
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
    // [-(n-1), (n-1)] -> [0, 2n-2]
    // int i_block = blockIdx.x;
    // int i_thread = threadIdx.x;
    // number of block
    int l = (n + N - 1) / N;
    // // block level index
    // int k_block = i_block - j_block + l - 1;
    // // thread level index
    // int k_thread = i_thread - j_thread + N - 1;
    // // batch global index
    // int b_ = idx / d;
    // int b_block = b_ / B;
    // int b_thread = b_ % B;
    // // feature global index
    // int d_ = idx % d;
    // int d_block = d_ / D;
    // int d_thread = d_ % D;

    // int start = max(s - i, 0);
    // int end = min(e - i, n);
    // int i1 = blockIdx.x;
    // int j1 = blockIdx.y;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s_y = 0;
    for (int j_block = 0; j_block < l; j_block++) {
        // int k_block = i_block - j_block + l - 1;
        for (int j_thread = 0; j_thread < N; j_thread++) {
            int j = j_block * N + j_thread;
            // printf("%d %d %d %d %d\n", b, d, n, s, e);
            if (j >= n) {
                break;
            }
            // int k_thread = i_thread - j_thread + N - 1;
            // int k = k_block * N + k_thread - s;
            int k = i - j + n - 1;
            // printf("%d %d %d %d\n", k_block, k_thread, k, e);
            // printf("%d %d %d %d %d\n", i, j, k, s, e);
            if (k >= e || k < s) {
                break;
            }
            k -= s;
            // printf("%d %d\n", k, 2 * n - 1);
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
    const int s,
    const int e,
    const F* T, 
    const F* x, 
    const F* gy, 
    F* gT, 
    F* gx
) {
    /**
    input:
        T: d, n, [t_(-(n-1)), ..., t_(-1), t0, t_1, ..., t_(n-1)]
        x: b, d, n
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
    // [-(n-1), (n-1)] -> [0, 2n-2]
    // int i_block = blockIdx.x;
    // int i_thread = threadIdx.x;
    // number of block
    int l = (n + N - 1) / N;

    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    F s_x = 0;
    F s_T = 0;
    for (int j_block = 0; j_block < l; j_block++) {
        // int k_block = i_block - j_block + l - 1;
        for (int j_thread = 0; j_thread < N; j_thread++) {
            int j = j_block * N + j_thread;
            // printf("%d %d %d %d %d\n", b, d, n, s, e);
            if (j >= n) {
                break;
            }
            // int k_thread = i_thread - j_thread + N - 1;
            // int k = k_block * N + k_thread - s;
            int k = j - i + n - 1;
            // printf("%d %d %d %d\n", k_block, k_thread, k, e);
            // printf("%d %d %d %d %d\n", i, j, k, s, e);
            if (k >= e || k < s) {
                break;
            }
            // printf("%d %d %d %d %d\n", i, j, k, s, e);
            k -= s;
            // printf("%d %d\n", k, 2 * n - 1);
            // printf("%d %d %d\n", k, i, j);
            // printf("%lf %lf\n", T[t_offset + k], gy[x_offset + j]);
            s_x += T[t_offset + k] * gy[x_offset + j];
            s_T += x[x_offset + k] * gy[x_offset + j];
        }
    }

    // y[x_offset + i] = s_y;
    gx[x_offset + i] = s_x;
    gT[x_offset + i] = s_T;
}

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    // dim3 DimGrid((n + N - 1) / N, (n + N - 1) / N, (b * d + B * D - 1) / (B * D));
    // dim3 DimBlock(N, N, B * D);
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    forward_kernel<<<DimGrid, DimBlock>>>(b, d, n, n - 1, 2 * n - 1, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, 0, n, T, x, gy, gT, gx);
}