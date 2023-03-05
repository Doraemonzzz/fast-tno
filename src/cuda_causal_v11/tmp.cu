#include<stdio.h>
#define B 2
#define N 16
#define D 32

template<typename F>
__global__ void kernel(
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
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n) {
        return;
    }
    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    if (idx >= b * d) {
        return;
    }
    // [-(n-1), (n-1)] -> [0, 2n-2]
    int i_block = blockIdx.y;
    int j_block = blockIdx.x;
    int i_thread = threadIdx.y;
    int j_thread = threadIdx.x;
    // number of block
    int l = (n + N - 1) / N;
    // block level index
    int k_block = i_block - j_block + l - 1;
    // thread level index
    int k_thread = i_thread - j_thread + N - 1;
    // batch global index
    int b_ = idx / d;
    int b_block = b_ / B;
    int b_thread = b_ % B;
    // feature global index
    int d_ = idx % d;
    int d_block = d_ / D;
    int d_thread = d_ % D;


    int start = max(s - i, 0);
    int end = min(e - i, n);
    // int i1 = blockIdx.x;
    // int j1 = blockIdx.y;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    // add bound check
    // __shared__ F T_shared[D][2 * N - 1];
    // __shared__ F x_shared[B][D][N];
    // T_shared[d_thread][k_thread] = T[t_offset + k_block * N + k_thread];
    // x_shared[b_thread][d_thread][k_thread] = x[x_offset + j];
    // __syncthreads();

    F tmp = 0;
    for (int u = 0; u < l; u++) {
        for (int v = 0; v < n; v++) {
            int j_ = u * N + v;
            if (j_ >= n) {
                break;
            }
            int k_block_ = i_block - u + l - 1;
            int k_thread_ = i_thread - v + N - 1;
            tmp += T[t_offset + k_block_ * N + k_thread_] * x[x_offset + j_];
        }
    }

    y[x_offset + i] = tmp;
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
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    lower_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid((n + N - 1) / N, (d + D - 1) / D, (b + B - 1) / B);
    dim3 DimBlock(N, D, B);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}