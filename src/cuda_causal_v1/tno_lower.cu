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
    int b_ = idx / d;
    int d_ = idx % d;
    int t_offset = d_ * n;
    int x_offset = b_ * d * n + d_ * n;

    for (int i = 0; i < n; i++) {
        F s = 0;
        for (int j = 0; j <= i; j++) {
            s += T[t_offset + i - j] * x[x_offset + j];
        }
        y[x_offset + i] = s;
    }
}

template<typename F>
__global__ void upper_kernel(const int b, const int d, const int n, const F* T, const F* x, F* y) {
    /**
    input:
        T: d, n, [t0, t_(-1), ..., t_(-(n-1))]
        x: b, d, n

    output:
        y: b, d, n
    **/
    int b_ = blockIdx.x;
    int d_ = threadIdx.x;
    int t_offset = threadIdx.x * n;
    int x_offset = b_ * d * n + d_ * n;

    for (int i = 0; i < n; i++) {
        F s = 0;
        for (int j = 0; j < n - i; j++) {
            s += T[t_offset + j] * x[x_offset + i + j];
        }
        y[x_offset + i] = s;
    }
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
    int b_ = blockIdx.x;
    int d_ = threadIdx.x;
    int x_offset = b_ * d * n + d_ * n;
    int t_offset = threadIdx.x * n;

    for (int i = 0; i < n; i++) {
        F s_x = 0;
        F s_T = 0;
        for (int j = 0; j < n - i; j++) {
            s_x += T[t_offset + j] * gy[x_offset + i + j];
            s_T += x[x_offset + j] * gy[x_offset + i + j];
        }
        gx[x_offset + i] = s_x;
        gT[x_offset + i] = s_T;
    }
}

void forward_cuda(int b, int d, int n, float* T, float* x, float* y) {
    dim3 DimGrid(b);
    dim3 DimBlock(d);
    printf("%d %d\n", b, d);
    // for (int i = 0; i < b; i++) {
    //     for (int j = 0; j < d; j++) {
    //         for (int k = 0; k < n; k++) {
    //             int index = i * d * n + j * n + k;
    //             printf("%lf ", x[index]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // for (int j = 0; j < d; j++) {
    //     for (int k = 0; k < n; k++) {
    //         int index = j * n + k;
    //         printf("%lf ", T[index]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("%d %lf\n", T, T[0]);
    lower_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, y);
}

void backward_cuda(int b, int d, int n, float* T, float* x, float* gy, float* gT, float* gx) {
    dim3 DimGrid(b);
    dim3 DimBlock(d);
    backward_kernel<<<DimGrid, DimBlock>>>(b, d, n, T, x, gy, gT, gx);
}