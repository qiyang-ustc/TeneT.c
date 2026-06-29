#pragma once

// Canonical CUDA helper kernels for TorchExactLRuMPS row-major Float64 native
// operator paths. Torch keeps the tensor wrappers; native numerics live here.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace tenet_native_torch_cuda {

inline cublasOperation_t op_from_char(char op) {
    return op == 'N' || op == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
}

inline cublasStatus_t rowmajor_dgemm_strided_batched(
    cublasHandle_t handle, char transa, char transb, int64_t m, int64_t n,
    int64_t k, const double *alpha, const double *A, int64_t strideA,
    const double *B, int64_t strideB, const double *beta, double *C,
    int64_t strideC, int64_t batch_count) {
    if (m > std::numeric_limits<int>::max() ||
        n > std::numeric_limits<int>::max() ||
        k > std::numeric_limits<int>::max() ||
        batch_count > std::numeric_limits<int>::max()) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const int rows_a_colmajor = (transa == 'N' || transa == 'n') ? k : m;
    const int rows_b_colmajor = (transb == 'N' || transb == 'n') ? n : k;
    return cublasDgemmStridedBatched(
        handle, op_from_char(transb), op_from_char(transa), static_cast<int>(n),
        static_cast<int>(m), static_cast<int>(k), alpha, B, rows_b_colmajor,
        static_cast<long long>(strideB), A, rows_a_colmajor,
        static_cast<long long>(strideA), beta, C, static_cast<int>(n),
        static_cast<long long>(strideC), static_cast<int>(batch_count));
}

static __global__ void copy_scaled_kernel(double *dst, const double *src,
                                          const double *denom, int64_t n) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        const double d = denom[0];
        dst[idx] = fabs(d) <= 1e-300 ? 0.0 : src[idx] / d;
    }
}

static __global__ void repeat_x_over_physical_kernel(double *Xpack,
                                                     const double *X, int64_t d,
                                                     int64_t n, int64_t total) {
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t elem = idx % n;
        const int64_t bt = idx / n;
        const int64_t b = bt / d;
        Xpack[idx] = X[b * n + elem];
    }
}

static __global__ void reduce_physical_kernel(double *Y, const double *terms,
                                              int64_t d, int64_t n,
                                              int64_t total) {
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t elem = idx % n;
        const int64_t b = idx / n;
        double acc = 0.0;
        const int64_t base = (b * d) * n + elem;
        for (int64_t s = 0; s < d; ++s) {
            acc += terms[base + s * n];
        }
        Y[idx] = acc;
    }
}

static __global__ void weighted_transfer_right_kernel(double *V,
                                                      const double *W,
                                                      const double *O,
                                                      int64_t d, int64_t n,
                                                      int64_t total) {
    for (int64_t idx =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t elem = idx % n;
        const int64_t bt = idx / n;
        const int64_t b = bt / d;
        const int64_t t = bt - b * d;
        double acc = 0.0;
        const int64_t w_base = b * d * n + elem;
        for (int64_t s = 0; s < d; ++s) {
            acc += O[s * d + t] * W[w_base + s * n];
        }
        V[idx] = acc;
    }
}

static __global__ void trace_kernel(const double *X, double *tr, int64_t D) {
    extern __shared__ double buf[];
    const int64_t b = blockIdx.x;
    double acc = 0.0;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) {
        acc += X[b * D * D + i * D + i];
    }
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        tr[b] = buf[0];
    }
}

static __global__ void hs_dot_kernel(const double *A, const double *B,
                                     double *out, int64_t n) {
    extern __shared__ double buf[];
    const int64_t b = blockIdx.x;
    double acc = 0.0;
    const int64_t offset = b * n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        acc += A[offset + i] * B[offset + i];
    }
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[b] = buf[0];
    }
}

static __global__ void norm_batch_kernel(const double *X, double *out,
                                         int64_t n) {
    extern __shared__ double buf[];
    const int64_t b = blockIdx.x;
    double acc = 0.0;
    const int64_t offset = b * n;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double x = X[offset + i];
        acc += x * x;
    }
    buf[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            buf[threadIdx.x] += buf[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[b] = sqrt(buf[0]);
    }
}

static __global__ void project_q_apply_kernel(double *Y, const double *R,
                                              const double *X,
                                              const double *tr, int64_t n) {
    const int64_t b = blockIdx.y;
    for (int64_t i =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t linear = b * n + i;
        Y[linear] = X[linear] - tr[b] * R[linear];
    }
}

static __global__ void project_q_adj_apply_kernel(double *Xbar,
                                                  const double *Y,
                                                  const double *alpha,
                                                  int64_t D) {
    const int64_t n = D * D;
    const int64_t b = blockIdx.y;
    for (int64_t i =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = i / D;
        const int64_t col = i - row * D;
        const int64_t linear = b * n + i;
        Xbar[linear] = Y[linear] - (row == col ? alpha[b] : 0.0);
    }
}

static __global__ void add_coeffs_to_h_column_kernel(double *H,
                                                     const double *coeffs,
                                                     int64_t kmax,
                                                     int64_t active,
                                                     int64_t j) {
    const int64_t b = blockIdx.y;
    for (int64_t i =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < active;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        H[b * (kmax + 1) * kmax + i * kmax + j] += coeffs[b * kmax + i];
    }
}

static __global__ void set_h_subdiag_kernel(double *H, const double *values,
                                            int64_t B, int64_t kmax,
                                            int64_t j) {
    const int64_t b =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b < B) {
        H[b * (kmax + 1) * kmax + (j + 1) * kmax + j] = values[b];
    }
}

static __global__ void copy_scaled_to_vpack_col_kernel(double *V,
                                                       const double *src,
                                                       const double *denom,
                                                       int64_t kmax,
                                                       int64_t len,
                                                       int64_t col) {
    const int64_t b = blockIdx.y;
    for (int64_t i =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < len;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const double d = denom[b];
        V[b * kmax * len + col * len + i] =
            fabs(d) <= 1e-300 ? 0.0 : src[b * len + i] / d;
    }
}

}  // namespace tenet_native_torch_cuda
