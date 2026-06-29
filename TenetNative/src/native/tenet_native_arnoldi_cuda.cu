#include "tenet_native_arnoldi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

extern "C" {
void dgemm_64_(char *transa, char *transb, int64_t *m, int64_t *n,
               int64_t *k, double *alpha, const double *a, int64_t *lda,
               const double *b, int64_t *ldb, double *beta, double *c,
               int64_t *ldc);
void dgeqrf_64_(int64_t *m, int64_t *n, double *a, int64_t *lda, double *tau,
                double *work, int64_t *lwork, int64_t *info);
void dorgqr_64_(int64_t *m, int64_t *n, int64_t *k, double *a, int64_t *lda,
                double *tau, double *work, int64_t *lwork, int64_t *info);
void dgeevx_64_(char *balanc, char *jobvl, char *jobvr, char *sense,
                int64_t *n, double *a, int64_t *lda, double *wr, double *wi,
                double *vl, int64_t *ldvl, double *vr, int64_t *ldvr,
                int64_t *ilo, int64_t *ihi, double *scale, double *abnrm,
                double *rconde, double *rcondv, double *work, int64_t *lwork,
                int64_t *iwork, int64_t *info);
typedef int (*dgees_select_64_fn)(double *, double *);
void dgees_64_(char *jobvs, char *sort, dgees_select_64_fn select, int64_t *n,
               double *a, int64_t *lda, int64_t *sdim, double *wr,
               double *wi, double *vs, int64_t *ldvs, double *work,
               int64_t *lwork, int64_t *bwork, int64_t *info);
}

namespace {

thread_local char last_error[256] = "success";
thread_local double last_dominant_relres = 0.0;

void set_error(const char *msg) {
    std::snprintf(last_error, sizeof(last_error), "%s", msg);
}

bool check_last_dominant_residual(const char *context, double residual_tol) {
    if (std::isfinite(last_dominant_relres) &&
        last_dominant_relres <= residual_tol) {
        return true;
    }
    std::snprintf(last_error, sizeof(last_error),
                  "native CUDA Arnoldi %s residual %.6e exceeds tolerance %.6e",
                  context, last_dominant_relres, residual_tol);
    return false;
}

struct DeviceBuffer {
    double *ptr = nullptr;
    std::size_t bytes = 0;
};

bool debug_enabled() {
#if defined(TENET_NATIVE_ENABLE_DEBUG)
    const char *env = std::getenv("TENET_NATIVE_DEBUG");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
#else
    return false;
#endif
}

using SteadyClock = std::chrono::steady_clock;

bool profile_enabled() {
    const char *env = std::getenv("TENET_NATIVE_PROFILE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

double seconds_since(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

thread_local std::vector<double> host_schur_select_wr;
thread_local std::vector<double> host_schur_select_wi;

enum class RitzTarget { LargestMagnitude };

int host_schur_select_callback(double *wr, double *wi) {
    const double ar = wr == nullptr ? 0.0 : *wr;
    const double ai = wi == nullptr ? 0.0 : *wi;
    for (std::size_t i = 0; i < host_schur_select_wr.size(); ++i) {
        const double sr = host_schur_select_wr[i];
        const double si = host_schur_select_wi[i];
        const double scale =
            std::max(1.0, std::max(std::hypot(ar, ai), std::hypot(sr, si)));
        if (std::abs(ar - sr) <= 1e-7 * scale &&
            std::abs(ai - si) <= 1e-7 * scale) {
            return 1;
        }
    }
    return 0;
}

void profile_cuda_arnoldi(int64_t len, int64_t max_k, int64_t m,
                          int64_t applied_cols,
                          double apply_seconds, double orthog_seconds,
                          double norm_seconds, double final_resnorm) {
    if (!profile_enabled()) {
        return;
    }
    std::fprintf(stderr,
                 "TENET_NATIVE_PROFILE kind=cuda_arnoldi len=%lld max_k=%lld "
                 "m=%lld applied=%lld apply_seconds=%.9g orthog_seconds=%.9g "
                 "norm_seconds=%.9g final_resnorm=%.9g\n",
                 static_cast<long long>(len), static_cast<long long>(max_k),
                 static_cast<long long>(m), static_cast<long long>(applied_cols),
                 apply_seconds, orthog_seconds, norm_seconds, final_resnorm);
    std::fflush(stderr);
}

int cuda_status(cudaError_t err, const char *op) {
    if (err == cudaSuccess) {
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    }
    std::snprintf(last_error, sizeof(last_error), "%s: %s", op,
                  cudaGetErrorString(err));
    return TENET_NATIVE_BACKEND_ERROR;
}

int ensure_device_buffer(DeviceBuffer &buffer, std::size_t bytes,
                         const char *op) {
    if (buffer.bytes >= bytes && buffer.ptr != nullptr) {
        return TENET_NATIVE_SUCCESS;
    }
    if (buffer.ptr != nullptr) {
        cudaFree(buffer.ptr);
        buffer.ptr = nullptr;
        buffer.bytes = 0;
    }
    const int status =
        cuda_status(cudaMalloc(reinterpret_cast<void **>(&buffer.ptr), bytes),
                    op);
    if (status == TENET_NATIVE_SUCCESS) {
        buffer.bytes = bytes;
    }
    return status;
}

int cublas_status(cublasStatus_t err, const char *op) {
    if (err == CUBLAS_STATUS_SUCCESS) {
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    }
    std::snprintf(last_error, sizeof(last_error), "%s: cublas status %d", op,
                  static_cast<int>(err));
    return TENET_NATIVE_BACKEND_ERROR;
}

#ifndef CUBLAS_PEDANTIC_MATH
static const cublasMath_t TENET_NATIVE_DEFAULT_CUBLAS_MATH_MODE =
    CUBLAS_DEFAULT_MATH;
#else
static const cublasMath_t TENET_NATIVE_DEFAULT_CUBLAS_MATH_MODE =
    CUBLAS_PEDANTIC_MATH;
#endif

int resolve_cublas_math_mode(cublasMath_t *mode) {
    if (mode == nullptr) {
        set_error("invalid cublas math mode output pointer");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const char *mode_env = std::getenv("TENET_NATIVE_CUBLAS_MATH_MODE");
    if (mode_env == nullptr || mode_env[0] == '\0') {
        *mode = TENET_NATIVE_DEFAULT_CUBLAS_MATH_MODE;
        return TENET_NATIVE_SUCCESS;
    }
    if (std::strcmp(mode_env, "default") == 0 || std::strcmp(mode_env, "0") == 0) {
        *mode = CUBLAS_DEFAULT_MATH;
        return TENET_NATIVE_SUCCESS;
    }
    if (std::strcmp(mode_env, "pedantic") == 0) {
        *mode = TENET_NATIVE_DEFAULT_CUBLAS_MATH_MODE;
        return TENET_NATIVE_SUCCESS;
    }
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
    if (std::strcmp(mode_env, "tf32") == 0) {
        *mode = CUBLAS_TF32_TENSOR_OP_MATH;
        return TENET_NATIVE_SUCCESS;
    }
#endif
    set_error("invalid TENET_NATIVE_CUBLAS_MATH_MODE; expected default, pedantic, or tf32");
    return TENET_NATIVE_INVALID_VALUE;
}

int create_cublas_handle(cublasHandle_t *blas, const char *op) {
    int status = cublas_status(cublasCreate(blas), op);
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    cublasMath_t math_mode;
    status = resolve_cublas_math_mode(&math_mode);
    if (status != TENET_NATIVE_SUCCESS) {
        cublasDestroy(*blas);
        *blas = nullptr;
        return status;
    }
    status = cublas_status(cublasSetMathMode(*blas, math_mode),
                           "cublasSetMathMode");
    if (status != TENET_NATIVE_SUCCESS) {
        cublasDestroy(*blas);
        *blas = nullptr;
        return status;
    }
    status = cublas_status(cublasSetPointerMode(*blas, CUBLAS_POINTER_MODE_HOST),
                           "cublasSetPointerMode host");
    if (status != TENET_NATIVE_SUCCESS) {
        cublasDestroy(*blas);
        *blas = nullptr;
        return status;
    }
    return TENET_NATIVE_SUCCESS;
}

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        int _st = cuda_status((expr), #expr);                                   \
        if (_st != TENET_NATIVE_SUCCESS)                                        \
            return _st;                                                         \
    } while (0)

#define CUBLAS_CHECK(expr)                                                      \
    do {                                                                        \
        int _st = cublas_status((expr), #expr);                                 \
        if (_st != TENET_NATIVE_SUCCESS)                                        \
            return _st;                                                         \
    } while (0)

__host__ __device__ inline long long idx2(long long i, long long j, long long n1) {
    return i + n1 * j;
}

__host__ __device__ inline long long idx3(long long i, long long j, long long k,
                                 long long n1, long long n2) {
    return i + n1 * (j + n2 * k);
}

__host__ __device__ inline long long idx4(long long i, long long j, long long k,
                                 long long l, long long n1, long long n2,
                                 long long n3) {
    return i + n1 * (j + n2 * (k + n3 * l));
}

__global__ void two_layer_apply_kernel(long long chi, long long phys,
                                       const double *Aup, const double *Adn,
                                       const double *x, double *y,
                                       int transpose) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    const long long len = chi * chi;
    if (tid >= len) {
        return;
    }
    if (transpose == 0) {
        const long long c = tid % chi;
        const long long e = tid / chi;
        double acc = 0.0;
        for (long long a = 0; a < chi; ++a) {
            for (long long d = 0; d < chi; ++d) {
                const double xad = x[idx2(a, d, chi)];
                for (long long b = 0; b < phys; ++b) {
                    acc += xad * Adn[idx3(d, b, e, chi, phys)] *
                           Aup[idx3(a, b, c, chi, phys)];
                }
            }
        }
        y[tid] = acc;
    } else {
        const long long a = tid % chi;
        const long long d = tid / chi;
        double acc = 0.0;
        for (long long c = 0; c < chi; ++c) {
            for (long long e = 0; e < chi; ++e) {
                const double xce = x[idx2(c, e, chi)];
                for (long long b = 0; b < phys; ++b) {
                    acc += Aup[idx3(a, b, c, chi, phys)] * xce *
                           Adn[idx3(d, b, e, chi, phys)];
                }
            }
        }
        y[tid] = acc;
    }
}

__global__ void three_layer_apply_kernel(long long chi, long long phys,
                                         const double *Aup, const double *Adn,
                                         const double *M, const double *x,
                                         double *y, int transpose) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    const long long len = chi * phys * chi;
    if (tid >= len) {
        return;
    }
    if (transpose == 0) {
        const long long c = tid % chi;
        const long long t = tid / chi;
        const long long e = t % phys;
        const long long h = t / phys;
        double acc = 0.0;
        for (long long a = 0; a < chi; ++a) {
            for (long long d = 0; d < phys; ++d) {
                for (long long f = 0; f < chi; ++f) {
                    const double xadf = x[idx3(a, d, f, chi, phys)];
                    for (long long g = 0; g < phys; ++g) {
                        const double lower = Adn[idx3(f, g, h, chi, phys)];
                        for (long long b = 0; b < phys; ++b) {
                            acc += xadf * lower *
                                   M[idx4(d, g, e, b, phys, phys, phys)] *
                                   Aup[idx3(a, b, c, chi, phys)];
                        }
                    }
                }
            }
        }
        y[tid] = acc;
    } else {
        const long long a = tid % chi;
        const long long t = tid / chi;
        const long long d = t % phys;
        const long long f = t / phys;
        double acc = 0.0;
        for (long long c = 0; c < chi; ++c) {
            for (long long e = 0; e < phys; ++e) {
                for (long long h = 0; h < chi; ++h) {
                    const double xceh = x[idx3(c, e, h, chi, phys)];
                    for (long long g = 0; g < phys; ++g) {
                        const double lower = Adn[idx3(f, g, h, chi, phys)];
                        for (long long b = 0; b < phys; ++b) {
                            acc += Aup[idx3(a, b, c, chi, phys)] * xceh *
                                   M[idx4(d, g, e, b, phys, phys, phys)] *
                                   lower;
                        }
                    }
                }
            }
        }
        y[tid] = acc;
    }
}

__global__ void add_diagonal_kernel(long long chi, double alpha, double *y) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (i < chi) {
        y[idx2(i, i, chi)] += alpha;
    }
}

__global__ void pack_leg3_slice_kernel(long long chi, long long phys,
                                       const double *src, long long s,
                                       double *dst) {
    const long long len = chi * chi;
    const long long linear =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= len) {
        return;
    }
    const long long i = linear % chi;
    const long long k = linear / chi;
    dst[linear] = src[idx3(i, s, k, chi, phys)];
}

__global__ void trace_reduce_kernel(long long chi, const double *x,
                                    double *partial) {
    extern __shared__ double scratch[];
    double acc = 0.0;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         i < chi; i += stride) {
        acc += x[idx2(i, i, chi)];
    }
    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = scratch[0];
    }
}

__global__ void dot_reduce_kernel(long long n, const double *x, const double *y,
                                  double *partial) {
    extern __shared__ double scratch[];
    double acc = 0.0;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         i < n; i += stride) {
        acc += x[i] * y[i];
    }
    scratch[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = scratch[0];
    }
}

int device_dot(long long n, const double *x, const double *y, double *result) {
    if (n <= 0 || x == nullptr || y == nullptr || result == nullptr) {
        set_error("invalid device dot inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const int threads = 256;
    const int blocks = static_cast<int>(
        std::min<long long>((n + threads - 1) / threads, 4096));
    double *partial_dev = nullptr;
    int status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&partial_dev),
                   static_cast<std::size_t>(blocks) * sizeof(double)),
        "allocate dot reduction workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    dot_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(n, x, y,
                                                                     partial_dev);
    status = cuda_status(cudaGetLastError(), "launch dot reduction");
    if (status != TENET_NATIVE_SUCCESS) {
        cudaFree(partial_dev);
        return status;
    }
    std::vector<double> partial(static_cast<std::size_t>(blocks));
    status = cuda_status(
        cudaMemcpy(partial.data(), partial_dev,
                   static_cast<std::size_t>(blocks) * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "copy dot reduction");
    cudaFree(partial_dev);
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    double acc = 0.0;
    for (double value : partial) {
        acc += value;
    }
    *result = acc;
    return TENET_NATIVE_SUCCESS;
}

int device_trace(long long chi, const double *x, double *result) {
    if (chi <= 0 || x == nullptr || result == nullptr) {
        set_error("invalid device trace inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const int threads = 256;
    const int blocks = static_cast<int>(
        std::min<long long>((chi + threads - 1) / threads, 4096));
    double *partial_dev = nullptr;
    int status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&partial_dev),
                   static_cast<std::size_t>(blocks) * sizeof(double)),
        "allocate trace reduction workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    trace_reduce_kernel<<<blocks, threads, threads * sizeof(double)>>>(
        chi, x, partial_dev);
    status = cuda_status(cudaGetLastError(), "launch trace reduction");
    if (status != TENET_NATIVE_SUCCESS) {
        cudaFree(partial_dev);
        return status;
    }
    std::vector<double> partial(static_cast<std::size_t>(blocks));
    status = cuda_status(
        cudaMemcpy(partial.data(), partial_dev,
                   static_cast<std::size_t>(blocks) * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "copy trace reduction");
    cudaFree(partial_dev);
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    double acc = 0.0;
    for (double value : partial) {
        acc += value;
    }
    *result = acc;
    return TENET_NATIVE_SUCCESS;
}

int device_norm2(long long n, const double *x, double *result) {
    double sumsq = 0.0;
    const int status = device_dot(n, x, x, &sumsq);
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    *result = std::sqrt(std::max(sumsq, 0.0));
    return TENET_NATIVE_SUCCESS;
}

bool check_common(int64_t len, int64_t max_k, double breakdown_tol,
                  const double *x0, double *V, int64_t ldv, double *H,
                  int64_t ldh, double *beta, int64_t *m,
                  double *final_resnorm) {
    if (len <= 0 || max_k < 0 || max_k > len || breakdown_tol < 0.0) {
        set_error("invalid Arnoldi dimensions or tolerance");
        return false;
    }
    if (len > INT_MAX) {
        set_error("Arnoldi dimension exceeds cuBLAS int range");
        return false;
    }
    if (x0 == nullptr || V == nullptr || H == nullptr || beta == nullptr ||
        m == nullptr || final_resnorm == nullptr) {
        set_error("null Arnoldi pointer argument");
        return false;
    }
    if (ldv < len || ldh < max_k + 1) {
        set_error("invalid leading dimension");
        return false;
    }
    set_error("success");
    return true;
}

bool check_tensors(long long chi, long long phys, const double *Aup,
                   const double *Adn) {
    if (chi <= 0 || phys <= 0) {
        set_error("chi and phys must be positive");
        return false;
    }
    if (Aup == nullptr || Adn == nullptr) {
        set_error("null tensor pointer argument");
        return false;
    }
    return true;
}

int dominant_restart_blocks(int64_t len, int64_t max_k) {
    if (max_k >= len) {
        return 1;
    }
    int blocks = 100;
    if (const char *env = std::getenv("TENET_NATIVE_ARNOLDI_RESTARTS")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            blocks = std::min(parsed, 1024);
        }
    }
    return blocks;
}

double dominant_residual_tol(double breakdown_tol) { return breakdown_tol; }

int64_t dominant_convergence_nvalues() {
    int64_t nvalues = 1;
    if (const char *env = std::getenv("TENET_NATIVE_ARNOLDI_NVALUES")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            nvalues = parsed;
        }
    }
    return std::max<int64_t>(1, std::min<int64_t>(nvalues, 16));
}

int64_t dominant_thick_keep_count(int64_t max_k) {
    if (max_k < 4) {
        return 1;
    }
    int64_t keep = (3 * max_k) / 5;
    if (const char *env = std::getenv("TENET_NATIVE_ARNOLDI_THICK_KEEP")) {
        if (env[0] == 'a' || env[0] == 'A' || env[0] == 'k' || env[0] == 'K') {
            keep = (3 * max_k) / 5;
        } else {
            const int parsed = std::atoi(env);
            if (parsed > 0) {
                keep = parsed;
            }
        }
    }
    keep = std::max<int64_t>(1, keep);
    keep = std::min<int64_t>(keep, std::max<int64_t>(1, max_k - 1));
    return keep;
}

int two_layer_apply_gemm(cublasHandle_t blas, long long chi, long long phys,
                         const double *Aup, const double *Adn, const double *x,
                         double *tmp, double *y, int transpose) {
    static thread_local DeviceBuffer A_slice_buffer;
    static thread_local DeviceBuffer B_slice_buffer;
    const long long len = chi * chi;
    const std::size_t len_bytes = static_cast<std::size_t>(len) * sizeof(double);
    int st = ensure_device_buffer(A_slice_buffer, len_bytes,
                                  "allocate two_layer A slice");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    st = ensure_device_buffer(B_slice_buffer, len_bytes,
                              "allocate two_layer B slice");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    double *A_slice = A_slice_buffer.ptr;
    double *B_slice = B_slice_buffer.ptr;
    st = cuda_status(cudaMemset(y, 0, len_bytes), "cudaMemset(two_layer y)");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const double one = 1.0;
    const double zero = 0.0;
    for (long long b = 0; b < phys; ++b) {
        const int threads = 256;
        const int blocks = static_cast<int>((len + threads - 1) / threads);
        pack_leg3_slice_kernel<<<blocks, threads>>>(chi, phys, Aup, b,
                                                    A_slice);
        CUDA_CHECK(cudaGetLastError());
        pack_leg3_slice_kernel<<<blocks, threads>>>(chi, phys, Adn, b,
                                                    B_slice);
        CUDA_CHECK(cudaGetLastError());
        if (transpose == 0) {
            st = cublas_status(
                cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                            &one, x, static_cast<int>(chi), B_slice, static_cast<int>(chi),
                            &zero, tmp, static_cast<int>(chi)),
                "two_layer temp = x * B");
            if (st != TENET_NATIVE_SUCCESS) return st;
            st = cublas_status(
                cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                            &one, A_slice, static_cast<int>(chi), tmp, static_cast<int>(chi),
                            &one, y, static_cast<int>(chi)),
                "two_layer y += A' * temp");
            if (st != TENET_NATIVE_SUCCESS) return st;
        } else {
            st = cublas_status(
                cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                            static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                            &one, x, static_cast<int>(chi), B_slice, static_cast<int>(chi),
                            &zero, tmp, static_cast<int>(chi)),
                "two_layer transpose temp = x * B'");
            if (st != TENET_NATIVE_SUCCESS) return st;
            st = cublas_status(
                cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                            static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                            &one, A_slice, static_cast<int>(chi), tmp, static_cast<int>(chi),
                            &one, y, static_cast<int>(chi)),
                "two_layer transpose y += A * temp");
            if (st != TENET_NATIVE_SUCCESS) return st;
        }
    }
    return TENET_NATIVE_SUCCESS;
}

int projected_two_layer_apply_gemm(cublasHandle_t blas, long long chi,
                                   long long phys, const double *Aup,
                                   const double *Adn, const double *rho,
                                   const double *x, double *tmp, double *y,
                                   int transpose) {
    int st = two_layer_apply_gemm(blas, chi, phys, Aup, Adn, x, tmp, y,
                                  transpose);
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const int len = static_cast<int>(chi * chi);
    const double minus_one = -1.0;
    const double one = 1.0;
    CUBLAS_CHECK(cublasDscal(blas, len, &minus_one, y, 1));
    CUBLAS_CHECK(cublasDaxpy(blas, len, &one, x, 1, y, 1));
    double projection = 0.0;
    {
        const int st = device_dot(len, x, rho, &projection);
        if (st != TENET_NATIVE_SUCCESS) return st;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((chi + threads - 1) / threads);
    add_diagonal_kernel<<<blocks, threads>>>(chi, projection, y);
    CUDA_CHECK(cudaGetLastError());
    return TENET_NATIVE_SUCCESS;
}

int qprojected_two_layer_apply_gemm(cublasHandle_t blas, long long chi,
                                    long long phys, const double *Aup,
                                    const double *Adn, const double *rho,
                                    const double *x, double *gemm_tmp,
                                    double *qwork, double *y,
                                    int transpose) {
    const int len = static_cast<int>(chi * chi);
    if (transpose == 0) {
        CUBLAS_CHECK(cublasDcopy(blas, len, x, 1, qwork, 1));
        double dot_rx = 0.0;
        int st = device_dot(chi * chi, rho, x, &dot_rx);
        if (st != TENET_NATIVE_SUCCESS) return st;
        const int threads = 256;
        const int blocks = static_cast<int>((chi + threads - 1) / threads);
        add_diagonal_kernel<<<blocks, threads>>>(chi, -dot_rx, qwork);
        CUDA_CHECK(cudaGetLastError());

        st = two_layer_apply_gemm(blas, chi, phys, Aup, Adn, qwork, gemm_tmp,
                                  y, 0);
        if (st != TENET_NATIVE_SUCCESS) return st;

        double dot_ry = 0.0;
        st = device_dot(chi * chi, rho, y, &dot_ry);
        if (st != TENET_NATIVE_SUCCESS) return st;
        add_diagonal_kernel<<<blocks, threads>>>(chi, -dot_ry, y);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUBLAS_CHECK(cublasDcopy(blas, len, x, 1, qwork, 1));
        double trace_x = 0.0;
        int st = device_trace(chi, x, &trace_x);
        if (st != TENET_NATIVE_SUCCESS) return st;
        const double minus_trace_x = -trace_x;
        CUBLAS_CHECK(cublasDaxpy(blas, len, &minus_trace_x, rho, 1, qwork, 1));

        st = two_layer_apply_gemm(blas, chi, phys, Aup, Adn, qwork, gemm_tmp,
                                  y, 1);
        if (st != TENET_NATIVE_SUCCESS) return st;

        double trace_y = 0.0;
        st = device_trace(chi, y, &trace_y);
        if (st != TENET_NATIVE_SUCCESS) return st;
        const double minus_trace_y = -trace_y;
        CUBLAS_CHECK(cublasDaxpy(blas, len, &minus_trace_y, rho, 1, y, 1));
    }
    return TENET_NATIVE_SUCCESS;
}

const double *batch_const_ptr(const double *base, long long stride,
                              long long batch_index) {
    return base + (stride == 0 ? 0 : batch_index * stride);
}

double *batch_ptr(double *base, long long stride, long long batch_index) {
    return base + batch_index * stride;
}

bool check_batch_stride(const char *name, long long stride, long long min_stride,
                        bool allow_zero) {
    if (stride == 0 && allow_zero) {
        return true;
    }
    if (stride < min_stride) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid %s batch stride", name);
        return false;
    }
    return true;
}

bool check_two_layer_batch_common(long long batch, long long chi,
                                  long long phys, const double *Aup,
                                  long long stride_Aup, const double *Adn,
                                  long long stride_Adn, const double *X,
                                  long long stride_X, double *Y,
                                  long long stride_Y) {
    if (!check_tensors(chi, phys, Aup, Adn)) {
        return false;
    }
    if (batch <= 0 || batch > INT_MAX) {
        set_error("invalid batch count");
        return false;
    }
    const long long len = chi * chi;
    const long long tensor_len = chi * phys * chi;
    if (len <= 0 || len > INT_MAX || tensor_len <= 0) {
        set_error("invalid batched two-layer dimensions");
        return false;
    }
    if (X == nullptr || Y == nullptr) {
        set_error("null batched two-layer X/Y pointer");
        return false;
    }
    return check_batch_stride("Aup", stride_Aup, tensor_len, true) &&
           check_batch_stride("Adn", stride_Adn, tensor_len, true) &&
           check_batch_stride("X", stride_X, len, false) &&
           check_batch_stride("Y", stride_Y, len, false);
}

bool check_projected_batch_common(long long batch, long long chi,
                                  long long phys, const double *Aup,
                                  long long stride_Aup, const double *Adn,
                                  long long stride_Adn, const double *rho,
                                  long long stride_rho, const double *X,
                                  long long stride_X, double *Y,
                                  long long stride_Y) {
    if (!check_two_layer_batch_common(batch, chi, phys, Aup, stride_Aup, Adn,
                                      stride_Adn, X, stride_X, Y, stride_Y)) {
        return false;
    }
    if (rho == nullptr) {
        set_error("null rho pointer");
        return false;
    }
    return check_batch_stride("rho", stride_rho, chi * chi, true);
}

int zero_strided_batch(double *Y, long long stride_Y, long long len,
                       long long batch) {
    if (stride_Y == len) {
        CUDA_CHECK(cudaMemset(Y, 0,
                              static_cast<std::size_t>(len * batch) *
                                  sizeof(double)));
        return TENET_NATIVE_SUCCESS;
    }
    for (long long b = 0; b < batch; ++b) {
        CUDA_CHECK(cudaMemset(batch_ptr(Y, stride_Y, b), 0,
                              static_cast<std::size_t>(len) *
                                  sizeof(double)));
    }
    return TENET_NATIVE_SUCCESS;
}

int two_layer_apply_batch_gemm(cublasHandle_t blas, long long batch,
                               long long chi, long long phys,
                               const double *Aup, long long stride_Aup,
                               const double *Adn, long long stride_Adn,
                               const double *X, long long stride_X,
                               double *tmp, double *Y, long long stride_Y,
                               int transpose) {
    const long long len = chi * chi;
    for (long long b = 0; b < batch; ++b) {
        const double *Ab = batch_const_ptr(Aup, stride_Aup, b);
        const double *Bb = batch_const_ptr(Adn, stride_Adn, b);
        const double *Xb = batch_const_ptr(X, stride_X, b);
        double *Yb = batch_ptr(Y, stride_Y, b);
        double *tmpb = batch_ptr(tmp, len, b);
        const int st = two_layer_apply_gemm(blas, chi, phys, Ab, Bb, Xb, tmpb,
                                            Yb, transpose);
        if (st != TENET_NATIVE_SUCCESS) {
            return st;
        }
    }
    return TENET_NATIVE_SUCCESS;
}

int projected_two_layer_apply_batch_gemm(cublasHandle_t blas, long long batch,
                                         long long chi, long long phys,
                                         const double *Aup,
                                         long long stride_Aup,
                                         const double *Adn,
                                         long long stride_Adn,
                                         const double *rho,
                                         long long stride_rho,
                                         const double *X,
                                         long long stride_X, double *tmp,
                                         double *Y, long long stride_Y,
                                         int transpose) {
    const long long len = chi * chi;
    int st = two_layer_apply_batch_gemm(blas, batch, chi, phys, Aup,
                                        stride_Aup, Adn, stride_Adn, X,
                                        stride_X, tmp, Y, stride_Y,
                                        transpose);
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const double minus_one = -1.0;
    const double one = 1.0;
    for (long long b = 0; b < batch; ++b) {
        const double *xb = batch_const_ptr(X, stride_X, b);
        const double *rhob = batch_const_ptr(rho, stride_rho, b);
        double *yb = batch_ptr(Y, stride_Y, b);
        CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &minus_one, yb,
                                 1));
        CUBLAS_CHECK(cublasDaxpy(blas, static_cast<int>(len), &one, xb, 1, yb,
                                 1));
        double projection = 0.0;
        st = device_dot(len, xb, rhob, &projection);
        if (st != TENET_NATIVE_SUCCESS) {
            return st;
        }
        const int threads = 256;
        const int blocks = static_cast<int>((chi + threads - 1) / threads);
        add_diagonal_kernel<<<blocks, threads>>>(chi, projection, yb);
        CUDA_CHECK(cudaGetLastError());
    }
    return TENET_NATIVE_SUCCESS;
}

int copy_strided_batch(cublasHandle_t blas, long long batch, long long len,
                       const double *X, long long stride_X, double *Y,
                       long long stride_Y) {
    for (long long b = 0; b < batch; ++b) {
        CUBLAS_CHECK(cublasDcopy(blas, static_cast<int>(len),
                                 batch_const_ptr(X, stride_X, b), 1,
                                 batch_ptr(Y, stride_Y, b), 1));
    }
    return TENET_NATIVE_SUCCESS;
}

int qprojected_two_layer_apply_batch_gemm(cublasHandle_t blas, long long batch,
                                          long long chi, long long phys,
                                          const double *Aup,
                                          long long stride_Aup,
                                          const double *Adn,
                                          long long stride_Adn,
                                          const double *rho,
                                          long long stride_rho,
                                          const double *X,
                                          long long stride_X, double *tmp,
                                          double *qwork, double *Y,
                                          long long stride_Y, int transpose) {
    const long long len = chi * chi;
    int st = copy_strided_batch(blas, batch, len, X, stride_X, qwork, len);
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const int threads = 256;
    const int blocks = static_cast<int>((chi + threads - 1) / threads);
    if (transpose == 0) {
        for (long long b = 0; b < batch; ++b) {
            const double *xb = batch_const_ptr(X, stride_X, b);
            const double *rhob = batch_const_ptr(rho, stride_rho, b);
            double *qwb = batch_ptr(qwork, len, b);
            double dot_rx = 0.0;
            st = device_dot(len, rhob, xb, &dot_rx);
            if (st != TENET_NATIVE_SUCCESS) {
                return st;
            }
            add_diagonal_kernel<<<blocks, threads>>>(chi, -dot_rx, qwb);
            CUDA_CHECK(cudaGetLastError());
        }
        st = two_layer_apply_batch_gemm(blas, batch, chi, phys, Aup,
                                        stride_Aup, Adn, stride_Adn, qwork,
                                        len, tmp, Y, stride_Y, 0);
        if (st != TENET_NATIVE_SUCCESS) {
            return st;
        }
        for (long long b = 0; b < batch; ++b) {
            const double *rhob = batch_const_ptr(rho, stride_rho, b);
            double *yb = batch_ptr(Y, stride_Y, b);
            double dot_ry = 0.0;
            st = device_dot(len, rhob, yb, &dot_ry);
            if (st != TENET_NATIVE_SUCCESS) {
                return st;
            }
            add_diagonal_kernel<<<blocks, threads>>>(chi, -dot_ry, yb);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        for (long long b = 0; b < batch; ++b) {
            const double *xb = batch_const_ptr(X, stride_X, b);
            const double *rhob = batch_const_ptr(rho, stride_rho, b);
            double *qwb = batch_ptr(qwork, len, b);
            double trace_x = 0.0;
            st = device_trace(chi, xb, &trace_x);
            if (st != TENET_NATIVE_SUCCESS) {
                return st;
            }
            const double minus_trace_x = -trace_x;
            CUBLAS_CHECK(cublasDaxpy(blas, static_cast<int>(len),
                                     &minus_trace_x, rhob, 1, qwb, 1));
        }
        st = two_layer_apply_batch_gemm(blas, batch, chi, phys, Aup,
                                        stride_Aup, Adn, stride_Adn, qwork,
                                        len, tmp, Y, stride_Y, 1);
        if (st != TENET_NATIVE_SUCCESS) {
            return st;
        }
        for (long long b = 0; b < batch; ++b) {
            const double *rhob = batch_const_ptr(rho, stride_rho, b);
            double *yb = batch_ptr(Y, stride_Y, b);
            double trace_y = 0.0;
            st = device_trace(chi, yb, &trace_y);
            if (st != TENET_NATIVE_SUCCESS) {
                return st;
            }
            const double minus_trace_y = -trace_y;
            CUBLAS_CHECK(cublasDaxpy(blas, static_cast<int>(len),
                                     &minus_trace_y, rhob, 1, yb, 1));
        }
    }
    return TENET_NATIVE_SUCCESS;
}

int three_layer_apply_gemm(cublasHandle_t blas, long long chi, long long phys,
                           const double *Aup, const double *Adn, const double *M,
                           const double *x, double *tmp, double *y, int transpose) {
    const long long len = chi * phys * chi;
    int st = cuda_status(cudaMemset(y, 0, static_cast<std::size_t>(len) * sizeof(double)),
                         "cudaMemset(three_layer y)");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const double one = 1.0;
    const double zero = 0.0;
    const long long tensor_ld = chi * phys;
    if (transpose == 0) {
        for (long long d = 0; d < phys; ++d) {
            const double *Xd = x + chi * d;
            for (long long g = 0; g < phys; ++g) {
                bool has_alpha = false;
                for (long long e = 0; e < phys && !has_alpha; ++e) {
                    for (long long b = 0; b < phys; ++b) {
                        has_alpha = M[idx4(d, g, e, b, phys, phys, phys)] != 0.0;
                        if (has_alpha) break;
                    }
                }
                if (!has_alpha) continue;
                const double *Bg = Adn + chi * g;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                                &one, Xd, static_cast<int>(tensor_ld), Bg, static_cast<int>(tensor_ld),
                                &zero, tmp, static_cast<int>(chi)),
                    "three_layer temp = Xd * Bg");
                if (st != TENET_NATIVE_SUCCESS) return st;
                for (long long e = 0; e < phys; ++e) {
                    double *Ye = y + chi * e;
                    for (long long b = 0; b < phys; ++b) {
                        const double alpha = M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha == 0.0) continue;
                        const double *Ab = Aup + chi * b;
                        st = cublas_status(
                            cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                        static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                                        &alpha, Ab, static_cast<int>(tensor_ld), tmp, static_cast<int>(chi),
                                        &one, Ye, static_cast<int>(tensor_ld)),
                            "three_layer Ye += A' * temp");
                        if (st != TENET_NATIVE_SUCCESS) return st;
                    }
                }
            }
        }
    } else {
        for (long long e = 0; e < phys; ++e) {
            const double *Xe = x + chi * e;
            for (long long g = 0; g < phys; ++g) {
                bool has_alpha = false;
                for (long long d = 0; d < phys && !has_alpha; ++d) {
                    for (long long b = 0; b < phys; ++b) {
                        has_alpha = M[idx4(d, g, e, b, phys, phys, phys)] != 0.0;
                        if (has_alpha) break;
                    }
                }
                if (!has_alpha) continue;
                const double *Bg = Adn + chi * g;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                                static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                                &one, Xe, static_cast<int>(tensor_ld), Bg, static_cast<int>(tensor_ld),
                                &zero, tmp, static_cast<int>(chi)),
                    "three_layer transpose temp = Xe * Bg'");
                if (st != TENET_NATIVE_SUCCESS) return st;
                for (long long d = 0; d < phys; ++d) {
                    double *Yd = y + chi * d;
                    for (long long b = 0; b < phys; ++b) {
                        const double alpha = M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha == 0.0) continue;
                        const double *Ab = Aup + chi * b;
                        st = cublas_status(
                            cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                        static_cast<int>(chi), static_cast<int>(chi), static_cast<int>(chi),
                                        &alpha, Ab, static_cast<int>(tensor_ld), tmp, static_cast<int>(chi),
                                        &one, Yd, static_cast<int>(tensor_ld)),
                            "three_layer transpose Yd += A * temp");
                        if (st != TENET_NATIVE_SUCCESS) return st;
                    }
                }
            }
        }
    }
    return TENET_NATIVE_SUCCESS;
}

int three_layer_apply_gemm_factored(cublasHandle_t blas, long long chi,
                                    long long phys, const double *Aup,
                                    const double *Adn, const double *M,
                                    const double *x, double *pair_work,
                                    double *accum_work, double *y,
                                    int transpose) {
    const long long len = chi * phys * chi;
    const long long len2 = chi * chi;
    const long long nblocks = phys * phys;
    int st = cuda_status(
        cudaMemset(y, 0, static_cast<std::size_t>(len) * sizeof(double)),
        "cudaMemset(three_layer factored y)");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    st = cuda_status(cudaMemset(accum_work, 0,
                                static_cast<std::size_t>(nblocks * len2) *
                                    sizeof(double)),
                     "cudaMemset(three_layer factored accum)");
    if (st != TENET_NATIVE_SUCCESS) {
        return st;
    }
    const double one = 1.0;
    const double zero = 0.0;
    const long long tensor_ld = chi * phys;
    if (transpose == 0) {
        for (long long d = 0; d < phys; ++d) {
            const double *Xd = x + chi * d;
            for (long long g = 0; g < phys; ++g) {
                const double *Bg = Adn + chi * g;
                double *pair = pair_work + (d + phys * g) * len2;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(chi), static_cast<int>(chi),
                                static_cast<int>(chi), &one, Xd,
                                static_cast<int>(tensor_ld), Bg,
                                static_cast<int>(tensor_ld), &zero, pair,
                                static_cast<int>(chi)),
                    "three_layer factored pair = Xd * Bg");
                if (st != TENET_NATIVE_SUCCESS) return st;
            }
        }
        for (long long e = 0; e < phys; ++e) {
            for (long long b = 0; b < phys; ++b) {
                double *accum = accum_work + (e + phys * b) * len2;
                for (long long d = 0; d < phys; ++d) {
                    for (long long g = 0; g < phys; ++g) {
                        const double alpha =
                            M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha != 0.0) {
                            const double *pair =
                                pair_work + (d + phys * g) * len2;
                            st = cublas_status(
                                cublasDaxpy(blas, static_cast<int>(len2),
                                            &alpha, pair, 1, accum, 1),
                                "three_layer factored accum += alpha pair");
                            if (st != TENET_NATIVE_SUCCESS) return st;
                        }
                    }
                }
            }
        }
        for (long long e = 0; e < phys; ++e) {
            double *Ye = y + chi * e;
            for (long long b = 0; b < phys; ++b) {
                const double *Ab = Aup + chi * b;
                const double *accum = accum_work + (e + phys * b) * len2;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(chi), static_cast<int>(chi),
                                static_cast<int>(chi), &one, Ab,
                                static_cast<int>(tensor_ld), accum,
                                static_cast<int>(chi), &one, Ye,
                                static_cast<int>(tensor_ld)),
                    "three_layer factored Ye += A' * accum");
                if (st != TENET_NATIVE_SUCCESS) return st;
            }
        }
    } else {
        for (long long e = 0; e < phys; ++e) {
            const double *Xe = x + chi * e;
            for (long long g = 0; g < phys; ++g) {
                const double *Bg = Adn + chi * g;
                double *pair = pair_work + (e + phys * g) * len2;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                                static_cast<int>(chi), static_cast<int>(chi),
                                static_cast<int>(chi), &one, Xe,
                                static_cast<int>(tensor_ld), Bg,
                                static_cast<int>(tensor_ld), &zero, pair,
                                static_cast<int>(chi)),
                    "three_layer factored pair = Xe * Bg'");
                if (st != TENET_NATIVE_SUCCESS) return st;
            }
        }
        for (long long d = 0; d < phys; ++d) {
            for (long long b = 0; b < phys; ++b) {
                double *accum = accum_work + (d + phys * b) * len2;
                for (long long e = 0; e < phys; ++e) {
                    for (long long g = 0; g < phys; ++g) {
                        const double alpha =
                            M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha != 0.0) {
                            const double *pair =
                                pair_work + (e + phys * g) * len2;
                            st = cublas_status(
                                cublasDaxpy(blas, static_cast<int>(len2),
                                            &alpha, pair, 1, accum, 1),
                                "three_layer factored transpose accum += alpha pair");
                            if (st != TENET_NATIVE_SUCCESS) return st;
                        }
                    }
                }
            }
        }
        for (long long d = 0; d < phys; ++d) {
            double *Yd = y + chi * d;
            for (long long b = 0; b < phys; ++b) {
                const double *Ab = Aup + chi * b;
                const double *accum = accum_work + (d + phys * b) * len2;
                st = cublas_status(
                    cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                                static_cast<int>(chi), static_cast<int>(chi),
                                static_cast<int>(chi), &one, Ab,
                                static_cast<int>(tensor_ld), accum,
                                static_cast<int>(chi), &one, Yd,
                                static_cast<int>(tensor_ld)),
                    "three_layer factored transpose Yd += A * accum");
                if (st != TENET_NATIVE_SUCCESS) return st;
            }
        }
    }
    return TENET_NATIVE_SUCCESS;
}

template <typename Apply>
int arnoldi_cuda_seeded(int64_t len, int64_t max_k, double breakdown_tol,
                        const double *seed_dev, int64_t seed_cols,
                        int64_t seed_ld, double *V, int64_t ldv,
                        double *H_host, int64_t ldh, double *beta, int64_t *m,
                        double *final_resnorm, Apply apply) {
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG arnoldi_cuda enter len=%lld max_k=%lld seed_cols=%lld ldv=%lld ldh=%lld seed=%p V=%p\n",
                     static_cast<long long>(len), static_cast<long long>(max_k),
                     static_cast<long long>(seed_cols), static_cast<long long>(ldv),
                     static_cast<long long>(ldh), static_cast<const void *>(seed_dev),
                     static_cast<void *>(V));
        std::fflush(stderr);
    }
    static thread_local DeviceBuffer H_buffer;
    static thread_local DeviceBuffer w_buffer;
    static thread_local DeviceBuffer g_buffer;
    static thread_local DeviceBuffer tmp_buffer;
    cublasHandle_t blas = nullptr;
    const std::size_t len_bytes = static_cast<std::size_t>(len) * sizeof(double);
    const std::size_t V_bytes = static_cast<std::size_t>(ldv) *
                                static_cast<std::size_t>(max_k + 1) *
                                sizeof(double);
    const std::size_t H_bytes = static_cast<std::size_t>(max_k + 1) *
                                static_cast<std::size_t>(max_k) * sizeof(double);
    const std::size_t g_bytes = static_cast<std::size_t>(max_k + 1) * sizeof(double);
    const bool do_profile = profile_enabled();
    double apply_seconds = 0.0;
    double orthog_seconds = 0.0;
    double norm_seconds = 0.0;
    int64_t applied_cols = 0;

    int status = TENET_NATIVE_SUCCESS;
    double *H = nullptr;
    double *w = nullptr;
    double *g = nullptr;
    double *tmp = nullptr;
    int64_t basis_cols = 0;
    status = ensure_device_buffer(H_buffer, H_bytes, "allocate Arnoldi H");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = ensure_device_buffer(w_buffer, len_bytes, "allocate Arnoldi w");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = ensure_device_buffer(g_buffer, g_bytes, "allocate Arnoldi g");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = ensure_device_buffer(tmp_buffer, len_bytes, "allocate Arnoldi tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    H = H_buffer.ptr;
    w = w_buffer.ptr;
    g = g_buffer.ptr;
    tmp = tmp_buffer.ptr;
    status = create_cublas_handle(&blas, "cublasCreate arnoldi");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG arnoldi_cuda after alloc H=%p w=%p g=%p tmp=%p\n",
                     static_cast<void *>(H), static_cast<void *>(w),
                     static_cast<void *>(g), static_cast<void *>(tmp));
        std::fflush(stderr);
    }
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG arnoldi_cuda before memset V\n");
        std::fflush(stderr);
    }
    CUDA_CHECK(cudaMemset(V, 0, V_bytes));
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG arnoldi_cuda before memset H\n");
        std::fflush(stderr);
    }
    CUDA_CHECK(cudaMemset(H, 0, H_bytes));
    *beta = 1.0;
    *m = 0;
    *final_resnorm = 0.0;
    if (max_k == 0) {
        goto copy_out;
    }
    if (seed_dev == nullptr || seed_cols <= 0 || seed_ld < len) {
        set_error("invalid CUDA Arnoldi seed basis");
        status = TENET_NATIVE_BACKEND_ERROR;
        goto cleanup;
    }
    {
        const int64_t max_seed_cols = std::min(seed_cols, max_k);
        for (int64_t col = 0; col < max_seed_cols; ++col) {
            double *v = V + basis_cols * ldv;
            CUDA_CHECK(cudaMemcpy(v, seed_dev + col * seed_ld, len_bytes,
                                  cudaMemcpyDeviceToDevice));
            for (int pass = 0; pass < 2; ++pass) {
                for (int64_t j = 0; j < basis_cols; ++j) {
                    double alpha = 0.0;
                    CUBLAS_CHECK(cublasDdot(blas, static_cast<int>(len),
                                            V + j * ldv, 1, v, 1, &alpha));
                    alpha = -alpha;
                    CUBLAS_CHECK(cublasDaxpy(blas, static_cast<int>(len),
                                             &alpha, V + j * ldv, 1, v, 1));
                }
            }
            double qn = 0.0;
            CUBLAS_CHECK(cublasDnrm2(blas, static_cast<int>(len), v, 1, &qn));
            if (!(qn > breakdown_tol) || !std::isfinite(qn)) {
                continue;
            }
            const double inv_qn = 1.0 / qn;
            CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &inv_qn, v, 1));
            ++basis_cols;
        }
    }
    if (basis_cols == 0) {
        goto copy_out;
    }

    for (long long j = 0; j < max_k && j < basis_cols; ++j) {
        const double *vj = V + j * ldv;
        double *hj = H + j * (max_k + 1);
        if (debug_enabled()) {
            std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld before apply vj=%p hj=%p\n",
                         j, static_cast<const void *>(vj), static_cast<void *>(hj));
            std::fflush(stderr);
        }
        auto section_start = SteadyClock::now();
        status = apply(blas, tmp, vj, w);
        ++applied_cols;
        if (do_profile) {
            CUDA_CHECK(cudaDeviceSynchronize());
            apply_seconds += seconds_since(section_start);
        }
        if (debug_enabled()) {
            std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld after apply status=%d\n",
                         j, status);
            std::fflush(stderr);
        }
        if (status != TENET_NATIVE_SUCCESS) {
            goto cleanup;
        }
        const int k = static_cast<int>(basis_cols);
        const double one = 1.0;
        const double zero = 0.0;
        const double minus_one = -1.0;
        section_start = SteadyClock::now();
        for (int pass = 0; pass < 2; ++pass) {
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld pass=%d before gemvT k=%d\n",
                             j, pass, k);
                std::fflush(stderr);
            }
            CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_T, static_cast<int>(len), k,
                                    &one, V, static_cast<int>(ldv), w, 1, &zero,
                                    g, 1));
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld pass=%d before daxpy\n",
                             j, pass);
                std::fflush(stderr);
            }
            CUBLAS_CHECK(cublasDaxpy(blas, k, &one, g, 1, hj, 1));
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld pass=%d before gemvN\n",
                             j, pass);
                std::fflush(stderr);
            }
            CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_N, static_cast<int>(len), k,
                                    &minus_one, V, static_cast<int>(ldv), g, 1,
                                    &one, w, 1));
        }
        if (do_profile) {
            CUDA_CHECK(cudaDeviceSynchronize());
            orthog_seconds += seconds_since(section_start);
        }
        if (debug_enabled()) {
            std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld before hnext norm\n", j);
            std::fflush(stderr);
        }
        double hnext = 0.0;
        section_start = SteadyClock::now();
        {
            const int st = device_norm2(len, w, &hnext);
            if (st != TENET_NATIVE_SUCCESS) goto cleanup;
        }
        if (do_profile) {
            norm_seconds += seconds_since(section_start);
        }
        if (debug_enabled()) {
            std::fprintf(stderr, "DEBUG arnoldi_cuda j=%lld hnext=%.16g\n", j, hnext);
            std::fflush(stderr);
        }
        *final_resnorm = hnext;
        *m = j + 1;
        if (hnext > breakdown_tol && basis_cols < max_k) {
            CUDA_CHECK(cudaMemcpy(hj + basis_cols, &hnext, sizeof(double),
                                  cudaMemcpyHostToDevice));
            double *vnext = V + basis_cols * ldv;
            CUBLAS_CHECK(cublasDcopy(blas, static_cast<int>(len), w, 1, vnext, 1));
            double inv_hnext = 1.0 / hnext;
            CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &inv_hnext,
                                     vnext, 1));
            ++basis_cols;
        } else if (hnext > breakdown_tol && basis_cols == max_k) {
            CUDA_CHECK(cudaMemcpy(hj + basis_cols, &hnext, sizeof(double),
                                  cudaMemcpyHostToDevice));
            double *vnext = V + basis_cols * ldv;
            CUBLAS_CHECK(cublasDcopy(blas, static_cast<int>(len), w, 1, vnext, 1));
            double inv_hnext = 1.0 / hnext;
            CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &inv_hnext,
                                     vnext, 1));
        } else if (hnext <= breakdown_tol && j + 1 >= basis_cols) {
            break;
        }
    }
    profile_cuda_arnoldi(len, max_k, *m, applied_cols, apply_seconds, orthog_seconds,
                         norm_seconds, *final_resnorm);

copy_out:
    for (long long col = 0; col < *m; ++col) {
        CUDA_CHECK(cudaMemcpy(H_host + col * ldh, H + col * (max_k + 1),
                              static_cast<std::size_t>(max_k + 1) * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

cleanup:
    if (blas != nullptr) {
        cublasDestroy(blas);
    }
    return status;
}

template <typename Apply>
int arnoldi_cuda(int64_t len, int64_t max_k, double breakdown_tol,
                 const double *x0_dev, double *V, int64_t ldv,
                 double *H_host, int64_t ldh, double *beta, int64_t *m,
                 double *final_resnorm, Apply apply) {
    return arnoldi_cuda_seeded(len, max_k, breakdown_tol, x0_dev, 1, len, V,
                               ldv, H_host, ldh, beta, m, final_resnorm,
                               apply);
}

template <typename Apply>
int arnoldi_cuda_prefilled(int64_t len, int64_t max_k, double breakdown_tol,
                           const double *initial_V, int64_t initial_cols,
                           int64_t initial_ldv, const double *initial_H,
                           int64_t initial_ldh, int64_t completed_cols,
                           double *V, int64_t ldv, double *H_host,
                           int64_t ldh, double *beta, int64_t *m,
                           double *final_resnorm, Apply apply) {
    if (initial_V == nullptr || initial_cols <= 0 || initial_ldv < len ||
        initial_H == nullptr || initial_ldh < completed_cols + 1 ||
        completed_cols < 0 || completed_cols > initial_cols ||
        initial_cols > max_k + 1) {
        set_error("invalid prefilled CUDA Arnoldi restart");
        return TENET_NATIVE_BACKEND_ERROR;
    }
    static thread_local DeviceBuffer H_buffer;
    static thread_local DeviceBuffer w_buffer;
    static thread_local DeviceBuffer g_buffer;
    static thread_local DeviceBuffer tmp_buffer;
    cublasHandle_t blas = nullptr;
    const std::size_t len_bytes = static_cast<std::size_t>(len) * sizeof(double);
    const std::size_t V_bytes = static_cast<std::size_t>(ldv) *
                                static_cast<std::size_t>(max_k + 1) *
                                sizeof(double);
    const std::size_t H_bytes = static_cast<std::size_t>(max_k + 1) *
                                static_cast<std::size_t>(max_k) *
                                sizeof(double);
    const std::size_t g_bytes =
        static_cast<std::size_t>(max_k + 1) * sizeof(double);
    const bool do_profile = profile_enabled();
    double apply_seconds = 0.0;
    double orthog_seconds = 0.0;
    double norm_seconds = 0.0;
    int64_t applied_cols = 0;
    int64_t basis_cols = initial_cols;
    int status = TENET_NATIVE_SUCCESS;
    double *H = nullptr;
    double *w = nullptr;
    double *g = nullptr;
    double *tmp = nullptr;

    status = ensure_device_buffer(H_buffer, H_bytes, "allocate prefilled Arnoldi H");
    if (status != TENET_NATIVE_SUCCESS) goto cleanup;
    status = ensure_device_buffer(w_buffer, len_bytes, "allocate prefilled Arnoldi w");
    if (status != TENET_NATIVE_SUCCESS) goto cleanup;
    status = ensure_device_buffer(g_buffer, g_bytes, "allocate prefilled Arnoldi g");
    if (status != TENET_NATIVE_SUCCESS) goto cleanup;
    status = ensure_device_buffer(tmp_buffer, len_bytes, "allocate prefilled Arnoldi tmp");
    if (status != TENET_NATIVE_SUCCESS) goto cleanup;
    H = H_buffer.ptr;
    w = w_buffer.ptr;
    g = g_buffer.ptr;
    tmp = tmp_buffer.ptr;
    status = create_cublas_handle(&blas, "cublasCreate prefilled arnoldi");
    if (status != TENET_NATIVE_SUCCESS) goto cleanup;

    CUDA_CHECK(cudaMemset(V, 0, V_bytes));
    CUDA_CHECK(cudaMemset(H, 0, H_bytes));
    for (int64_t col = 0; col < initial_cols; ++col) {
        CUDA_CHECK(cudaMemcpy(V + col * ldv, initial_V + col * initial_ldv,
                              len_bytes, cudaMemcpyDeviceToDevice));
    }
    for (int64_t col = 0; col < completed_cols; ++col) {
        CUDA_CHECK(cudaMemcpy(H + col * (max_k + 1),
                              initial_H + col * initial_ldh,
                              static_cast<std::size_t>(completed_cols + 1) *
                                  sizeof(double),
                              cudaMemcpyHostToDevice));
    }
    *beta = 1.0;
    *m = completed_cols;
    *final_resnorm = 0.0;
    if (completed_cols >= max_k) {
        profile_cuda_arnoldi(len, max_k, *m, 0, 0.0, 0.0, 0.0,
                             *final_resnorm);
        goto copy_out;
    }

    for (long long j = completed_cols; j < max_k && j < basis_cols; ++j) {
        const double *vj = V + j * ldv;
        double *hj = H + j * (max_k + 1);
        auto section_start = SteadyClock::now();
        status = apply(blas, tmp, vj, w);
        ++applied_cols;
        if (do_profile) {
            CUDA_CHECK(cudaDeviceSynchronize());
            apply_seconds += seconds_since(section_start);
        }
        if (status != TENET_NATIVE_SUCCESS) {
            goto cleanup;
        }
        const int k = static_cast<int>(basis_cols);
        const double one = 1.0;
        const double zero = 0.0;
        const double minus_one = -1.0;
        section_start = SteadyClock::now();
        for (int pass = 0; pass < 2; ++pass) {
            CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_T, static_cast<int>(len),
                                    k, &one, V, static_cast<int>(ldv), w, 1,
                                    &zero, g, 1));
            CUBLAS_CHECK(cublasDaxpy(blas, k, &one, g, 1, hj, 1));
            CUBLAS_CHECK(cublasDgemv(blas, CUBLAS_OP_N, static_cast<int>(len),
                                    k, &minus_one, V, static_cast<int>(ldv), g,
                                    1, &one, w, 1));
        }
        if (do_profile) {
            CUDA_CHECK(cudaDeviceSynchronize());
            orthog_seconds += seconds_since(section_start);
        }
        double hnext = 0.0;
        section_start = SteadyClock::now();
        {
            const int st = device_norm2(len, w, &hnext);
            if (st != TENET_NATIVE_SUCCESS) goto cleanup;
        }
        if (do_profile) {
            norm_seconds += seconds_since(section_start);
        }
        *final_resnorm = hnext;
        *m = j + 1;
        if (hnext > breakdown_tol && basis_cols < max_k) {
            CUDA_CHECK(cudaMemcpy(hj + basis_cols, &hnext, sizeof(double),
                                  cudaMemcpyHostToDevice));
            double *vnext = V + basis_cols * ldv;
            CUBLAS_CHECK(cublasDcopy(blas, static_cast<int>(len), w, 1, vnext, 1));
            double inv_hnext = 1.0 / hnext;
            CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &inv_hnext,
                                     vnext, 1));
            ++basis_cols;
        } else if (hnext > breakdown_tol && basis_cols == max_k) {
            CUDA_CHECK(cudaMemcpy(hj + basis_cols, &hnext, sizeof(double),
                                  cudaMemcpyHostToDevice));
            double *vnext = V + basis_cols * ldv;
            CUBLAS_CHECK(cublasDcopy(blas, static_cast<int>(len), w, 1, vnext, 1));
            double inv_hnext = 1.0 / hnext;
            CUBLAS_CHECK(cublasDscal(blas, static_cast<int>(len), &inv_hnext,
                                     vnext, 1));
        } else if (hnext <= breakdown_tol && j + 1 >= basis_cols) {
            break;
        }
    }
    profile_cuda_arnoldi(len, max_k, *m, applied_cols, apply_seconds,
                         orthog_seconds, norm_seconds, *final_resnorm);

copy_out:
    for (long long col = 0; col < *m; ++col) {
        CUDA_CHECK(cudaMemcpy(H_host + col * ldh, H + col * (max_k + 1),
                              static_cast<std::size_t>(max_k + 1) *
                                  sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

cleanup:
    if (blas != nullptr) {
        cublasDestroy(blas);
    }
    return status;
}

double host_dot(int64_t n, const double *x, const double *y) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }
    return acc;
}

double host_norm2(int64_t n, const double *x) {
    return std::sqrt(host_dot(n, x, x));
}

bool host_lapack_geev(int n, std::vector<double> &A, std::vector<double> &wr,
                      std::vector<double> &wi, std::vector<double> &vr) {
    char balanc = 'B';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';
    int64_t n_lapack = n;
    int64_t lda = n;
    int64_t ldvl = 1;
    int64_t ldvr = n;
    int64_t ilo = 0;
    int64_t ihi = 0;
    int64_t info = 0;
    int64_t lwork = -1;
    double vl_dummy = 0.0;
    double abnrm = 0.0;
    double work_query = 0.0;
    std::vector<double> scale(static_cast<std::size_t>(n));
    std::vector<double> rconde(static_cast<std::size_t>(n));
    std::vector<double> rcondv(static_cast<std::size_t>(n));
    std::vector<int64_t> iwork(static_cast<std::size_t>(2 * n));
    dgeevx_64_(&balanc, &jobvl, &jobvr, &sense, &n_lapack, A.data(), &lda,
               wr.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &ilo,
               &ihi, scale.data(), &abnrm, rconde.data(), rcondv.data(),
               &work_query, &lwork, iwork.data(), &info);
    if (info != 0) {
        set_error("LAPACK dgeevx workspace query failed");
        return false;
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeevx_64_(&balanc, &jobvl, &jobvr, &sense, &n_lapack, A.data(), &lda,
               wr.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &ilo,
               &ihi, scale.data(), &abnrm, rconde.data(), rcondv.data(),
               work.data(), &lwork, iwork.data(), &info);
    if (info != 0) {
        set_error("LAPACK dgeevx failed");
        return false;
    }
    return true;
}

bool host_choose_schur_restart_values(const std::vector<double> &wr,
                                      const std::vector<double> &wi,
                                      const std::vector<int> &order,
                                      int keep) {
    const int n = static_cast<int>(wr.size());
    host_schur_select_wr.clear();
    host_schur_select_wi.clear();
    std::vector<char> chosen(static_cast<std::size_t>(n), 0);
    const auto same_value = [&](int a, int b) {
        const double scale =
            std::max(1.0, std::max(std::hypot(wr[a], wi[a]),
                                   std::hypot(wr[b], wi[b])));
        return std::abs(wr[a] - wr[b]) <= 1e-8 * scale &&
               std::abs(wi[a] + wi[b]) <= 1e-8 * scale;
    };
    for (const int idx : order) {
        if (static_cast<int>(host_schur_select_wr.size()) >= keep) {
            break;
        }
        if (chosen[static_cast<std::size_t>(idx)]) {
            continue;
        }
        const double scale = std::max(1.0, std::hypot(wr[idx], wi[idx]));
        if (std::abs(wi[idx]) > 1e-10 * scale) {
            int partner = -1;
            for (int j = 0; j < n; ++j) {
                if (j != idx && !chosen[static_cast<std::size_t>(j)] &&
                    same_value(idx, j)) {
                    partner = j;
                    break;
                }
            }
            if (partner < 0) {
                continue;
            }
            if (static_cast<int>(host_schur_select_wr.size()) + 2 > keep &&
                !host_schur_select_wr.empty()) {
                break;
            }
            chosen[static_cast<std::size_t>(idx)] = 1;
            chosen[static_cast<std::size_t>(partner)] = 1;
            host_schur_select_wr.push_back(wr[idx]);
            host_schur_select_wi.push_back(wi[idx]);
            host_schur_select_wr.push_back(wr[partner]);
            host_schur_select_wi.push_back(wi[partner]);
        } else {
            chosen[static_cast<std::size_t>(idx)] = 1;
            host_schur_select_wr.push_back(wr[idx]);
            host_schur_select_wi.push_back(wi[idx]);
        }
    }
    return !host_schur_select_wr.empty();
}

bool host_lapack_schur_selected(int n, std::vector<double> &A,
                                std::vector<double> &wr,
                                std::vector<double> &wi,
                                std::vector<double> &vs, int *sdim_out) {
    char jobvs = 'V';
    char sort = 'S';
    int64_t n_lapack = n;
    int64_t lda = n;
    int64_t ldvs = n;
    int64_t sdim = 0;
    int64_t info = 0;
    int64_t lwork = -1;
    double work_query = 0.0;
    std::vector<int64_t> bwork(static_cast<std::size_t>(n));
    dgees_64_(&jobvs, &sort, host_schur_select_callback, &n_lapack, A.data(),
              &lda, &sdim, wr.data(), wi.data(), vs.data(), &ldvs,
              &work_query, &lwork, bwork.data(), &info);
    if (info != 0) {
        set_error("LAPACK dgees workspace query failed");
        return false;
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgees_64_(&jobvs, &sort, host_schur_select_callback, &n_lapack, A.data(),
              &lda, &sdim, wr.data(), wi.data(), vs.data(), &ldvs,
              work.data(), &lwork, bwork.data(), &info);
    if (info != 0) {
        set_error("LAPACK dgees failed");
        return false;
    }
    *sdim_out = static_cast<int>(sdim);
    return *sdim_out > 0;
}

void host_dgemm(char transa, char transb, int64_t m64, int64_t n64,
                int64_t k64, double alpha, const double *A, int64_t lda64,
                const double *B, int64_t ldb64, double beta, double *C,
                int64_t ldc64) {
    int64_t m = m64;
    int64_t n = n64;
    int64_t k = k64;
    int64_t lda = lda64;
    int64_t ldb = ldb64;
    int64_t ldc = ldc64;
    dgemm_64_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta,
              C, &ldc);
}

bool host_qrpos(int64_t m64, int64_t n64, const double *A, double *Q,
                double *R) {
    if (m64 < n64 || m64 <= 0 || n64 <= 0 || m64 > INT_MAX ||
        n64 > INT_MAX) {
        set_error("invalid qrpos dimensions");
        return false;
    }
    int64_t m = m64;
    int64_t n = n64;
    int64_t lda = m64;
    std::vector<double> work_A(static_cast<std::size_t>(m) *
                               static_cast<std::size_t>(n));
    std::copy(A, A + static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
              work_A.begin());
    std::vector<double> tau(static_cast<std::size_t>(n));
    int64_t info = 0;
    int64_t lwork = -1;
    double work_query = 0.0;
    dgeqrf_64_(&m, &n, work_A.data(), &lda, tau.data(), &work_query, &lwork,
               &info);
    if (info != 0) {
        set_error("LAPACK dgeqrf workspace query failed");
        return false;
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgeqrf_64_(&m, &n, work_A.data(), &lda, tau.data(), work.data(), &lwork,
               &info);
    if (info != 0) {
        set_error("LAPACK dgeqrf failed");
        return false;
    }
    std::fill(R, R + static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
              0.0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i <= j; ++i) {
            R[idx2(i, j, n)] = work_A[idx2(i, j, m)];
        }
    }
    lwork = -1;
    work_query = 0.0;
    dorgqr_64_(&m, &n, &n, work_A.data(), &lda, tau.data(), &work_query,
               &lwork, &info);
    if (info != 0) {
        set_error("LAPACK dorgqr workspace query failed");
        return false;
    }
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work_query));
    work.assign(static_cast<std::size_t>(lwork), 0.0);
    dorgqr_64_(&m, &n, &n, work_A.data(), &lda, tau.data(), work.data(),
               &lwork, &info);
    if (info != 0) {
        set_error("LAPACK dorgqr failed");
        return false;
    }
    std::copy(work_A.begin(), work_A.end(), Q);
    for (int64_t j = 0; j < n; ++j) {
        const double phase = R[idx2(j, j, n)] < 0.0 ? -1.0 : 1.0;
        if (phase < 0.0) {
            for (int64_t i = 0; i < m; ++i) {
                Q[idx2(i, j, m)] = -Q[idx2(i, j, m)];
            }
            for (int64_t col = 0; col < n; ++col) {
                R[idx2(j, col, n)] = -R[idx2(j, col, n)];
            }
        }
    }
    return true;
}

bool host_lqpos(int64_t m64, int64_t n64, const double *A, double *L,
                double *Q) {
    if (m64 > n64 || m64 <= 0 || n64 <= 0) {
        set_error("invalid lqpos dimensions");
        return false;
    }
    const long long m = m64;
    const long long n = n64;
    std::vector<double> At(static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(m));
    for (long long j = 0; j < n; ++j) {
        for (long long i = 0; i < m; ++i) {
            At[idx2(j, i, n)] = A[idx2(i, j, m)];
        }
    }
    std::vector<double> Qr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(m));
    std::vector<double> Rr(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
    if (!host_qrpos(n, m, At.data(), Qr.data(), Rr.data())) {
        return false;
    }
    for (long long j = 0; j < m; ++j) {
        for (long long i = 0; i < m; ++i) {
            L[idx2(i, j, m)] = Rr[idx2(j, i, m)];
        }
    }
    for (long long j = 0; j < n; ++j) {
        for (long long i = 0; i < m; ++i) {
            Q[idx2(i, j, m)] = Qr[idx2(j, i, n)];
        }
    }
    return true;
}

bool select_dominant_ritz_coeff_power(int64_t m, double beta,
                                      const double *H, int64_t ldh,
                                      double *coeff_out) {
    if (m <= 0 || m > INT_MAX) {
        set_error("dominant Ritz search has invalid dimension");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(n),
                          0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hm[idx2(row, col, n)] = H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> coeff(static_cast<std::size_t>(n), 0.0);
    std::vector<double> tmp(static_cast<std::size_t>(n), 0.0);
    std::vector<double> work(static_cast<std::size_t>(n), 0.0);
    coeff[0] = 1.0;
    double lambda = 0.0;
    for (int iter = 0; iter < 256; ++iter) {
        for (int row = 0; row < n; ++row) {
            double acc = 0.0;
            for (int col = 0; col < n; ++col) {
                acc += Hm[idx2(row, col, n)] * coeff[static_cast<size_t>(col)];
            }
            tmp[static_cast<size_t>(row)] = acc;
        }
        const double nrm = host_norm2(n, tmp.data());
        if (!(nrm > 0.0) || !std::isfinite(nrm)) {
            set_error("dominant Ritz vector search failed");
            return false;
        }
        const double inv_nrm = 1.0 / nrm;
        for (int i = 0; i < n; ++i) {
            tmp[static_cast<size_t>(i)] *= inv_nrm;
        }
        for (int i = 0; i < n; ++i) {
            work[static_cast<size_t>(i)] = 0.0;
            for (int col = 0; col < n; ++col) {
                work[static_cast<size_t>(i)] +=
                    Hm[idx2(i, col, n)] * tmp[static_cast<size_t>(col)];
            }
        }
        lambda = host_dot(n, tmp.data(), work.data());
        double residual = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = work[static_cast<size_t>(i)] - lambda *
                                                  tmp[static_cast<size_t>(i)];
            residual += d * d;
        }
        coeff.swap(tmp);
        if (!std::isfinite(lambda)) {
            set_error("dominant Ritz value is not finite");
            return false;
        }
        if (residual <= 1e-12 * (1.0 + std::abs(lambda))) {
            break;
        }
    }
    for (int i = 0; i < n; ++i) {
        coeff_out[i] = beta * coeff[static_cast<std::size_t>(i)];
    }
    return true;
}

bool host_ritz_precedes(double ar, double ai, double br, double bi,
                        RitzTarget target) {
    switch (target) {
    case RitzTarget::LargestMagnitude: {
        const double ma = std::hypot(ar, ai);
        const double mb = std::hypot(br, bi);
        const double scale = std::max(1.0, std::max(ma, mb));
        if (std::abs(ma - mb) > 1e-10 * scale) {
            return ma > mb;
        }
        return ar > br;
    }
    }
    return false;
}

bool select_ritz_coeff_host(int64_t m, double beta, double final_resnorm,
                            const double *H, int64_t ldh,
                            std::vector<double> &coeff,
                            double *ritz_resnorm,
                            RitzTarget target = RitzTarget::LargestMagnitude) {
    if (m <= 0 || m > INT_MAX) {
        set_error("CUDA Ritz search has invalid dimension");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n),
                           0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hm[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!host_lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }

    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (host_ritz_precedes(wr[i], wi[i], wr[best], wi[best], target)) {
            best = i;
        }
    }

    coeff.assign(static_cast<std::size_t>(m), 0.0);
    if (std::abs(wi[best]) > 1e-8 * std::max(1.0, std::abs(wr[best]))) {
        if (target != RitzTarget::LargestMagnitude ||
            !select_dominant_ritz_coeff_power(m, beta, H, ldh, coeff.data())) {
            set_error("selected CUDA Ritz value is complex");
            return false;
        }
    } else {
        for (int64_t col = 0; col < m; ++col) {
            coeff[static_cast<std::size_t>(col)] =
                beta * vr[idx2(col, best, m)];
        }
    }

    const double coeff_norm = host_norm2(m, coeff.data());
    if (!(coeff_norm > 0.0) || !std::isfinite(coeff_norm)) {
        set_error("CUDA Ritz coefficient vector has invalid norm");
        return false;
    }
    if (ritz_resnorm != nullptr) {
        *ritz_resnorm =
            final_resnorm * std::abs(coeff[static_cast<std::size_t>(m - 1)]) /
            coeff_norm;
    }
    return true;
}

bool select_ritz_values_host(int64_t m, const double *H, int64_t ldh,
                             int64_t nvalues, double *lambda_real,
                             double *lambda_imag,
                             RitzTarget target = RitzTarget::LargestMagnitude) {
    if (m <= 0 || m > INT_MAX || nvalues <= 0 || lambda_real == nullptr ||
        lambda_imag == nullptr) {
        set_error("invalid CUDA Ritz value output dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n),
                           0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hm[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!host_lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return host_ritz_precedes(wr[a], wi[a], wr[b], wi[b], target);
    });
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t i = 0; i < nvalues; ++i) {
        if (i < m) {
            const int idx = order[static_cast<std::size_t>(i)];
            lambda_real[i] = wr[static_cast<std::size_t>(idx)];
            lambda_imag[i] = wi[static_cast<std::size_t>(idx)];
        } else {
            lambda_real[i] = nan;
            lambda_imag[i] = nan;
        }
    }
    return true;
}

bool select_ritz_convergence_residual_host(
    int64_t m, double beta, double final_resnorm, const double *H,
    int64_t ldh, int64_t nvalues, double *max_ritz_resnorm,
    RitzTarget target = RitzTarget::LargestMagnitude) {
    if (m <= 0 || m > INT_MAX || nvalues <= 0 ||
        max_ritz_resnorm == nullptr) {
        set_error("invalid CUDA Ritz convergence dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(n),
                           0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hm[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!host_lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return host_ritz_precedes(wr[a], wi[a], wr[b], wi[b], target);
    });

    const int ncheck = std::min<int>(n, static_cast<int>(nvalues));
    double max_res = 0.0;
    for (int pos = 0; pos < ncheck; ++pos) {
        int idx = order[static_cast<std::size_t>(pos)];
        double tail = 0.0;
        double coeff_norm_sq = 0.0;
        if (std::abs(wi[static_cast<std::size_t>(idx)]) >
            1e-8 * std::max(1.0, std::abs(wr[static_cast<std::size_t>(idx)]))) {
            if (wi[static_cast<std::size_t>(idx)] < 0.0 && idx > 0) {
                --idx;
            }
            if (idx + 1 >= n) {
                set_error("complex CUDA Ritz pair is missing its conjugate column");
                return false;
            }
            for (int64_t row = 0; row < m; ++row) {
                const double cr = beta * vr[idx2(row, idx, m)];
                const double ci = beta * vr[idx2(row, idx + 1, m)];
                coeff_norm_sq += cr * cr + ci * ci;
            }
            const double tr = beta * vr[idx2(m - 1, idx, m)];
            const double ti = beta * vr[idx2(m - 1, idx + 1, m)];
            tail = std::hypot(tr, ti);
        } else {
            for (int64_t row = 0; row < m; ++row) {
                const double c = beta * vr[idx2(row, idx, m)];
                coeff_norm_sq += c * c;
            }
            tail = std::abs(beta * vr[idx2(m - 1, idx, m)]);
        }
        const double coeff_norm = std::sqrt(coeff_norm_sq);
        if (!(coeff_norm > 0.0) || !std::isfinite(coeff_norm)) {
            set_error("CUDA Ritz convergence coefficient norm is invalid");
            return false;
        }
        const double res = final_resnorm * tail / coeff_norm;
        if (!std::isfinite(res)) {
            set_error("CUDA Ritz convergence residual is not finite");
            return false;
        }
        max_res = std::max(max_res, res);
    }
    *max_ritz_resnorm = max_res;
    return true;
}

bool select_restart_schur_basis_host(int64_t m, const double *H, int64_t ldh,
                                     int64_t keep, double *schur_basis,
                                     double *schur_form,
                                     int64_t *schur_cols) {
    if (m <= 0 || m > INT_MAX || keep <= 0) {
        set_error("invalid CUDA restart coefficient dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n),
                           0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hm[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!host_lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return host_ritz_precedes(wr[a], wi[a], wr[b], wi[b],
                                  RitzTarget::LargestMagnitude);
    });

    if (!host_choose_schur_restart_values(wr, wi, order,
                                          static_cast<int>(keep))) {
        set_error("failed to choose CUDA Schur restart values");
        return false;
    }

    std::vector<double> Hs(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n),
                           0.0);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            Hs[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<long long>(ldh))];
        }
    }
    std::vector<double> schur_wr(static_cast<std::size_t>(n));
    std::vector<double> schur_wi(static_cast<std::size_t>(n));
    std::vector<double> schur_vecs(static_cast<std::size_t>(n) *
                                   static_cast<std::size_t>(n));
    int sdim = 0;
    const bool schur_ok = host_lapack_schur_selected(
        n, Hs, schur_wr, schur_wi, schur_vecs, &sdim);
    host_schur_select_wr.clear();
    host_schur_select_wi.clear();
    if (!schur_ok) {
        return false;
    }
    if (sdim <= 0 || sdim > keep) {
        set_error("selected CUDA Schur restart dimension is invalid");
        return false;
    }
    for (int64_t col = 0; col < sdim; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            schur_basis[idx2(row, col, m)] = schur_vecs[idx2(row, col, m)];
        }
    }
    for (int64_t col = 0; col < sdim; ++col) {
        for (int64_t row = 0; row < sdim; ++row) {
            schur_form[idx2(row, col, keep)] = Hs[idx2(row, col, m)];
        }
    }
    *schur_cols = sdim;
    return true;
}

struct HostRealHouseholder {
    double beta = 0.0;
    std::vector<double> v;
    double nu = 0.0;
};

HostRealHouseholder host_row_householder_last_pivot(
    const std::vector<double> &x) {
    HostRealHouseholder h;
    const int64_t n = static_cast<int64_t>(x.size());
    h.v = x;
    if (n <= 0) {
        return h;
    }
    double sigma = 0.0;
    for (int64_t i = 0; i < n - 1; ++i) {
        sigma += x[static_cast<std::size_t>(i)] *
                 x[static_cast<std::size_t>(i)];
    }
    const double pivot = x[static_cast<std::size_t>(n - 1)];
    const double nu = std::sqrt(pivot * pivot + sigma);
    h.nu = nu;
    if (sigma == 0.0 && pivot == nu) {
        h.beta = 0.0;
        return h;
    }
    double vi = pivot;
    if (pivot < 0.0) {
        vi = pivot - nu;
    } else {
        vi = -sigma / (pivot + nu);
    }
    if (vi == 0.0 || !std::isfinite(vi) || !(nu > 0.0)) {
        h.beta = 0.0;
        return h;
    }
    for (int64_t i = 0; i < n - 1; ++i) {
        h.v[static_cast<std::size_t>(i)] /= vi;
    }
    h.v[static_cast<std::size_t>(n - 1)] = 1.0;
    h.beta = -vi / nu;
    return h;
}

void host_apply_householder_left(int64_t rows, int64_t cols,
                                 const HostRealHouseholder &h, double *A,
                                 int64_t lda) {
    if (h.beta == 0.0) {
        return;
    }
    for (int64_t col = 0; col < cols; ++col) {
        double mu = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            mu += h.v[static_cast<std::size_t>(i)] *
                  A[idx2(i, col, lda)];
        }
        mu *= h.beta;
        for (int64_t i = 0; i < rows; ++i) {
            A[idx2(i, col, lda)] -=
                mu * h.v[static_cast<std::size_t>(i)];
        }
    }
}

void host_apply_householder_right(int64_t active_rows, int64_t cols,
                                  const HostRealHouseholder &h, double *A,
                                  int64_t lda) {
    if (h.beta == 0.0) {
        return;
    }
    std::vector<double> work(static_cast<std::size_t>(active_rows), 0.0);
    for (int64_t col = 0; col < cols; ++col) {
        const double vc = h.v[static_cast<std::size_t>(col)];
        for (int64_t row = 0; row < active_rows; ++row) {
            work[static_cast<std::size_t>(row)] +=
                A[idx2(row, col, lda)] * vc;
        }
    }
    for (int64_t col = 0; col < cols; ++col) {
        const double vc = h.beta * h.v[static_cast<std::size_t>(col)];
        for (int64_t row = 0; row < active_rows; ++row) {
            A[idx2(row, col, lda)] -=
                work[static_cast<std::size_t>(row)] * vc;
        }
    }
}

void host_restore_krylovkit_arnoldi_form(int64_t k, double *Hwork,
                                         int64_t ldh, double *basis_coeffs,
                                         int64_t ldb) {
    for (int64_t j = k; j >= 1; --j) {
        std::vector<double> row(static_cast<std::size_t>(j));
        for (int64_t col = 0; col < j; ++col) {
            row[static_cast<std::size_t>(col)] = Hwork[idx2(j, col, ldh)];
        }
        const HostRealHouseholder h = host_row_householder_last_pivot(row);
        for (int64_t col = 0; col < j - 1; ++col) {
            Hwork[idx2(j, col, ldh)] = 0.0;
        }
        Hwork[idx2(j, j - 1, ldh)] = h.nu;
        host_apply_householder_left(j, k, h, Hwork, ldh);
        host_apply_householder_right(j, j, h, Hwork, ldh);
        host_apply_householder_right(ldb, j, h, basis_coeffs, ldb);
    }
}

int build_compressed_restart_cuda(cublasHandle_t blas, int64_t len,
                                  int64_t max_k, int64_t m,
                                  const double *V_dev, const double *H,
                                  int64_t ldh, double final_resnorm,
                                  int64_t keep, double breakdown_tol,
                                  double *coeff_dev, double *restart_dev,
                                  double *restart_H, int64_t *restart_cols,
                                  int64_t *completed_cols) {
    if (m <= 0 || keep <= 1 || keep >= max_k || V_dev == nullptr ||
        H == nullptr || coeff_dev == nullptr || restart_dev == nullptr ||
        restart_H == nullptr) {
        set_error("invalid CUDA compressed restart inputs");
        return TENET_NATIVE_BACKEND_ERROR;
    }
    const int64_t kmax = std::min<int64_t>(keep, m);
    std::vector<double> C(static_cast<std::size_t>(m) *
                          static_cast<std::size_t>(kmax),
                          0.0);
    std::vector<double> T(static_cast<std::size_t>(kmax) *
                          static_cast<std::size_t>(kmax),
                          0.0);
    int64_t k = 0;
    if (!select_restart_schur_basis_host(m, H, ldh, kmax, C.data(), T.data(),
                                         &k) ||
        k <= 1) {
        return TENET_NATIVE_BACKEND_ERROR;
    }

    std::vector<double> Hwork(static_cast<std::size_t>(k + 1) *
                              static_cast<std::size_t>(k),
                              0.0);
    for (int64_t col = 0; col < k; ++col) {
        for (int64_t row = 0; row < k; ++row) {
            Hwork[idx2(row, col, k + 1)] = T[idx2(row, col, kmax)];
        }
    }
    for (int64_t col = 0; col < k; ++col) {
        Hwork[idx2(k, col, k + 1)] =
            final_resnorm * C[idx2(m - 1, col, m)];
    }
    host_restore_krylovkit_arnoldi_form(k, Hwork.data(), k + 1, C.data(), m);
    const double tail_norm = std::abs(Hwork[idx2(k, k - 1, k + 1)]);

    int status = cuda_status(
        cudaMemset(restart_dev, 0,
                   static_cast<std::size_t>(len) *
                       static_cast<std::size_t>(max_k + 1) * sizeof(double)),
        "clear CUDA restart basis");
    if (status != TENET_NATIVE_SUCCESS) return status;
    const double one = 1.0;
    const double zero = 0.0;
    for (int64_t col = 0; col < k; ++col) {
        status = cuda_status(
            cudaMemcpy(coeff_dev, C.data() + col * m,
                       static_cast<std::size_t>(m) * sizeof(double),
                       cudaMemcpyHostToDevice),
            "copy compressed restart coefficients");
        if (status != TENET_NATIVE_SUCCESS) return status;
        status = cublas_status(
            cublasDgemv(blas, CUBLAS_OP_N, static_cast<int>(len),
                        static_cast<int>(m), &one, V_dev, static_cast<int>(len),
                        coeff_dev, 1, &zero, restart_dev + col * len, 1),
            "compressed restart vector = V * coeff");
        if (status != TENET_NATIVE_SUCCESS) return status;
    }

    std::fill(restart_H, restart_H + (max_k + 1) * max_k, 0.0);
    for (int64_t col = 0; col < k; ++col) {
        for (int64_t row = 0; row < k + 1; ++row) {
            restart_H[idx2(row, col, max_k + 1)] =
                Hwork[idx2(row, col, k + 1)];
        }
    }
    *completed_cols = k;
    if (tail_norm > breakdown_tol) {
        status = cuda_status(cudaMemcpy(restart_dev + k * len, V_dev + m * len,
                                        static_cast<std::size_t>(len) *
                                            sizeof(double),
                                        cudaMemcpyDeviceToDevice),
                             "copy compressed restart residual vector");
        if (status != TENET_NATIVE_SUCCESS) return status;
        *restart_cols = k + 1;
    } else {
        *restart_cols = k;
    }
    return TENET_NATIVE_SUCCESS;
}

template <typename Apply>
int restarted_arnoldi_ritz_values_cuda(int64_t len, int64_t max_k,
                                       double breakdown_tol, const double *x0,
                                       int64_t nvalues, double *lambda_real,
                                       double *lambda_imag, int64_t *m_out,
                                       Apply apply) {
    if (len <= 0 || max_k <= 0 || max_k > len || breakdown_tol < 0.0 ||
        x0 == nullptr || nvalues <= 0 || lambda_real == nullptr ||
        lambda_imag == nullptr || m_out == nullptr || len > INT_MAX ||
        max_k > INT_MAX) {
        set_error("invalid CUDA restarted Ritz inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    static thread_local DeviceBuffer V_buffer;
    static thread_local DeviceBuffer coeff_buffer;
    static thread_local DeviceBuffer restart_buffer;
    static thread_local DeviceBuffer single_buffer;
    const int64_t thick_keep = dominant_thick_keep_count(max_k);
    const std::size_t V_bytes =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(max_k + 1) *
        sizeof(double);
    const std::size_t coeff_bytes =
        static_cast<std::size_t>(max_k) * sizeof(double);
    std::vector<double> H(static_cast<std::size_t>(max_k + 1) *
                          static_cast<std::size_t>(max_k));
    std::vector<double> restart_H(static_cast<std::size_t>(max_k + 1) *
                                  static_cast<std::size_t>(max_k));
    double *V_dev = nullptr;
    double *coeff_dev = nullptr;
    double *restart_dev = nullptr;
    double *single_dev = nullptr;
    cublasHandle_t blas = nullptr;

    int status = ensure_device_buffer(V_buffer, V_bytes,
                                      "allocate CUDA Ritz Arnoldi V");
    if (status != TENET_NATIVE_SUCCESS) return status;
    status = ensure_device_buffer(coeff_buffer, coeff_bytes,
                                  "allocate CUDA Ritz coefficients");
    if (status != TENET_NATIVE_SUCCESS) return status;
    status = ensure_device_buffer(restart_buffer, V_bytes,
                                  "allocate CUDA Ritz restart basis");
    if (status != TENET_NATIVE_SUCCESS) return status;
    status = ensure_device_buffer(single_buffer,
                                  static_cast<std::size_t>(len) * sizeof(double),
                                  "allocate CUDA Ritz single restart");
    if (status != TENET_NATIVE_SUCCESS) return status;
    V_dev = V_buffer.ptr;
    coeff_dev = coeff_buffer.ptr;
    restart_dev = restart_buffer.ptr;
    single_dev = single_buffer.ptr;
    status = create_cublas_handle(&blas, "cublasCreate restarted Ritz");
    if (status != TENET_NATIVE_SUCCESS) return status;

    const double *seed = x0;
    int64_t seed_cols = 1;
    bool have_compressed_restart = false;
    int64_t restart_cols = 0;
    int64_t completed_cols = 0;
    const int max_blocks = dominant_restart_blocks(len, max_k);
    const double one = 1.0;
    const double zero = 0.0;
    last_dominant_relres = std::numeric_limits<double>::infinity();
    for (int block = 0; block < max_blocks; ++block) {
        double beta = 0.0;
        int64_t m = 0;
        double final_resnorm = 0.0;
        status = have_compressed_restart
                     ? arnoldi_cuda_prefilled(
                           len, max_k, breakdown_tol, restart_dev,
                           restart_cols, len, restart_H.data(), max_k + 1,
                           completed_cols, V_dev, len, H.data(), max_k + 1,
                           &beta, &m, &final_resnorm, apply)
                     : arnoldi_cuda_seeded(len, max_k, breakdown_tol, seed,
                                           seed_cols, len, V_dev, len,
                                           H.data(), max_k + 1, &beta, &m,
                                           &final_resnorm, apply);
        if (status != TENET_NATIVE_SUCCESS) goto cleanup;
        std::vector<double> coeff;
        double ritz_resnorm = std::numeric_limits<double>::infinity();
        if (!select_ritz_coeff_host(m, beta, final_resnorm, H.data(), max_k + 1,
                                    coeff, &ritz_resnorm) ||
            !select_ritz_values_host(m, H.data(), max_k + 1, nvalues,
                                     lambda_real, lambda_imag)) {
            status = TENET_NATIVE_BACKEND_ERROR;
            goto cleanup;
        }
        *m_out = m;
        double convergence_resnorm = ritz_resnorm;
        if (nvalues > 1 &&
            !select_ritz_convergence_residual_host(
                m, beta, final_resnorm, H.data(), max_k + 1, nvalues,
                &convergence_resnorm)) {
            status = TENET_NATIVE_BACKEND_ERROR;
            goto cleanup;
        }
        last_dominant_relres = convergence_resnorm;
        if (block + 1 == max_blocks) {
            status = TENET_NATIVE_SUCCESS;
            goto cleanup;
        }
        if (thick_keep > 1 &&
            build_compressed_restart_cuda(
                blas, len, max_k, m, V_dev, H.data(), max_k + 1,
                final_resnorm, thick_keep, breakdown_tol, coeff_dev,
                restart_dev, restart_H.data(), &restart_cols,
                &completed_cols) == TENET_NATIVE_SUCCESS) {
            have_compressed_restart = true;
            seed = nullptr;
            seed_cols = 0;
        } else {
            status = cuda_status(cudaMemcpy(coeff_dev, coeff.data(),
                                            static_cast<std::size_t>(m) *
                                                sizeof(double),
                                            cudaMemcpyHostToDevice),
                                 "copy CUDA Ritz fallback coefficients");
            if (status != TENET_NATIVE_SUCCESS) goto cleanup;
            status = cublas_status(cublasDgemv(
                                       blas, CUBLAS_OP_N, static_cast<int>(len),
                                       static_cast<int>(m), &one, V_dev,
                                       static_cast<int>(len), coeff_dev, 1,
                                       &zero, single_dev, 1),
                                   "CUDA Ritz fallback y = V * coeff");
            if (status != TENET_NATIVE_SUCCESS) goto cleanup;
            double yn = 0.0;
            status = device_norm2(len, single_dev, &yn);
            if (status != TENET_NATIVE_SUCCESS) goto cleanup;
            if (!(yn > 0.0) || !std::isfinite(yn)) {
                set_error("CUDA Ritz fallback vector has invalid norm");
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            {
                const double inv_yn = 1.0 / yn;
                status = cublas_status(cublasDscal(blas, static_cast<int>(len),
                                                   &inv_yn, single_dev, 1),
                                       "normalize CUDA Ritz fallback");
                if (status != TENET_NATIVE_SUCCESS) goto cleanup;
            }
            seed = single_dev;
            seed_cols = 1;
            have_compressed_restart = false;
            restart_cols = 0;
            completed_cols = 0;
        }
    }

cleanup:
    if (blas != nullptr) {
        cublasDestroy(blas);
    }
    return status;
}

template <typename Apply>
int dominant_arnoldi_vector_cuda(int64_t len, int64_t max_k,
                                double breakdown_tol, const double *x0,
                                double *y_out, Apply apply) {
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_cuda enter len=%lld max_k=%lld tol=%.3e x0=%p y=%p\n",
                     static_cast<long long>(len), static_cast<long long>(max_k),
                     breakdown_tol, static_cast<const void *>(x0),
                     static_cast<void *>(y_out));
        std::fflush(stderr);
    }
    if (len <= 0 || max_k < 1 || breakdown_tol < 0.0 || x0 == nullptr ||
        y_out == nullptr) {
        set_error("invalid dominant Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const int64_t effective_k = std::min(max_k, len);
    if (effective_k > INT_MAX || len > INT_MAX) {
        set_error("dominant Arnoldi dimensions exceed BLAS int range");
        return TENET_NATIVE_INVALID_VALUE;
    }
    static thread_local DeviceBuffer V_buffer;
    static thread_local DeviceBuffer coeff_buffer;
    static thread_local DeviceBuffer restart_buffer;
    cublasHandle_t blas = nullptr;
    const int64_t thick_keep = dominant_thick_keep_count(effective_k);
    const std::size_t V_bytes =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(effective_k + 1) *
        sizeof(double);
    const std::size_t coeff_bytes =
        static_cast<std::size_t>(effective_k) * sizeof(double);
    const std::size_t restart_bytes =
        static_cast<std::size_t>(len) *
        static_cast<std::size_t>(effective_k + 1) * sizeof(double);
    std::vector<double> H(static_cast<std::size_t>(effective_k + 1) *
                          static_cast<std::size_t>(effective_k));
    std::vector<double> restart_H(static_cast<std::size_t>(effective_k + 1) *
                                  static_cast<std::size_t>(effective_k));
    int status = TENET_NATIVE_SUCCESS;
    double *V_dev = nullptr;
    double *coeff_dev = nullptr;
    double *restart_dev = nullptr;
    status = ensure_device_buffer(V_buffer, V_bytes,
                                  "allocate dominant Arnoldi V");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = ensure_device_buffer(coeff_buffer, coeff_bytes,
                                  "allocate Ritz coefficient vector");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = ensure_device_buffer(restart_buffer, restart_bytes,
                                  "allocate dominant restart basis");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    V_dev = V_buffer.ptr;
    coeff_dev = coeff_buffer.ptr;
    restart_dev = restart_buffer.ptr;
    status = create_cublas_handle(&blas, "cublasCreate dominant Ritz");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }

    {
        const double *seed = x0;
        int64_t seed_cols = 1;
        bool have_compressed_restart = false;
        int64_t restart_cols = 0;
        int64_t completed_cols = 0;
        const int max_blocks = dominant_restart_blocks(len, effective_k);
        const double eig_tol = dominant_residual_tol(breakdown_tol);
        const int64_t convergence_nvalues = dominant_convergence_nvalues();
        const double one = 1.0;
        const double zero = 0.0;
        last_dominant_relres = std::numeric_limits<double>::infinity();
        if (debug_enabled()) {
            std::fprintf(stderr, "DEBUG dominant_cuda allocated effective_k=%lld blocks=%d eig_tol=%.3e\n",
                         static_cast<long long>(effective_k), max_blocks, eig_tol);
            std::fflush(stderr);
        }
        for (int block = 0; block < max_blocks; ++block) {
            double beta = 0.0;
            int64_t m = 0;
            double final_resnorm = 0.0;
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d before arnoldi seed=%p\n",
                             block, static_cast<const void *>(seed));
                std::fflush(stderr);
            }
            status = have_compressed_restart
                         ? arnoldi_cuda_prefilled(
                               len, effective_k, breakdown_tol, restart_dev,
                               restart_cols, len, restart_H.data(),
                               effective_k + 1, completed_cols, V_dev, len,
                               H.data(), effective_k + 1, &beta, &m,
                               &final_resnorm, apply)
                         : arnoldi_cuda_seeded(len, effective_k, breakdown_tol,
                                               seed, seed_cols, len, V_dev, len,
                                               H.data(), effective_k + 1, &beta,
                                               &m, &final_resnorm, apply);
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d after arnoldi status=%d beta=%.16g m=%lld hnext=%.3e\n",
                             block, status, beta, static_cast<long long>(m),
                             final_resnorm);
                std::fflush(stderr);
            }
            if (status != TENET_NATIVE_SUCCESS) {
                goto cleanup;
            }
            if (m <= 0) {
                set_error("dominant Arnoldi produced no basis");
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            const bool want_thick_restart =
                thick_keep > 1 && block + 1 < max_blocks;
            std::vector<double> coeff;
            double ritz_resnorm = std::numeric_limits<double>::infinity();
            if (!select_ritz_coeff_host(m, beta, final_resnorm, H.data(),
                                        effective_k + 1, coeff,
                                        &ritz_resnorm)) {
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            const double coeff_norm = host_norm2(m, coeff.data());
            if (!(coeff_norm > 0.0) || !std::isfinite(coeff_norm)) {
                set_error("dominant Ritz coefficient vector has invalid norm");
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            if (!std::isfinite(ritz_resnorm)) {
                set_error("dominant Arnoldi Ritz residual is not finite");
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            double convergence_resnorm = ritz_resnorm;
            if (convergence_nvalues > 1 &&
                !select_ritz_convergence_residual_host(
                    m, beta, final_resnorm, H.data(), effective_k + 1,
                    convergence_nvalues, &convergence_resnorm)) {
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d after select\n", block);
                std::fflush(stderr);
            }
            status = cuda_status(cudaMemcpy(coeff_dev, coeff.data(),
                                            static_cast<std::size_t>(m) * sizeof(double),
                                            cudaMemcpyHostToDevice),
                                 "copy Ritz coefficients to device");
            if (status != TENET_NATIVE_SUCCESS) {
                goto cleanup;
            }
            status = cublas_status(cublasDgemv(
                                       blas, CUBLAS_OP_N, static_cast<int>(len),
                                       static_cast<int>(m), &one, V_dev,
                                       static_cast<int>(len), coeff_dev, 1, &zero,
                                       y_out, 1),
                                   "dominant Ritz y = V * coeff");
            if (status != TENET_NATIVE_SUCCESS) {
                goto cleanup;
            }
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d after gemv\n", block);
                std::fflush(stderr);
            }
            double yn = 0.0;
            status = device_norm2(len, y_out, &yn);
            if (status != TENET_NATIVE_SUCCESS) {
                goto cleanup;
            }
            if (!(yn > 0.0) || !std::isfinite(yn)) {
                set_error("dominant Ritz vector has invalid norm");
                status = TENET_NATIVE_BACKEND_ERROR;
                goto cleanup;
            }
            {
                const double inv_yn = 1.0 / yn;
                status = cublas_status(cublasDscal(blas, static_cast<int>(len),
                                                   &inv_yn, y_out, 1),
                                       "dominant Ritz normalize");
                if (status != TENET_NATIVE_SUCCESS) {
                    goto cleanup;
                }
            }
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d after normalize\n", block);
                std::fflush(stderr);
            }
            if (debug_enabled()) {
                std::fprintf(stderr, "DEBUG dominant_cuda block=%d ritz_resnorm=%.3e convergence_resnorm=%.3e\n",
                             block, ritz_resnorm, convergence_resnorm);
                std::fflush(stderr);
            }
            last_dominant_relres = convergence_resnorm;
            if (convergence_resnorm <= eig_tol || block + 1 == max_blocks) {
                status = TENET_NATIVE_SUCCESS;
                goto cleanup;
            }
            if (want_thick_restart &&
                build_compressed_restart_cuda(
                    blas, len, effective_k, m, V_dev, H.data(), effective_k + 1,
                    final_resnorm, thick_keep, breakdown_tol, coeff_dev,
                    restart_dev, restart_H.data(), &restart_cols,
                    &completed_cols) == TENET_NATIVE_SUCCESS) {
                have_compressed_restart = true;
                seed = nullptr;
                seed_cols = 0;
            } else {
                seed = y_out;
                seed_cols = 1;
                have_compressed_restart = false;
                restart_cols = 0;
                completed_cols = 0;
            }
        }
    }

cleanup:
    if (blas != nullptr) {
        cublasDestroy(blas);
    }
    return status;
}

int alc_to_ac_cuda(int64_t chi, long long phys, const double *AL, const double *C,
                  double *AC) {
    cublasHandle_t blas = nullptr;
    if (create_cublas_handle(&blas, "cublasCreate alc_to_ac") != TENET_NATIVE_SUCCESS) {
        set_error("allocation or cublasCreate failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    }
    const double one = 1.0;
    const double zero = 0.0;
    const long long tensor_ld = chi * phys;
    int status = TENET_NATIVE_SUCCESS;
    for (long long s = 0; s < phys; ++s) {
        const double *ALs = AL + chi * s;
        double *ACs = AC + chi * s;
        status = cublas_status(cublasDgemm(
            blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(chi),
            static_cast<int>(chi), static_cast<int>(chi), &one, ALs,
            static_cast<int>(tensor_ld), C, static_cast<int>(chi), &zero, ACs,
            static_cast<int>(tensor_ld)),
                               "alc_to_ac C = AL * C");
        if (status != TENET_NATIVE_SUCCESS) {
            cublasDestroy(blas);
            return status;
        }
    }
    cublasDestroy(blas);
    return TENET_NATIVE_SUCCESS;
}

void permute_m_for_ac_host(int64_t phys, const double *M, double *Mp) {
    for (long long i = 0; i < phys; ++i) {
        for (long long j = 0; j < phys; ++j) {
            for (long long k = 0; k < phys; ++k) {
                for (long long l = 0; l < phys; ++l) {
                    Mp[idx4(i, j, k, l, phys, phys, phys)] =
                        M[idx4(l, k, j, i, phys, phys, phys)];
                }
            }
        }
    }
}

int acc_to_alar_cuda(int64_t chi, int64_t phys, const double *AC,
                     const double *C, double *AL, double *AR, double *err) {
    const long long len2 = chi * chi;
    const long long len3 = chi * phys * chi;
    if (len2 <= 0 || len3 <= 0 || phys <= 0 || AL == nullptr || AR == nullptr ||
        err == nullptr) {
        set_error("invalid ACCtoALAR inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long tall = chi * phys;
    std::vector<double> AC_host(static_cast<std::size_t>(len3));
    std::vector<double> C_host(static_cast<std::size_t>(len2));
    int status = cuda_status(cudaMemcpy(AC_host.data(), AC,
                                        static_cast<std::size_t>(len3) * sizeof(double),
                                        cudaMemcpyDeviceToHost),
                             "copy AC to host");
    if (status != TENET_NATIVE_SUCCESS) return status;
    status = cuda_status(cudaMemcpy(C_host.data(), C,
                                    static_cast<std::size_t>(len2) * sizeof(double),
                                    cudaMemcpyDeviceToHost),
                         "copy C to host");
    if (status != TENET_NATIVE_SUCCESS) return status;

    std::vector<double> QAC(static_cast<std::size_t>(tall) *
                              static_cast<std::size_t>(chi));
    std::vector<double> RAC(static_cast<std::size_t>(chi) *
                              static_cast<std::size_t>(chi));
    std::vector<double> QC(static_cast<std::size_t>(chi) *
                              static_cast<std::size_t>(chi));
    std::vector<double> RC(static_cast<std::size_t>(chi) *
                              static_cast<std::size_t>(chi));
    if (!host_qrpos(tall, chi, AC_host.data(), QAC.data(), RAC.data()) ||
        !host_qrpos(chi, chi, C_host.data(), QC.data(), RC.data())) {
        return TENET_NATIVE_BACKEND_ERROR;
    }
    double errL = 0.0;
    for (long long i = 0; i < len2; ++i) {
        const double d = RAC[static_cast<std::size_t>(i)] -
                         RC[static_cast<std::size_t>(i)];
        errL += d * d;
    }
    errL = std::sqrt(errL);
    std::vector<double> AL_host(static_cast<std::size_t>(tall) *
                               static_cast<std::size_t>(chi),
                               0.0);
    host_dgemm('N', 'T', tall, chi, chi, 1.0, QAC.data(), tall, QC.data(), chi,
               0.0, AL_host.data(), tall);

    std::vector<double> ACtail(static_cast<std::size_t>(chi) *
                              static_cast<std::size_t>(tall));
    for (long long j = 0; j < tall; ++j) {
        for (long long i = 0; i < chi; ++i) {
            ACtail[idx2(i, j, chi)] = AC_host[static_cast<std::size_t>(i) +
                                             static_cast<std::size_t>(chi) * j];
        }
    }
    std::vector<double> LAC(static_cast<std::size_t>(chi) *
                            static_cast<std::size_t>(chi));
    std::vector<double> QLAC(static_cast<std::size_t>(chi) *
                             static_cast<std::size_t>(tall));
    std::vector<double> LC(static_cast<std::size_t>(chi) *
                           static_cast<std::size_t>(chi));
    std::vector<double> QLC(static_cast<std::size_t>(chi) *
                            static_cast<std::size_t>(chi));
    if (!host_lqpos(chi, tall, ACtail.data(), LAC.data(), QLAC.data()) ||
        !host_lqpos(chi, chi, C_host.data(), LC.data(), QLC.data())) {
        return TENET_NATIVE_BACKEND_ERROR;
    }
    double errR = 0.0;
    for (long long i = 0; i < len2; ++i) {
        const double d = LAC[static_cast<std::size_t>(i)] -
                         LC[static_cast<std::size_t>(i)];
        errR += d * d;
    }
    errR = std::sqrt(errR);
    std::vector<double> ARtail(static_cast<std::size_t>(chi) *
                              static_cast<std::size_t>(tall));
    host_dgemm('T', 'N', chi, tall, chi, 1.0, QLC.data(), chi,
               QLAC.data(), chi, 0.0, ARtail.data(), chi);

    status = cuda_status(cudaMemcpy(AL, AL_host.data(),
                                    static_cast<std::size_t>(tall) *
                                        static_cast<std::size_t>(chi) *
                                        sizeof(double),
                                    cudaMemcpyHostToDevice),
                        "copy AL back to device");
    if (status != TENET_NATIVE_SUCCESS) return status;
    status = cuda_status(cudaMemcpy(AR, ARtail.data(),
                                    static_cast<std::size_t>(tall) *
                                        static_cast<std::size_t>(chi) *
                                        sizeof(double),
                                    cudaMemcpyHostToDevice),
                        "copy AR back to device");
    if (status != TENET_NATIVE_SUCCESS) return status;
    *err = errL + errR;
    return TENET_NATIVE_SUCCESS;
}

} // namespace

extern "C" int tenet_native_abi_version(void) {
    return TENET_NATIVE_ABI_VERSION;
}

extern "C" const char *tenet_native_abi_version_string(void) {
    return TENET_NATIVE_ABI_VERSION_STRING;
}

extern "C" int tenet_native_krylov_abi_version(void) {
    return TENET_NATIVE_KRYLOV_ABI_VERSION;
}

extern "C" const char *tenet_native_krylov_abi_version_string(void) {
    return TENET_NATIVE_KRYLOV_ABI_VERSION_STRING;
}

extern "C" const char *tenet_native_status_string(int status) {
    switch (status) {
    case TENET_NATIVE_SUCCESS:
        return "success";
    case TENET_NATIVE_INVALID_VALUE:
        return "invalid value";
    case TENET_NATIVE_ALLOCATION_FAILED:
        return "allocation failed";
    case TENET_NATIVE_BACKEND_ERROR:
        return "backend error";
    default:
        return "unknown status";
    }
}

extern "C" const char *tenet_native_last_error(void) { return last_error; }

extern "C" int tenet_native_arnoldi_two_layer_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    double *V, int64_t ldv, double *H, int64_t ldh, double *beta,
    int64_t *m, double *final_resnorm) {
    const long long len = chi * chi;
    if (!check_tensors(chi, phys, Aup, Adn) ||
        !check_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
                      final_resnorm)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    return arnoldi_cuda(
        len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
        final_resnorm, [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            return two_layer_apply_gemm(blas, chi, phys, Aup, Adn, src, tmp, dst, transpose);
        });
}

extern "C" int tenet_native_arnoldi_projected_two_layer_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *rho, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    const long long len = chi * chi;
    if (!check_tensors(chi, phys, Aup, Adn) || rho == nullptr ||
        !check_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
                      final_resnorm)) {
        if (rho == nullptr) {
            set_error("null rho pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    return arnoldi_cuda(
        len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
        final_resnorm,
        [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            return projected_two_layer_apply_gemm(blas, chi, phys, Aup, Adn,
                                                  rho, src, tmp, dst,
                                                  transpose);
        });
}

extern "C" int tenet_native_arnoldi_qprojected_two_layer_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *rho, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    const long long len = chi * chi;
    if (!check_tensors(chi, phys, Aup, Adn) || rho == nullptr ||
        !check_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
                      final_resnorm)) {
        if (rho == nullptr) {
            set_error("null rho pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    static thread_local DeviceBuffer qwork_buffer;
    const std::size_t len_bytes = static_cast<std::size_t>(len) * sizeof(double);
    int status = ensure_device_buffer(qwork_buffer, len_bytes,
                                      "allocate qprojected Arnoldi qwork");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    double *qwork = qwork_buffer.ptr;
    return arnoldi_cuda(
        len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
        final_resnorm,
        [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            return qprojected_two_layer_apply_gemm(
                blas, chi, phys, Aup, Adn, rho, src, tmp, qwork, dst,
                transpose);
        });
}

extern "C" int tenet_native_two_layer_apply_batch_d_cuda(
    int64_t batch, int64_t chi, int64_t phys, const double *Aup,
    int64_t stride_Aup, const double *Adn, int64_t stride_Adn,
    const double *X, int64_t stride_X, int transpose, double *Y,
    int64_t stride_Y) {
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (!check_two_layer_batch_common(batch, chi, phys, Aup, stride_Aup, Adn,
                                      stride_Adn, X, stride_X, Y,
                                      stride_Y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long len = chi * chi;
    static thread_local DeviceBuffer tmp_buffer;
    const std::size_t tmp_bytes =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(batch) *
        sizeof(double);
    int status = ensure_device_buffer(tmp_buffer, tmp_bytes,
                                      "allocate batched two-layer tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    cublasHandle_t blas = nullptr;
    status = create_cublas_handle(&blas, "cublasCreate batched two-layer");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = two_layer_apply_batch_gemm(
        blas, batch, chi, phys, Aup, stride_Aup, Adn, stride_Adn, X, stride_X,
        tmp_buffer.ptr, Y, stride_Y, transpose);
    cublasDestroy(blas);
    return status;
}

extern "C" int tenet_native_raw_two_layer_apply_batch_d_cuda(
    int64_t batch, int64_t chi, int64_t phys, const double *Aup,
    int64_t stride_Aup, const double *Adn, int64_t stride_Adn,
    const double *X, int64_t stride_X, int transpose, double *Y,
    int64_t stride_Y) {
    return tenet_native_two_layer_apply_batch_d_cuda(
        batch, chi, phys, Aup, stride_Aup, Adn, stride_Adn, X, stride_X,
        transpose, Y, stride_Y);
}

extern "C" int tenet_native_projected_two_layer_apply_batch_d_cuda(
    int64_t batch, int64_t chi, int64_t phys, const double *Aup,
    int64_t stride_Aup, const double *Adn, int64_t stride_Adn,
    const double *rho, int64_t stride_rho, const double *X, int64_t stride_X,
    int transpose, double *Y, int64_t stride_Y) {
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (!check_projected_batch_common(batch, chi, phys, Aup, stride_Aup, Adn,
                                      stride_Adn, rho, stride_rho, X,
                                      stride_X, Y, stride_Y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long len = chi * chi;
    static thread_local DeviceBuffer tmp_buffer;
    const std::size_t tmp_bytes =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(batch) *
        sizeof(double);
    int status = ensure_device_buffer(tmp_buffer, tmp_bytes,
                                      "allocate batched projected tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    cublasHandle_t blas = nullptr;
    status = create_cublas_handle(&blas, "cublasCreate batched projected");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = projected_two_layer_apply_batch_gemm(
        blas, batch, chi, phys, Aup, stride_Aup, Adn, stride_Adn, rho,
        stride_rho, X, stride_X, tmp_buffer.ptr, Y, stride_Y, transpose);
    cublasDestroy(blas);
    return status;
}

extern "C" int tenet_native_qprojected_two_layer_apply_batch_d_cuda(
    int64_t batch, int64_t chi, int64_t phys, const double *Aup,
    int64_t stride_Aup, const double *Adn, int64_t stride_Adn,
    const double *rho, int64_t stride_rho, const double *X, int64_t stride_X,
    int transpose, double *Y, int64_t stride_Y) {
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (!check_projected_batch_common(batch, chi, phys, Aup, stride_Aup, Adn,
                                      stride_Adn, rho, stride_rho, X,
                                      stride_X, Y, stride_Y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long len = chi * chi;
    const std::size_t workspace_bytes =
        static_cast<std::size_t>(len) * static_cast<std::size_t>(batch) *
        sizeof(double);
    static thread_local DeviceBuffer tmp_buffer;
    static thread_local DeviceBuffer qwork_buffer;
    int status = ensure_device_buffer(tmp_buffer, workspace_bytes,
                                      "allocate batched qprojected tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = ensure_device_buffer(qwork_buffer, workspace_bytes,
                                  "allocate batched qprojected qwork");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    cublasHandle_t blas = nullptr;
    status = create_cublas_handle(&blas, "cublasCreate batched qprojected");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = qprojected_two_layer_apply_batch_gemm(
        blas, batch, chi, phys, Aup, stride_Aup, Adn, stride_Adn, rho,
        stride_rho, X, stride_X, tmp_buffer.ptr, qwork_buffer.ptr, Y,
        stride_Y, transpose);
    cublasDestroy(blas);
    return status;
}

extern "C" int tenet_native_arnoldi_three_layer_leg4_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *M, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long len = chi * phys * chi;
    if (!check_tensors(chi, phys, Aup, Adn) || M == nullptr ||
        !check_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
                      final_resnorm)) {
        if (M == nullptr) {
            set_error("null M pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    std::vector<double> M_host(static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys));
    int copy_status = cuda_status(
        cudaMemcpy(M_host.data(), M, M_host.size() * sizeof(double), cudaMemcpyDeviceToHost),
        "copy M to host");
    if (copy_status != TENET_NATIVE_SUCCESS) {
        return copy_status;
    }
    double *pair_work = nullptr;
    double *accum_work = nullptr;
    const std::size_t work_len =
        static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys) *
        static_cast<std::size_t>(chi) * static_cast<std::size_t>(chi);
    int status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&pair_work),
                   work_len * sizeof(double)),
        "allocate three-layer pair workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&accum_work),
                   work_len * sizeof(double)),
        "allocate three-layer accum workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        cudaFree(pair_work);
        return status;
    }
    auto cleanup = [&]() {
        cudaFree(pair_work);
        cudaFree(accum_work);
    };
    status = arnoldi_cuda(
        len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
        final_resnorm, [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            (void)tmp;
            return three_layer_apply_gemm_factored(
                blas, chi, phys, Aup, Adn, M_host.data(), src, pair_work,
                accum_work, dst, transpose);
        });
    cleanup();
    return status;
}

extern "C" int tenet_native_dominant_two_layer_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    double *y, double *lambda) {
    const long long len = chi * chi;
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_two_layer_export enter chi=%lld phys=%lld len=%lld max_k=%lld transpose=%d A=%p B=%p x=%p y=%p lambda=%p\n",
                     static_cast<long long>(chi), static_cast<long long>(phys),
                     len, static_cast<long long>(max_k), transpose,
                     static_cast<const void *>(Aup), static_cast<const void *>(Adn),
                     static_cast<const void *>(x0), static_cast<void *>(y),
                     static_cast<void *>(lambda));
        std::fflush(stderr);
    }
    if (!check_tensors(chi, phys, Aup, Adn) || x0 == nullptr ||
        y == nullptr || lambda == nullptr || max_k <= 0 || max_k > len ||
        breakdown_tol < 0.0 || len > INT_MAX) {
        set_error("invalid dominant two-layer CUDA inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    double *tmp = nullptr;
    double *fy = nullptr;
    cublasHandle_t blas = nullptr;
    auto cleanup = [&]() {
        if (blas != nullptr) {
            cublasDestroy(blas);
        }
        if (fy != nullptr) {
            cudaFree(fy);
        }
        if (tmp != nullptr) {
            cudaFree(tmp);
        }
    };
    int status = cuda_status(cudaMalloc(reinterpret_cast<void **>(&tmp),
                                        static_cast<std::size_t>(len) * sizeof(double)),
                             "allocate dominant two-layer tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = cuda_status(cudaMalloc(reinterpret_cast<void **>(&fy),
                                    static_cast<std::size_t>(len) * sizeof(double)),
                         "allocate dominant two-layer fy");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = create_cublas_handle(&blas, "cublasCreate dominant two-layer");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = dominant_arnoldi_vector_cuda(
        len, max_k, breakdown_tol, x0, y,
        [&](cublasHandle_t inner_blas, double *inner_tmp, const double *src,
            double *dst) {
            return two_layer_apply_gemm(inner_blas, chi, phys, Aup, Adn, src,
                                        inner_tmp, dst, transpose);
        });
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_two_layer_export after dominant status=%d\n",
                     status);
        std::fflush(stderr);
    }
    if (status == TENET_NATIVE_SUCCESS) {
        status = two_layer_apply_gemm(blas, chi, phys, Aup, Adn, y, tmp, fy,
                                      transpose);
    }
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_two_layer_export after final apply status=%d\n",
                     status);
        std::fflush(stderr);
    }
    if (status == TENET_NATIVE_SUCCESS) {
        double yy = 0.0;
        double yfy = 0.0;
        status = device_dot(len, y, y, &yy);
        if (status == TENET_NATIVE_SUCCESS) {
            status = device_dot(len, y, fy, &yfy);
        }
        if (status == TENET_NATIVE_SUCCESS) {
            *lambda = yy > 0.0 ? yfy / yy : 0.0;
        }
    }
    cleanup();
    return status;
}

extern "C" int tenet_native_dominant_three_layer_leg4_d_cuda(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *M, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *y, double *lambda) {
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const long long len = chi * phys * chi;
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_three_layer_export enter chi=%lld phys=%lld len=%lld max_k=%lld transpose=%d A=%p B=%p M=%p x=%p y=%p lambda=%p\n",
                     static_cast<long long>(chi), static_cast<long long>(phys),
                     len, static_cast<long long>(max_k), transpose,
                     static_cast<const void *>(Aup), static_cast<const void *>(Adn),
                     static_cast<const void *>(M), static_cast<const void *>(x0),
                     static_cast<void *>(y), static_cast<void *>(lambda));
        std::fflush(stderr);
    }
    if (!check_tensors(chi, phys, Aup, Adn) || M == nullptr ||
        x0 == nullptr || y == nullptr || lambda == nullptr || max_k <= 0 ||
        max_k > len || breakdown_tol < 0.0 || len > INT_MAX) {
        set_error("invalid dominant three-layer CUDA inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    std::vector<double> M_host(static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys));
    int status = cuda_status(
        cudaMemcpy(M_host.data(), M, M_host.size() * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "copy M to host");
    if (status != TENET_NATIVE_SUCCESS) {
        return status;
    }
    double *tmp = nullptr;
    double *fy = nullptr;
    double *pair_work = nullptr;
    double *accum_work = nullptr;
    cublasHandle_t blas = nullptr;
    auto cleanup = [&]() {
        if (blas != nullptr) {
            cublasDestroy(blas);
        }
        if (fy != nullptr) {
            cudaFree(fy);
        }
        if (tmp != nullptr) {
            cudaFree(tmp);
        }
        if (pair_work != nullptr) {
            cudaFree(pair_work);
        }
        if (accum_work != nullptr) {
            cudaFree(accum_work);
        }
    };
    const long long tmp_len = chi * chi;
    const std::size_t factored_work_len =
        static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys) *
        static_cast<std::size_t>(tmp_len);
    status = cuda_status(cudaMalloc(reinterpret_cast<void **>(&tmp),
                                    static_cast<std::size_t>(tmp_len) * sizeof(double)),
                         "allocate dominant three-layer tmp");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = cuda_status(cudaMalloc(reinterpret_cast<void **>(&fy),
                                    static_cast<std::size_t>(len) * sizeof(double)),
                         "allocate dominant three-layer fy");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&pair_work),
                   factored_work_len * sizeof(double)),
        "allocate dominant three-layer pair workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = cuda_status(
        cudaMalloc(reinterpret_cast<void **>(&accum_work),
                   factored_work_len * sizeof(double)),
        "allocate dominant three-layer accum workspace");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = create_cublas_handle(&blas, "cublasCreate dominant three-layer");
    if (status != TENET_NATIVE_SUCCESS) {
        cleanup();
        return status;
    }
    status = dominant_arnoldi_vector_cuda(
        len, max_k, breakdown_tol, x0, y,
        [&](cublasHandle_t inner_blas, double *inner_tmp, const double *src,
            double *dst) {
            (void)inner_tmp;
            return three_layer_apply_gemm_factored(
                inner_blas, chi, phys, Aup, Adn, M_host.data(), src,
                pair_work, accum_work, dst, transpose);
        });
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_three_layer_export after dominant status=%d\n",
                     status);
        std::fflush(stderr);
    }
    if (status == TENET_NATIVE_SUCCESS) {
        status = three_layer_apply_gemm_factored(
            blas, chi, phys, Aup, Adn, M_host.data(), y, pair_work,
            accum_work, fy, transpose);
    }
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG dominant_three_layer_export after final apply status=%d\n",
                     status);
        std::fflush(stderr);
    }
    if (status == TENET_NATIVE_SUCCESS) {
        double yy = 0.0;
        double yfy = 0.0;
        status = device_dot(len, y, y, &yy);
        if (status == TENET_NATIVE_SUCCESS) {
            status = device_dot(len, y, fy, &yfy);
        }
        if (status == TENET_NATIVE_SUCCESS) {
            *lambda = yy > 0.0 ? yfy / yy : 0.0;
        }
    }
    cleanup();
    return status;
}

int tenet_native_ising_vumps_step_impl_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double *err, int check_residual, double residual_tol) {
    if (debug_enabled()) {
        std::fprintf(stderr,
                     "DEBUG ising_step_cuda enter chi=%lld phys=%lld max_k=%lld "
                     "breakdown_tol=%.3e check=%d residual_tol=%.3e "
                     "M=%p AL=%p AR=%p C=%p FL=%p FR=%p err=%p\n",
                     static_cast<long long>(chi), static_cast<long long>(phys),
                     static_cast<long long>(max_k), breakdown_tol,
                     check_residual, residual_tol, static_cast<const void *>(M),
                     static_cast<void *>(AL), static_cast<void *>(AR),
                     static_cast<void *>(C), static_cast<void *>(FL),
                     static_cast<void *>(FR), static_cast<void *>(err));
        std::fflush(stderr);
    }
    const long long len2 = chi * chi;
    const long long len3 = chi * phys * chi;
    if (!check_tensors(chi, phys, AL, AR) || M == nullptr || C == nullptr ||
        FL == nullptr || FR == nullptr || err == nullptr || max_k <= 0 ||
        breakdown_tol < 0.0 || len3 > INT_MAX || len2 > INT_MAX ||
        (check_residual && residual_tol < 0.0)) {
        if (M == nullptr) {
            set_error("null M pointer");
        } else if (C == nullptr) {
            set_error("null C pointer");
        } else if (FL == nullptr) {
            set_error("null FL pointer");
        } else if (FR == nullptr) {
            set_error("null FR pointer");
        } else if (err == nullptr) {
            set_error("null err pointer");
        } else {
            set_error("invalid native Ising VUMPS step dimensions");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (max_k > std::max(len2, len3)) {
        set_error("invalid native Ising VUMPS step dimensions");
        return TENET_NATIVE_INVALID_VALUE;
    }
    double *AC = nullptr;
    double *FL_new = nullptr;
    double *FR_new = nullptr;
    double *C_new = nullptr;
    double *pair_work = nullptr;
    double *accum_work = nullptr;
    int status = TENET_NATIVE_SUCCESS;
    std::vector<double> M_host(static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys));
    int copy_status = cuda_status(
        cudaMemcpy(M_host.data(), M, M_host.size() * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "copy M to host");
    if (copy_status != TENET_NATIVE_SUCCESS) {
        return copy_status;
    }
    std::vector<double> Mp(M_host.size());
    permute_m_for_ac_host(phys, M_host.data(), Mp.data());
    const std::size_t factored_work_len =
        static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys) *
        static_cast<std::size_t>(len2);
    const int alloc_status =
        (cudaMalloc(reinterpret_cast<void **>(&AC), static_cast<std::size_t>(len3) * sizeof(double)) != cudaSuccess ||
         cudaMalloc(reinterpret_cast<void **>(&FL_new), static_cast<std::size_t>(len3) * sizeof(double)) != cudaSuccess ||
         cudaMalloc(reinterpret_cast<void **>(&FR_new), static_cast<std::size_t>(len3) * sizeof(double)) != cudaSuccess ||
         cudaMalloc(reinterpret_cast<void **>(&C_new), static_cast<std::size_t>(len2) * sizeof(double)) != cudaSuccess ||
         cudaMalloc(reinterpret_cast<void **>(&pair_work), factored_work_len * sizeof(double)) != cudaSuccess ||
         cudaMalloc(reinterpret_cast<void **>(&accum_work), factored_work_len * sizeof(double)) != cudaSuccess);
    if (alloc_status != 0) {
        cudaFree(AC);
        cudaFree(FL_new);
        cudaFree(FR_new);
        cudaFree(C_new);
        cudaFree(pair_work);
        cudaFree(accum_work);
        set_error("allocation or cublasCreate failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    }
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG ising_step_cuda before alc_to_ac AC=%p\n",
                     static_cast<void *>(AC));
        std::fflush(stderr);
    }
    status = alc_to_ac_cuda(chi, phys, AL, C, AC);
    if (debug_enabled()) {
        std::fprintf(stderr, "DEBUG ising_step_cuda after alc_to_ac status=%d\n",
                     status);
        std::fflush(stderr);
    }
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status =
        dominant_arnoldi_vector_cuda(
            len3, std::min(max_k, static_cast<int64_t>(len3)), breakdown_tol, FL, FL_new,
            [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
                (void)tmp;
                return three_layer_apply_gemm_factored(
                    blas, chi, phys, AL, AL, M_host.data(), src, pair_work,
                    accum_work, dst, 0);
            });
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    if (check_residual &&
        !check_last_dominant_residual("FLmap", residual_tol)) {
        status = TENET_NATIVE_BACKEND_ERROR;
        goto cleanup;
    }
    status =
        dominant_arnoldi_vector_cuda(
            len3, std::min(max_k, static_cast<int64_t>(len3)), breakdown_tol, FR, FR_new,
            [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
                (void)tmp;
                return three_layer_apply_gemm_factored(
                    blas, chi, phys, AR, AR, M_host.data(), src, pair_work,
                    accum_work, dst, 1);
            });
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    if (check_residual &&
        !check_last_dominant_residual("FRmap", residual_tol)) {
        status = TENET_NATIVE_BACKEND_ERROR;
        goto cleanup;
    }
    status = dominant_arnoldi_vector_cuda(
        len3, std::min(max_k, static_cast<int64_t>(len3)), breakdown_tol, AC, AC,
        [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            (void)tmp;
            return three_layer_apply_gemm_factored(
                blas, chi, phys, FL_new, FR_new, Mp.data(), src, pair_work,
                accum_work, dst, 0);
        });
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    if (check_residual &&
        !check_last_dominant_residual("ACmap", residual_tol)) {
        status = TENET_NATIVE_BACKEND_ERROR;
        goto cleanup;
    }
    status = dominant_arnoldi_vector_cuda(
        len2, std::min(max_k, static_cast<int64_t>(len2)), breakdown_tol, C, C_new,
        [&](cublasHandle_t blas, double *tmp, const double *src, double *dst) {
            return two_layer_apply_gemm(blas, chi, phys, FL_new, FR_new, src, tmp,
                                       dst, 0);
        });
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    if (check_residual &&
        !check_last_dominant_residual("Cmap", residual_tol)) {
        status = TENET_NATIVE_BACKEND_ERROR;
        goto cleanup;
    }
    status = acc_to_alar_cuda(chi, phys, AC, C_new, AL, AR, err);
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = cuda_status(
        cudaMemcpy(C, C_new, static_cast<std::size_t>(len2) * sizeof(double),
                   cudaMemcpyDeviceToDevice),
                        "copy C back to device for step");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = cuda_status(cudaMemcpy(FL, FL_new, static_cast<std::size_t>(len3) * sizeof(double),
                                   cudaMemcpyDeviceToDevice),
                        "copy FL back to device");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
    status = cuda_status(cudaMemcpy(FR, FR_new, static_cast<std::size_t>(len3) * sizeof(double),
                                   cudaMemcpyDeviceToDevice),
                        "copy FR back to device");
    if (status != TENET_NATIVE_SUCCESS) {
        goto cleanup;
    }
cleanup:
    if (AC != nullptr) {
        cudaFree(AC);
    }
    if (FL_new != nullptr) {
        cudaFree(FL_new);
    }
    if (FR_new != nullptr) {
        cudaFree(FR_new);
    }
    if (C_new != nullptr) {
        cudaFree(C_new);
    }
    if (pair_work != nullptr) {
        cudaFree(pair_work);
    }
    if (accum_work != nullptr) {
        cudaFree(accum_work);
    }
    return status;
}

extern "C" int tenet_native_ising_vumps_step_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double *err) {
    return tenet_native_ising_vumps_step_impl_d_cuda(
        chi, phys, M, AL, AR, C, FL, FR, max_k, breakdown_tol, err, 0, 0.0);
}

extern "C" int tenet_native_ising_vumps_step_checked_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double residual_tol, double *err) {
    return tenet_native_ising_vumps_step_impl_d_cuda(
        chi, phys, M, AL, AR, C, FL, FR, max_k, breakdown_tol, err, 1,
        residual_tol);
}

int tenet_native_ising_vumps_run_impl_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double *err, int64_t *iterations, int *converged, int check_residual,
    double residual_tol) {
    if (debug_enabled()) {
        std::fprintf(stderr,
                     "DEBUG ising_run_cuda enter chi=%lld phys=%lld max_k=%lld "
                     "breakdown_tol=%.3e tol=%.3e miniter=%lld maxiter=%lld "
                     "check=%d residual_tol=%.3e err=%p iterations=%p converged=%p\n",
                     static_cast<long long>(chi), static_cast<long long>(phys),
                     static_cast<long long>(arnoldi_max_k), breakdown_tol, tol,
                     static_cast<long long>(miniter), static_cast<long long>(maxiter),
                     check_residual, residual_tol, static_cast<void *>(err),
                     static_cast<void *>(iterations), static_cast<void *>(converged));
        std::fflush(stderr);
    }
    if (err == nullptr || iterations == nullptr || converged == nullptr) {
        set_error("null pointer in native CUDA Ising VUMPS run");
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (miniter < 0 || maxiter < 0 || miniter > maxiter || tol < 0.0 ||
        (check_residual && residual_tol < 0.0)) {
        set_error("invalid native CUDA Ising VUMPS run iteration controls");
        return TENET_NATIVE_INVALID_VALUE;
    }
    *err = 0.0;
    *iterations = 0;
    *converged = maxiter == 0 ? 1 : 0;
    for (int64_t iter = 1; iter <= maxiter; ++iter) {
        const int status = tenet_native_ising_vumps_step_impl_d_cuda(
            chi, phys, M, AL, AR, C, FL, FR, arnoldi_max_k, breakdown_tol,
            err, check_residual, residual_tol);
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        *iterations = iter;
        if (*err < tol && iter >= miniter) {
            *converged = 1;
            break;
        }
    }
    return TENET_NATIVE_SUCCESS;
}

extern "C" int tenet_native_ising_vumps_run_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double *err, int64_t *iterations, int *converged) {
    return tenet_native_ising_vumps_run_impl_d_cuda(
        chi, phys, M, AL, AR, C, FL, FR, arnoldi_max_k, breakdown_tol, tol,
        miniter, maxiter, err, iterations, converged, 0, 0.0);
}

extern "C" int tenet_native_ising_vumps_run_checked_d_cuda(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double residual_tol, double *err, int64_t *iterations, int *converged) {
    return tenet_native_ising_vumps_run_impl_d_cuda(
        chi, phys, M, AL, AR, C, FL, FR, arnoldi_max_k, breakdown_tol, tol,
        miniter, maxiter, err, iterations, converged, 1, residual_tol);
}
