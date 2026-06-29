#include "tenet_native_arnoldi.h"
#include "krylov_core.hpp"
#include "tenet_native_shared_two_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <numeric>
#include <complex>
#include <vector>

#if defined(__APPLE__) && defined(TENET_NATIVE_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif __has_include(<cblas.h>)
#include <cblas.h>
#define TENET_NATIVE_HAS_CBLAS 1
#else
#define TENET_NATIVE_USE_FORTRAN_BLAS 1
extern "C" {
void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
void dgemm_(const char *transa, const char *transb, const int *m,
            const int *n, const int *k, const double *alpha, const double *a,
            const int *lda, const double *b, const int *ldb,
            const double *beta, double *c, const int *ldc);
void daxpy_(const int *n, const double *alpha, const double *x,
            const int *incx, double *y, const int *incy);
}
#endif

#if !(defined(__APPLE__) && defined(TENET_NATIVE_USE_ACCELERATE))
extern "C" {
#if defined(TENET_NATIVE_USE_BLAS64)
void dgemv_64_(char *trans, int64_t *m, int64_t *n, double *alpha,
               const double *a, int64_t *lda, const double *x, int64_t *incx,
               double *beta, double *y, int64_t *incy);
void dgemm_64_(char *transa, char *transb, int64_t *m, int64_t *n,
               int64_t *k, double *alpha, const double *a, int64_t *lda,
               const double *b, int64_t *ldb, double *beta, double *c,
               int64_t *ldc);
void daxpy_64_(int64_t *n, double *alpha, const double *x, int64_t *incx,
               double *y, int64_t *incy);
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
void dgeqrf_64_(int64_t *m, int64_t *n, double *a, int64_t *lda, double *tau,
                double *work, int64_t *lwork, int64_t *info);
void dorgqr_64_(int64_t *m, int64_t *n, int64_t *k, double *a, int64_t *lda,
                double *tau, double *work, int64_t *lwork, int64_t *info);
#else
void dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *lda,
            double *wr, double *wi, double *vl, int *ldvl, double *vr,
            int *ldvr, double *work, int *lwork, int *info);
void dgeevx_(char *balanc, char *jobvl, char *jobvr, char *sense, int *n,
             double *a, int *lda, double *wr, double *wi, double *vl,
             int *ldvl, double *vr, int *ldvr, int *ilo, int *ihi,
             double *scale, double *abnrm, double *rconde, double *rcondv,
             double *work, int *lwork, int *iwork, int *info);
typedef int (*dgees_select_fn)(double *, double *);
void dgees_(char *jobvs, char *sort, dgees_select_fn select, int *n,
            double *a, int *lda, int *sdim, double *wr, double *wi,
            double *vs, int *ldvs, double *work, int *lwork, int *bwork,
            int *info);
void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work,
             int *lwork, int *info);
void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau,
             double *work, int *lwork, int *info);
#endif
}
#endif

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
                  "native Arnoldi %s residual %.6e exceeds tolerance %.6e",
                  context, last_dominant_relres, residual_tol);
    return false;
}

using SteadyClock = std::chrono::steady_clock;

bool profile_enabled() {
    const char *env = std::getenv("TENET_NATIVE_PROFILE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

double seconds_since(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

void profile_phase(const char *phase, int64_t chi, int64_t phys,
                   int64_t max_k, double seconds) {
    if (!profile_enabled()) {
        return;
    }
    std::fprintf(stderr,
                 "TENET_NATIVE_PROFILE phase=%s chi=%lld phys=%lld max_k=%lld "
                 "seconds=%.9g\n",
                 phase, static_cast<long long>(chi),
                 static_cast<long long>(phys), static_cast<long long>(max_k),
                 seconds);
    std::fflush(stderr);
}

thread_local std::vector<double> schur_select_wr;
thread_local std::vector<double> schur_select_wi;

int schur_select_callback(double *wr, double *wi) {
    const double ar = wr == nullptr ? 0.0 : *wr;
    const double ai = wi == nullptr ? 0.0 : *wi;
    for (std::size_t i = 0; i < schur_select_wr.size(); ++i) {
        const double sr = schur_select_wr[i];
        const double si = schur_select_wi[i];
        const double scale =
            std::max(1.0, std::max(std::hypot(ar, ai), std::hypot(sr, si)));
        if (std::abs(ar - sr) <= 1e-7 * scale &&
            std::abs(ai - si) <= 1e-7 * scale) {
            return 1;
        }
    }
    return 0;
}

void profile_arnoldi(int64_t len, int64_t max_k, int64_t m,
                     int64_t applied_cols,
                     double apply_seconds, double orthog_seconds,
                     double norm_seconds, double final_resnorm) {
    if (!profile_enabled()) {
        return;
    }
    std::fprintf(stderr,
                 "TENET_NATIVE_PROFILE kind=arnoldi len=%lld max_k=%lld "
                 "m=%lld applied=%lld apply_seconds=%.9g orthog_seconds=%.9g "
                 "norm_seconds=%.9g final_resnorm=%.9g\n",
                 static_cast<long long>(len), static_cast<long long>(max_k),
                 static_cast<long long>(m), static_cast<long long>(applied_cols),
                 apply_seconds, orthog_seconds, norm_seconds, final_resnorm);
    std::fflush(stderr);
}

inline int64_t idx2(int64_t i, int64_t j, int64_t n1) {
    return i + n1 * j;
}

inline int64_t idx4(int64_t i, int64_t j, int64_t k, int64_t l,
                    int64_t n1, int64_t n2, int64_t n3) {
    return i + n1 * (j + n2 * (k + n3 * l));
}

void blas_gemm(char transa, char transb, int64_t m64, int64_t n64,
               int64_t k64, double alpha, const double *A, int64_t lda64,
               const double *B, int64_t ldb64, double beta, double *C,
               int64_t ldc64);

struct Leg3ColMajorTwoLayerLayout {
    explicit Leg3ColMajorTwoLayerLayout(int64_t chi_in, int64_t phys_in)
        : chi(chi_in), phys(phys_in), tensor_ld(chi_in * phys_in) {}

    int64_t dim() const { return chi; }
    int64_t matrix_len() const { return chi * chi; }

    const double *slice(const double *base, int64_t s) const {
        return base + chi * s;
    }

    void apply_adjoint_slice(const double *A, const double *B, const double *x,
                             double *tmp, double *y) const {
        blas_gemm('N', 'N', chi, chi, chi, 1.0, x, chi, B, tensor_ld, 0.0,
                  tmp, chi);
        blas_gemm('T', 'N', chi, chi, chi, 1.0, A, tensor_ld, tmp, chi, 1.0,
                  y, chi);
    }

    void apply_forward_slice(const double *A, const double *B, const double *x,
                             double *tmp, double *y) const {
        blas_gemm('N', 'T', chi, chi, chi, 1.0, x, chi, B, tensor_ld, 0.0,
                  tmp, chi);
        blas_gemm('N', 'N', chi, chi, chi, 1.0, A, tensor_ld, tmp, chi, 1.0,
                  y, chi);
    }

    int64_t chi;
    int64_t phys;
    int64_t tensor_ld;
};

bool check_arnoldi_common(int64_t len, int64_t max_k, double breakdown_tol,
                          const double *x0, double *V, int64_t ldv,
                          double *H, int64_t ldh, double *beta, int64_t *m,
                          double *final_resnorm) {
    if (len <= 0 || max_k < 0 || max_k > len || breakdown_tol < 0.0) {
        set_error("invalid Arnoldi dimensions or tolerance");
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
    if (len > INT_MAX || max_k > INT_MAX || ldv > INT_MAX || ldh > INT_MAX) {
        set_error("native CPU BLAS Arnoldi dimensions exceed BLAS int range");
        return false;
    }
    set_error("success");
    return true;
}

bool check_tensor_common(int64_t chi, int64_t phys, const double *Aup,
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

bool check_raw_two_layer_inputs(int64_t chi, int64_t phys, const double *Aup,
                               const double *Adn, const double *x,
                               int transpose, double *y) {
    if (!check_tensor_common(chi, phys, Aup, Adn) || x == nullptr ||
        y == nullptr) {
        set_error("null raw two-layer tensor pointer");
        return false;
    }
    if (transpose != 0 && transpose != 1) {
        set_error("invalid transpose flag");
        return false;
    }
    return true;
}

void rowmajor_transfer_apply_forward_slice(int64_t d, const double *A,
                                          const double *x, double *tmp,
                                          double *y) {
    std::fill(tmp, tmp + d * d, 0.0);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                acc += A[i * d + k] * x[k * d + j];
            }
            tmp[i * d + j] = acc;
        }
    }
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                acc += tmp[i * d + k] * A[j * d + k];
            }
            y[i * d + j] += acc;
        }
    }
}

void rowmajor_transfer_apply_adjoint_slice(int64_t d, const double *A,
                                          const double *x, double *tmp,
                                          double *y) {
    std::fill(tmp, tmp + d * d, 0.0);
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                acc += A[k * d + i] * x[k * d + j];
            }
            tmp[i * d + j] = acc;
        }
    }
    for (int64_t i = 0; i < d; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double acc = 0.0;
            for (int64_t k = 0; k < d; ++k) {
                acc += tmp[i * d + k] * A[k * d + j];
            }
            y[i * d + j] += acc;
        }
    }
}

bool check_raw_rowmajor_inputs(int64_t d, int64_t D, const double *W,
                              const double *x, const double *y) {
    if (W == nullptr || x == nullptr || y == nullptr) {
        set_error("null row-major tensor pointer");
        return false;
    }
    if (d <= 0 || D <= 0) {
        set_error("invalid row-major dimensions");
        return false;
    }
    return true;
}

bool check_raw_rowmajor_transfer_op_inputs(int64_t d, int64_t D,
                                          const double *W, const double *O,
                                          const double *x, const double *y) {
    if (!check_raw_rowmajor_inputs(d, D, W, x, y)) {
        return false;
    }
    if (O == nullptr) {
        set_error("null row-major O pointer");
        return false;
    }
    return true;
}

double dot(int64_t n, const double *x, const double *y) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }
    return acc;
}

double norm2(int64_t n, const double *x) { return std::sqrt(dot(n, x, x)); }

void scal(int64_t n, double alpha, double *x) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
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

int arnoldi_reorthog_passes() {
    int passes = 1;
    if (const char *env = std::getenv("TENET_NATIVE_ARNOLDI_REORTHOG")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) {
            passes = parsed;
        }
    }
    return std::min(std::max(passes, 1), 4);
}

void blas_gemv(char trans, int64_t m64, int64_t n64, double alpha,
               const double *A, int64_t lda64, const double *x, double beta,
               double *y) {
#if defined(TENET_NATIVE_USE_ACCELERATE) || defined(TENET_NATIVE_HAS_CBLAS)
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int lda = static_cast<int>(lda64);
    const int inc = 1;
    const CBLAS_TRANSPOSE op = trans == 'T' ? CblasTrans : CblasNoTrans;
    cblas_dgemv(CblasColMajor, op, m, n, alpha, A, lda, x, inc, beta, y, inc);
#else
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t m_blas = m64;
    int64_t n_blas = n64;
    int64_t lda_blas = lda64;
    int64_t inc_blas = 1;
    dgemv_64_(&trans, &m_blas, &n_blas, &alpha, A, &lda_blas, x,
              &inc_blas, &beta, y, &inc_blas);
#else
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int lda = static_cast<int>(lda64);
    const int inc = 1;
    dgemv_(&trans, &m, &n, &alpha, A, &lda, x, &inc, &beta, y, &inc);
#endif
#endif
}

void blas_gemm(char transa, char transb, int64_t m64, int64_t n64,
               int64_t k64, double alpha, const double *A, int64_t lda64,
               const double *B, int64_t ldb64, double beta, double *C,
               int64_t ldc64) {
#if defined(TENET_NATIVE_USE_ACCELERATE) || defined(TENET_NATIVE_HAS_CBLAS)
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const int lda = static_cast<int>(lda64);
    const int ldb = static_cast<int>(ldb64);
    const int ldc = static_cast<int>(ldc64);
    const CBLAS_TRANSPOSE opa = transa == 'T' ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE opb = transb == 'T' ? CblasTrans : CblasNoTrans;
    cblas_dgemm(CblasColMajor, opa, opb, m, n, k, alpha, A, lda, B, ldb,
                beta, C, ldc);
#else
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t m_blas = m64;
    int64_t n_blas = n64;
    int64_t k_blas = k64;
    int64_t lda_blas = lda64;
    int64_t ldb_blas = ldb64;
    int64_t ldc_blas = ldc64;
    dgemm_64_(&transa, &transb, &m_blas, &n_blas, &k_blas, &alpha, A,
              &lda_blas, B, &ldb_blas, &beta, C, &ldc_blas);
#else
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const int lda = static_cast<int>(lda64);
    const int ldb = static_cast<int>(ldb64);
    const int ldc = static_cast<int>(ldc64);
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
           &ldc);
#endif
#endif
}

void blas_axpy(int64_t n64, double alpha, const double *x, double *y) {
#if defined(TENET_NATIVE_USE_ACCELERATE) || defined(TENET_NATIVE_HAS_CBLAS)
    const int n = static_cast<int>(n64);
    const int inc = 1;
    cblas_daxpy(n, alpha, x, inc, y, inc);
#else
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t n_blas = n64;
    int64_t inc_blas = 1;
    daxpy_64_(&n_blas, &alpha, x, &inc_blas, y, &inc_blas);
#else
    const int n = static_cast<int>(n64);
    const int inc = 1;
    daxpy_(&n, &alpha, x, &inc, y, &inc);
#endif
#endif
}

bool lapack_geev(int n, std::vector<double> &A, std::vector<double> &wr,
                 std::vector<double> &wi, std::vector<double> &vr) {
    char balanc = 'B';
    char jobvl = 'N';
    char jobvr = 'V';
    char sense = 'N';
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t n_lapack = n;
    int64_t lda = n;
    int64_t ldvl = 1;
    int64_t ldvr = n;
    int64_t ilo = 0;
    int64_t ihi = 0;
    int64_t info = 0;
    int64_t lwork = -1;
#else
    int lda = n;
    int ldvl = 1;
    int ldvr = n;
    int ilo = 0;
    int ihi = 0;
    int info = 0;
    int lwork = -1;
#endif
    double vl_dummy = 0.0;
    double abnrm = 0.0;
    double work_query = 0.0;
    std::vector<double> scale(static_cast<std::size_t>(n));
    std::vector<double> rconde(static_cast<std::size_t>(n));
    std::vector<double> rcondv(static_cast<std::size_t>(n));
#if defined(TENET_NATIVE_USE_BLAS64)
    std::vector<int64_t> iwork(static_cast<std::size_t>(2 * n));
    dgeevx_64_(&balanc, &jobvl, &jobvr, &sense, &n_lapack, A.data(), &lda,
               wr.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr,
               &ilo, &ihi, scale.data(), &abnrm, rconde.data(),
               rcondv.data(), &work_query, &lwork, iwork.data(), &info);
#else
    std::vector<int> iwork(static_cast<std::size_t>(2 * n));
    dgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, A.data(), &lda, wr.data(),
            wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &ilo, &ihi,
            scale.data(), &abnrm, rconde.data(), rcondv.data(), &work_query,
            &lwork, iwork.data(), &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgeevx workspace query failed");
        return false;
    }
    lwork = std::max<decltype(lwork)>(1, static_cast<decltype(lwork)>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
#if defined(TENET_NATIVE_USE_BLAS64)
    dgeevx_64_(&balanc, &jobvl, &jobvr, &sense, &n_lapack, A.data(), &lda,
               wr.data(), wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr,
               &ilo, &ihi, scale.data(), &abnrm, rconde.data(),
               rcondv.data(), work.data(), &lwork, iwork.data(), &info);
#else
    dgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, A.data(), &lda, wr.data(),
            wi.data(), &vl_dummy, &ldvl, vr.data(), &ldvr, &ilo, &ihi,
            scale.data(), &abnrm, rconde.data(), rcondv.data(), work.data(),
            &lwork, iwork.data(), &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgeevx failed");
        return false;
    }
    return true;
}

bool choose_schur_restart_values(const std::vector<double> &wr,
                                 const std::vector<double> &wi,
                                 const std::vector<int> &order, int keep) {
    const int n = static_cast<int>(wr.size());
    schur_select_wr.clear();
    schur_select_wi.clear();
    std::vector<char> chosen(static_cast<std::size_t>(n), 0);
    const auto same_value = [&](int a, int b) {
        const double scale =
            std::max(1.0, std::max(std::hypot(wr[a], wi[a]),
                                   std::hypot(wr[b], wi[b])));
        return std::abs(wr[a] - wr[b]) <= 1e-8 * scale &&
               std::abs(wi[a] + wi[b]) <= 1e-8 * scale;
    };
    for (const int idx : order) {
        if (static_cast<int>(schur_select_wr.size()) >= keep) {
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
            if (static_cast<int>(schur_select_wr.size()) + 2 > keep &&
                !schur_select_wr.empty()) {
                break;
            }
            chosen[static_cast<std::size_t>(idx)] = 1;
            chosen[static_cast<std::size_t>(partner)] = 1;
            schur_select_wr.push_back(wr[idx]);
            schur_select_wi.push_back(wi[idx]);
            schur_select_wr.push_back(wr[partner]);
            schur_select_wi.push_back(wi[partner]);
        } else {
            chosen[static_cast<std::size_t>(idx)] = 1;
            schur_select_wr.push_back(wr[idx]);
            schur_select_wi.push_back(wi[idx]);
        }
    }
    return !schur_select_wr.empty();
}

bool lapack_schur_selected(int n, std::vector<double> &A,
                           std::vector<double> &wr, std::vector<double> &wi,
                           std::vector<double> &vs, int *sdim_out) {
    char jobvs = 'V';
    char sort = 'S';
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t n_lapack = n;
    int64_t lda = n;
    int64_t ldvs = n;
    int64_t sdim = 0;
    int64_t info = 0;
    int64_t lwork = -1;
    std::vector<int64_t> bwork(static_cast<std::size_t>(n));
#else
    int lda = n;
    int ldvs = n;
    int sdim = 0;
    int info = 0;
    int lwork = -1;
    std::vector<int> bwork(static_cast<std::size_t>(n));
#endif
    double work_query = 0.0;
#if defined(TENET_NATIVE_USE_BLAS64)
    dgees_64_(&jobvs, &sort, schur_select_callback, &n_lapack, A.data(),
              &lda, &sdim, wr.data(), wi.data(), vs.data(), &ldvs,
              &work_query, &lwork, bwork.data(), &info);
#else
    dgees_(&jobvs, &sort, schur_select_callback, &n, A.data(), &lda, &sdim,
           wr.data(), wi.data(), vs.data(), &ldvs, &work_query, &lwork,
           bwork.data(), &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgees workspace query failed");
        return false;
    }
    lwork = std::max<decltype(lwork)>(1, static_cast<decltype(lwork)>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
#if defined(TENET_NATIVE_USE_BLAS64)
    dgees_64_(&jobvs, &sort, schur_select_callback, &n_lapack, A.data(),
              &lda, &sdim, wr.data(), wi.data(), vs.data(), &ldvs,
              work.data(), &lwork, bwork.data(), &info);
#else
    dgees_(&jobvs, &sort, schur_select_callback, &n, A.data(), &lda, &sdim,
           wr.data(), wi.data(), vs.data(), &ldvs, work.data(), &lwork,
           bwork.data(), &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgees failed");
        return false;
    }
    *sdim_out = static_cast<int>(sdim);
    return *sdim_out > 0;
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
            Hm[idx2(row, col, n)] =
                H[idx2(row, col, static_cast<int64_t>(ldh))];
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
                acc += Hm[idx2(row, col, n)] *
                       coeff[static_cast<std::size_t>(col)];
            }
            tmp[static_cast<std::size_t>(row)] = acc;
        }
        const double nrm = norm2(n, tmp.data());
        if (!(nrm > 0.0) || !std::isfinite(nrm)) {
            set_error("dominant Ritz vector search failed");
            return false;
        }
        const double inv_nrm = 1.0 / nrm;
        for (int i = 0; i < n; ++i) {
            tmp[static_cast<std::size_t>(i)] *= inv_nrm;
        }
        for (int i = 0; i < n; ++i) {
            work[static_cast<std::size_t>(i)] = 0.0;
            for (int col = 0; col < n; ++col) {
                work[static_cast<std::size_t>(i)] +=
                    Hm[idx2(i, col, n)] * tmp[static_cast<std::size_t>(col)];
            }
        }
        lambda = dot(n, tmp.data(), work.data());
        double residual = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d =
                work[static_cast<std::size_t>(i)] - lambda * tmp[static_cast<std::size_t>(i)];
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

bool lapack_qrpos(int64_t m64, int64_t n64, const double *A, double *Q,
                  double *R) {
    if (m64 < n64 || m64 <= 0 || n64 <= 0 || m64 > INT_MAX ||
        n64 > INT_MAX) {
        set_error("invalid qrpos dimensions");
        return false;
    }
    int m = static_cast<int>(m64);
    int n = static_cast<int>(n64);
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t m_lapack = m64;
    int64_t n_lapack = n64;
    int64_t lda_lapack = m64;
#else
    int lda = m;
#endif
    std::vector<double> work_A(static_cast<std::size_t>(m) *
                               static_cast<std::size_t>(n));
    std::copy(A, A + static_cast<std::size_t>(m) * static_cast<std::size_t>(n),
              work_A.begin());
    std::vector<double> tau(static_cast<std::size_t>(n));
#if defined(TENET_NATIVE_USE_BLAS64)
    int64_t info = 0;
    int64_t lwork = -1;
#else
    int info = 0;
    int lwork = -1;
#endif
    double work_query = 0.0;
#if defined(TENET_NATIVE_USE_BLAS64)
    dgeqrf_64_(&m_lapack, &n_lapack, work_A.data(), &lda_lapack, tau.data(),
               &work_query, &lwork, &info);
#else
    dgeqrf_(&m, &n, work_A.data(), &lda, tau.data(), &work_query, &lwork,
            &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgeqrf workspace query failed");
        return false;
    }
    lwork = std::max<decltype(lwork)>(1, static_cast<decltype(lwork)>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
#if defined(TENET_NATIVE_USE_BLAS64)
    dgeqrf_64_(&m_lapack, &n_lapack, work_A.data(), &lda_lapack, tau.data(),
               work.data(), &lwork, &info);
#else
    dgeqrf_(&m, &n, work_A.data(), &lda, tau.data(), work.data(), &lwork,
            &info);
#endif
    if (info != 0) {
        set_error("LAPACK dgeqrf failed");
        return false;
    }

    std::fill(R, R + static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
              0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) {
            R[idx2(i, j, n)] = work_A[idx2(i, j, m)];
        }
    }

    lwork = -1;
    work_query = 0.0;
#if defined(TENET_NATIVE_USE_BLAS64)
    dorgqr_64_(&m_lapack, &n_lapack, &n_lapack, work_A.data(), &lda_lapack,
               tau.data(), &work_query, &lwork, &info);
#else
    dorgqr_(&m, &n, &n, work_A.data(), &lda, tau.data(), &work_query,
            &lwork, &info);
#endif
    if (info != 0) {
        set_error("LAPACK dorgqr workspace query failed");
        return false;
    }
    lwork = std::max<decltype(lwork)>(1, static_cast<decltype(lwork)>(work_query));
    work.assign(static_cast<std::size_t>(lwork), 0.0);
#if defined(TENET_NATIVE_USE_BLAS64)
    dorgqr_64_(&m_lapack, &n_lapack, &n_lapack, work_A.data(), &lda_lapack,
               tau.data(), work.data(), &lwork, &info);
#else
    dorgqr_(&m, &n, &n, work_A.data(), &lda, tau.data(), work.data(),
            &lwork, &info);
#endif
    if (info != 0) {
        set_error("LAPACK dorgqr failed");
        return false;
    }
    std::copy(work_A.begin(), work_A.end(), Q);

    for (int j = 0; j < n; ++j) {
        const double phase = R[idx2(j, j, n)] < 0.0 ? -1.0 : 1.0;
        if (phase < 0.0) {
            for (int i = 0; i < m; ++i) {
                Q[idx2(i, j, m)] = -Q[idx2(i, j, m)];
            }
            for (int col = 0; col < n; ++col) {
                R[idx2(j, col, n)] = -R[idx2(j, col, n)];
            }
        }
    }
    return true;
}

bool lapack_lqpos(int64_t m64, int64_t n64, const double *A, double *L,
                  double *Q) {
    if (m64 > n64 || m64 <= 0 || n64 <= 0) {
        set_error("invalid lqpos dimensions");
        return false;
    }
    const std::size_t mt = static_cast<std::size_t>(m64);
    const std::size_t nt = static_cast<std::size_t>(n64);
    std::vector<double> At(nt * mt);
    for (int64_t j = 0; j < n64; ++j) {
        for (int64_t i = 0; i < m64; ++i) {
            At[idx2(j, i, n64)] = A[idx2(i, j, m64)];
        }
    }
    std::vector<double> Qr(nt * mt);
    std::vector<double> Rr(mt * mt);
    if (!lapack_qrpos(n64, m64, At.data(), Qr.data(), Rr.data())) {
        return false;
    }
    for (int64_t j = 0; j < m64; ++j) {
        for (int64_t i = 0; i < m64; ++i) {
            L[idx2(i, j, m64)] = Rr[idx2(j, i, m64)];
        }
    }
    for (int64_t j = 0; j < n64; ++j) {
        for (int64_t i = 0; i < m64; ++i) {
            Q[idx2(i, j, m64)] = Qr[idx2(j, i, n64)];
        }
    }
    return true;
}

void two_layer_apply_gemm(int64_t chi, int64_t phys, const double *Aup,
                          const double *Adn, const double *x, double *tmp,
                          double *y, int transpose) {
    const Leg3ColMajorTwoLayerLayout layout(chi, phys);
    const auto mode = transpose == 0 ? tenet_native_shared::TwoLayerMode::Adjoint
                                     : tenet_native_shared::TwoLayerMode::Forward;
    tenet_native_shared::two_layer_apply(layout, phys, Aup, Adn, x, tmp, y,
                                         mode);
}

void three_layer_apply_gemm_factored(int64_t chi, int64_t phys,
                                     const double *Aup, const double *Adn,
                                     const double *M, const double *x,
                                     double *pair_work, double *accum_work,
                                     double *y, int transpose) {
    const int64_t len2 = chi * chi;
    const int64_t tensor_ld = chi * phys;
    const int64_t nblocks = phys * phys;
    std::fill(y, y + chi * phys * chi, 0.0);
    std::fill(accum_work, accum_work + nblocks * len2, 0.0);
    if (transpose == 0) {
        for (int64_t d = 0; d < phys; ++d) {
            const double *Xd = x + chi * d;
            for (int64_t g = 0; g < phys; ++g) {
                const double *Bg = Adn + chi * g;
                double *pair = pair_work + (d + phys * g) * len2;
                blas_gemm('N', 'N', chi, chi, chi, 1.0, Xd, tensor_ld, Bg,
                          tensor_ld, 0.0, pair, chi);
            }
        }
        for (int64_t e = 0; e < phys; ++e) {
            for (int64_t b = 0; b < phys; ++b) {
                double *accum = accum_work + (e + phys * b) * len2;
                for (int64_t d = 0; d < phys; ++d) {
                    for (int64_t g = 0; g < phys; ++g) {
                        const double alpha =
                            M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha != 0.0) {
                            const double *pair =
                                pair_work + (d + phys * g) * len2;
                            blas_axpy(len2, alpha, pair, accum);
                        }
                    }
                }
            }
        }
        for (int64_t e = 0; e < phys; ++e) {
            double *Ye = y + chi * e;
            for (int64_t b = 0; b < phys; ++b) {
                const double *Ab = Aup + chi * b;
                const double *accum = accum_work + (e + phys * b) * len2;
                blas_gemm('T', 'N', chi, chi, chi, 1.0, Ab, tensor_ld, accum,
                          chi, 1.0, Ye, tensor_ld);
            }
        }
    } else {
        for (int64_t e = 0; e < phys; ++e) {
            const double *Xe = x + chi * e;
            for (int64_t g = 0; g < phys; ++g) {
                const double *Bg = Adn + chi * g;
                double *pair = pair_work + (e + phys * g) * len2;
                blas_gemm('N', 'T', chi, chi, chi, 1.0, Xe, tensor_ld, Bg,
                          tensor_ld, 0.0, pair, chi);
            }
        }
        for (int64_t d = 0; d < phys; ++d) {
            for (int64_t b = 0; b < phys; ++b) {
                double *accum = accum_work + (d + phys * b) * len2;
                for (int64_t e = 0; e < phys; ++e) {
                    for (int64_t g = 0; g < phys; ++g) {
                        const double alpha =
                            M[idx4(d, g, e, b, phys, phys, phys)];
                        if (alpha != 0.0) {
                            const double *pair =
                                pair_work + (e + phys * g) * len2;
                            blas_axpy(len2, alpha, pair, accum);
                        }
                    }
                }
            }
        }
        for (int64_t d = 0; d < phys; ++d) {
            double *Yd = y + chi * d;
            for (int64_t b = 0; b < phys; ++b) {
                const double *Ab = Aup + chi * b;
                const double *accum = accum_work + (d + phys * b) * len2;
                blas_gemm('N', 'N', chi, chi, chi, 1.0, Ab, tensor_ld, accum,
                          chi, 1.0, Yd, tensor_ld);
            }
        }
    }
}

template <typename Apply>
int arnoldi_driver_seeded(int64_t len, int64_t max_k, double breakdown_tol,
                          const double *seed, int64_t seed_cols,
                          int64_t seed_ld, double *V, int64_t ldv, double *H,
                          int64_t ldh, double *beta, int64_t *m,
                          double *final_resnorm, Apply apply) {
    std::fill(V, V + ldv * (max_k + 1), 0.0);
    std::fill(H, H + ldh * max_k, 0.0);
    *beta = 1.0;
    *m = 0;
    *final_resnorm = 0.0;
    const int reorthog_passes = arnoldi_reorthog_passes();
    if (max_k == 0) {
        return TENET_NATIVE_SUCCESS;
    }
    if (seed == nullptr || seed_cols <= 0 || seed_ld < len) {
        set_error("invalid Arnoldi seed basis");
        return TENET_NATIVE_BACKEND_ERROR;
    }

    std::vector<double> q(static_cast<std::size_t>(len));
    int64_t basis_cols = 0;
    const int64_t max_seed_cols = std::min(seed_cols, max_k);
    for (int64_t col = 0; col < max_seed_cols; ++col) {
        std::copy(seed + col * seed_ld, seed + col * seed_ld + len, q.begin());
        for (int pass = 0; pass < reorthog_passes; ++pass) {
            for (int64_t j = 0; j < basis_cols; ++j) {
                const double alpha = dot(len, V + j * ldv, q.data());
                blas_axpy(len, -alpha, V + j * ldv, q.data());
            }
        }
        const double qn = norm2(len, q.data());
        if (!(qn > breakdown_tol) || !std::isfinite(qn)) {
            continue;
        }
        double *v = V + basis_cols * ldv;
        std::copy(q.begin(), q.end(), v);
        scal(len, 1.0 / qn, v);
        ++basis_cols;
    }
    if (basis_cols == 0) {
        return TENET_NATIVE_SUCCESS;
    }

    std::vector<double> w(static_cast<std::size_t>(len));
    std::vector<double> g(static_cast<std::size_t>(max_k));
    double apply_seconds = 0.0;
    double orthog_seconds = 0.0;
    double norm_seconds = 0.0;
    int64_t applied_cols = 0;
    for (int64_t j = 0; j < max_k && j < basis_cols; ++j) {
        const double *vj = V + j * ldv;
        double *hj = H + j * ldh;
        auto section_start = SteadyClock::now();
        apply(vj, w.data());
        ++applied_cols;
        apply_seconds += seconds_since(section_start);
        const int64_t k = basis_cols;
        section_start = SteadyClock::now();
        for (int pass = 0; pass < reorthog_passes; ++pass) {
            blas_gemv('T', len, k, 1.0, V, ldv, w.data(), 0.0, g.data());
            blas_axpy(k, 1.0, g.data(), hj);
            blas_gemv('N', len, k, -1.0, V, ldv, g.data(), 1.0, w.data());
        }
        orthog_seconds += seconds_since(section_start);
        section_start = SteadyClock::now();
        const double hnext = norm2(len, w.data());
        norm_seconds += seconds_since(section_start);
        *final_resnorm = hnext;
        *m = j + 1;
        if (hnext > breakdown_tol && basis_cols < max_k) {
            hj[basis_cols] = hnext;
            double *vnext = V + basis_cols * ldv;
            std::copy(w.begin(), w.end(), vnext);
            scal(len, 1.0 / hnext, vnext);
            ++basis_cols;
        } else if (hnext > breakdown_tol && basis_cols == max_k) {
            hj[basis_cols] = hnext;
            double *vnext = V + basis_cols * ldv;
            std::copy(w.begin(), w.end(), vnext);
            scal(len, 1.0 / hnext, vnext);
        } else if (hnext <= breakdown_tol && j + 1 >= basis_cols) {
            break;
        }
    }
    profile_arnoldi(len, max_k, *m, applied_cols, apply_seconds, orthog_seconds,
                    norm_seconds, *final_resnorm);
    return TENET_NATIVE_SUCCESS;
}

template <typename Apply>
int arnoldi_driver(int64_t len, int64_t max_k, double breakdown_tol,
                   const double *x0, double *V, int64_t ldv, double *H,
                   int64_t ldh, double *beta, int64_t *m,
                   double *final_resnorm, Apply apply) {
    return arnoldi_driver_seeded(len, max_k, breakdown_tol, x0, 1, len, V, ldv,
                                 H, ldh, beta, m, final_resnorm, apply);
}

template <typename Apply>
int arnoldi_driver_prefilled(int64_t len, int64_t max_k, double breakdown_tol,
                             const double *initial_V, int64_t initial_cols,
                             int64_t initial_ldv, const double *initial_H,
                             int64_t initial_ldh, int64_t completed_cols,
                             double *V, int64_t ldv, double *H, int64_t ldh,
                             double *beta, int64_t *m,
                             double *final_resnorm, Apply apply) {
    if (initial_V == nullptr || initial_cols <= 0 || initial_ldv < len ||
        initial_H == nullptr || initial_ldh < completed_cols + 1 ||
        completed_cols < 0 || completed_cols > initial_cols ||
        initial_cols > max_k + 1) {
        set_error("invalid prefilled Arnoldi restart");
        return TENET_NATIVE_BACKEND_ERROR;
    }
    std::fill(V, V + ldv * (max_k + 1), 0.0);
    std::fill(H, H + ldh * max_k, 0.0);
    for (int64_t col = 0; col < initial_cols; ++col) {
        std::copy(initial_V + col * initial_ldv,
                  initial_V + col * initial_ldv + len, V + col * ldv);
    }
    for (int64_t col = 0; col < completed_cols; ++col) {
        std::copy(initial_H + col * initial_ldh,
                  initial_H + col * initial_ldh + completed_cols + 1,
                  H + col * ldh);
    }
    *beta = 1.0;
    *m = completed_cols;
    *final_resnorm = 0.0;
    const int reorthog_passes = arnoldi_reorthog_passes();
    if (completed_cols >= max_k) {
        profile_arnoldi(len, max_k, *m, 0, 0.0, 0.0, 0.0, *final_resnorm);
        return TENET_NATIVE_SUCCESS;
    }

    std::vector<double> w(static_cast<std::size_t>(len));
    std::vector<double> g(static_cast<std::size_t>(max_k + 1));
    double apply_seconds = 0.0;
    double orthog_seconds = 0.0;
    double norm_seconds = 0.0;
    int64_t applied_cols = 0;
    int64_t basis_cols = initial_cols;
    for (int64_t j = completed_cols; j < max_k && j < basis_cols; ++j) {
        const double *vj = V + j * ldv;
        double *hj = H + j * ldh;
        auto section_start = SteadyClock::now();
        apply(vj, w.data());
        ++applied_cols;
        apply_seconds += seconds_since(section_start);
        const int64_t k = basis_cols;
        section_start = SteadyClock::now();
        for (int pass = 0; pass < reorthog_passes; ++pass) {
            blas_gemv('T', len, k, 1.0, V, ldv, w.data(), 0.0, g.data());
            blas_axpy(k, 1.0, g.data(), hj);
            blas_gemv('N', len, k, -1.0, V, ldv, g.data(), 1.0, w.data());
        }
        orthog_seconds += seconds_since(section_start);
        section_start = SteadyClock::now();
        const double hnext = norm2(len, w.data());
        norm_seconds += seconds_since(section_start);
        *final_resnorm = hnext;
        *m = j + 1;
        if (hnext > breakdown_tol && basis_cols < max_k) {
            hj[basis_cols] = hnext;
            double *vnext = V + basis_cols * ldv;
            std::copy(w.begin(), w.end(), vnext);
            scal(len, 1.0 / hnext, vnext);
            ++basis_cols;
        } else if (hnext > breakdown_tol && basis_cols == max_k) {
            hj[basis_cols] = hnext;
            double *vnext = V + basis_cols * ldv;
            std::copy(w.begin(), w.end(), vnext);
            scal(len, 1.0 / hnext, vnext);
        } else if (hnext <= breakdown_tol && j + 1 >= basis_cols) {
            break;
        }
    }
    profile_arnoldi(len, max_k, *m, applied_cols, apply_seconds, orthog_seconds,
                    norm_seconds, *final_resnorm);
    return TENET_NATIVE_SUCCESS;
}

enum class RitzTarget { LargestMagnitude, SmallestReal };

bool ritz_precedes(double ar, double ai, double br, double bi,
                   RitzTarget target) {
    if (target == RitzTarget::SmallestReal) {
        const double scale = std::max(1.0, std::max(std::abs(ar), std::abs(br)));
        if (std::abs(ar - br) > 1e-10 * scale) {
            return ar < br;
        }
        return std::abs(ai) < std::abs(bi);
    }
    const double ma = std::hypot(ar, ai);
    const double mb = std::hypot(br, bi);
    if (std::abs(ma - mb) > 1e-10 * std::max(1.0, std::max(ma, mb))) {
        return ma > mb;
    }
    return ar > br;
}

bool select_ritz(int64_t len, int64_t m, double beta,
                 double final_resnorm, const double *V, int64_t ldv,
                 const double *H, int64_t ldh, double *out,
                 double *ritz_resnorm, RitzTarget target) {
    if (m <= 0 || len <= 0) {
        set_error("empty Arnoldi basis");
        return false;
    }
    if (m > INT_MAX || len > INT_MAX) {
        set_error("Ritz dimensions exceed int range");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            Hm[idx2(row, col, m)] = H[idx2(row, col, ldh)];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }

    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (ritz_precedes(wr[i], wi[i], wr[best], wi[best], target)) {
            best = i;
        }
    }
    std::vector<double> coeff(static_cast<std::size_t>(m), 0.0);
    if (std::abs(wi[best]) > 1e-8 * std::max(1.0, std::abs(wr[best]))) {
        if (target != RitzTarget::LargestMagnitude ||
            !select_dominant_ritz_coeff_power(m, beta, H, ldh, coeff.data())) {
            set_error("selected Ritz value is complex");
            return false;
        }
    } else {
        for (int64_t col = 0; col < m; ++col) {
            coeff[static_cast<std::size_t>(col)] =
                beta * vr[idx2(col, best, m)];
        }
    }
    const double coeff_norm = norm2(m, coeff.data());
    if (!(coeff_norm > 0.0) || !std::isfinite(coeff_norm)) {
        set_error("selected Ritz coefficient vector has invalid norm");
        return false;
    }
    if (ritz_resnorm != nullptr) {
        *ritz_resnorm =
            final_resnorm * std::abs(coeff[static_cast<std::size_t>(m - 1)]) /
            coeff_norm;
    }

    std::fill(out, out + len, 0.0);
    for (int64_t col = 0; col < m; ++col) {
        const double *vcol = V + col * ldv;
        for (int64_t row = 0; row < len; ++row) {
            out[row] += coeff[static_cast<std::size_t>(col)] * vcol[row];
        }
    }
    const double yn = norm2(len, out);
    if (!(yn > 0.0) || !std::isfinite(yn)) {
        set_error("selected Ritz vector has invalid norm");
        return false;
    }
    scal(len, 1.0 / yn, out);
    int64_t pivot = 0;
    double pivot_abs = std::abs(out[0]);
    for (int64_t i = 1; i < len; ++i) {
        const double ai = std::abs(out[i]);
        if (ai > pivot_abs) {
            pivot = i;
            pivot_abs = ai;
        }
    }
    if (out[pivot] < 0.0) {
        scal(len, -1.0, out);
    }
    return true;
}

bool select_ritz_values(int64_t m, const double *H, int64_t ldh,
                        int64_t nvalues, double *lambda_real,
                        double *lambda_imag, RitzTarget target) {
    if (m <= 0 || m > INT_MAX || nvalues <= 0 || lambda_real == nullptr ||
        lambda_imag == nullptr) {
        set_error("invalid Ritz value output dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            Hm[idx2(row, col, m)] = H[idx2(row, col, ldh)];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return ritz_precedes(wr[a], wi[a], wr[b], wi[b], target);
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

bool select_ritz_convergence_residual(int64_t m, double beta,
                                      double final_resnorm, const double *H,
                                      int64_t ldh, int64_t nvalues,
                                      double *max_ritz_resnorm,
                                      RitzTarget target) {
    if (m <= 0 || m > INT_MAX || nvalues <= 0 ||
        max_ritz_resnorm == nullptr) {
        set_error("invalid Ritz convergence dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            Hm[idx2(row, col, m)] = H[idx2(row, col, ldh)];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return ritz_precedes(wr[a], wi[a], wr[b], wi[b], target);
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
                set_error("complex Ritz pair is missing its conjugate column");
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
            set_error("Ritz convergence coefficient norm is invalid");
            return false;
        }
        const double res = final_resnorm * tail / coeff_norm;
        if (!std::isfinite(res)) {
            set_error("Ritz convergence residual is not finite");
            return false;
        }
        max_res = std::max(max_res, res);
    }
    *max_ritz_resnorm = max_res;
    return true;
}

bool select_restart_schur_basis(int64_t m, const double *H, int64_t ldh,
                                int64_t keep, double *schur_basis,
                                double *schur_form, int64_t *schur_cols,
                                RitzTarget target) {
    if (m <= 0 || m > INT_MAX || keep <= 0) {
        set_error("invalid restart coefficient dimensions");
        return false;
    }
    const int n = static_cast<int>(m);
    std::vector<double> Hm(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            Hm[idx2(row, col, m)] = H[idx2(row, col, ldh)];
        }
    }
    std::vector<double> wr(static_cast<std::size_t>(n));
    std::vector<double> wi(static_cast<std::size_t>(n));
    std::vector<double> vr(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    if (!lapack_geev(n, Hm, wr, wi, vr)) {
        return false;
    }
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return ritz_precedes(wr[a], wi[a], wr[b], wi[b], target);
    });

    if (!choose_schur_restart_values(wr, wi, order, static_cast<int>(keep))) {
        set_error("failed to choose Schur restart values");
        return false;
    }

    std::vector<double> Hs(static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(n));
    for (int64_t col = 0; col < m; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            Hs[idx2(row, col, m)] = H[idx2(row, col, ldh)];
        }
    }
    std::vector<double> schur_wr(static_cast<std::size_t>(n));
    std::vector<double> schur_wi(static_cast<std::size_t>(n));
    std::vector<double> schur_vecs(static_cast<std::size_t>(n) *
                                   static_cast<std::size_t>(n));
    int sdim = 0;
    const bool schur_ok =
        lapack_schur_selected(n, Hs, schur_wr, schur_wi, schur_vecs, &sdim);
    schur_select_wr.clear();
    schur_select_wi.clear();
    if (!schur_ok) {
        return false;
    }
    if (sdim <= 0 || sdim > keep) {
        set_error("selected Schur restart dimension is invalid");
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

struct RealHouseholder {
    double beta = 0.0;
    std::vector<double> v;
    double nu = 0.0;
};

RealHouseholder row_householder_last_pivot(const std::vector<double> &x) {
    RealHouseholder h;
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

void apply_householder_left(int64_t rows, int64_t cols, const RealHouseholder &h,
                            double *A, int64_t lda) {
    if (h.beta == 0.0) {
        return;
    }
    for (int64_t col = 0; col < cols; ++col) {
        double mu = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            mu += h.v[static_cast<std::size_t>(i)] * A[idx2(i, col, lda)];
        }
        mu *= h.beta;
        for (int64_t i = 0; i < rows; ++i) {
            A[idx2(i, col, lda)] -=
                mu * h.v[static_cast<std::size_t>(i)];
        }
    }
}

void apply_householder_right(int64_t active_rows, int64_t cols,
                             const RealHouseholder &h, double *A,
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
            A[idx2(row, col, lda)] -= work[static_cast<std::size_t>(row)] * vc;
        }
    }
}

void restore_krylovkit_arnoldi_form(int64_t k, double *Hwork, int64_t ldh,
                                    double *basis_coeffs, int64_t ldb) {
    for (int64_t j = k; j >= 1; --j) {
        std::vector<double> row(static_cast<std::size_t>(j));
        for (int64_t col = 0; col < j; ++col) {
            row[static_cast<std::size_t>(col)] = Hwork[idx2(j, col, ldh)];
        }
        const RealHouseholder h = row_householder_last_pivot(row);
        for (int64_t col = 0; col < j - 1; ++col) {
            Hwork[idx2(j, col, ldh)] = 0.0;
        }
        Hwork[idx2(j, j - 1, ldh)] = h.nu;
        apply_householder_left(j, k, h, Hwork, ldh);
        apply_householder_right(j, j, h, Hwork, ldh);
        apply_householder_right(ldb, j, h, basis_coeffs, ldb);
    }
}

bool build_compressed_restart(int64_t len, int64_t max_k, int64_t m,
                              const double *V, int64_t ldv, const double *H,
                              int64_t ldh, double final_resnorm, int64_t keep,
                              double breakdown_tol, double *out,
                              double *restart_V, double *restart_H,
                              int64_t *restart_cols,
                              int64_t *completed_cols, RitzTarget target) {
    if (m <= 0 || keep <= 1 || keep >= max_k || V == nullptr || H == nullptr ||
        out == nullptr || restart_V == nullptr || restart_H == nullptr) {
        set_error("invalid compressed restart inputs");
        return false;
    }
    const int64_t kmax = std::min<int64_t>(keep, m);
    std::vector<double> C(static_cast<std::size_t>(m) *
                          static_cast<std::size_t>(kmax),
                          0.0);
    std::vector<double> T(static_cast<std::size_t>(kmax) *
                          static_cast<std::size_t>(kmax),
                          0.0);
    int64_t k = 0;
    if (!select_restart_schur_basis(m, H, ldh, kmax, C.data(), T.data(), &k,
                                    target)) {
        return false;
    }
    if (k <= 1) {
        set_error("compressed restart selected only one vector");
        return false;
    }

    std::fill(out, out + len, 0.0);
    for (int64_t col = 0; col < m; ++col) {
        const double alpha = C[idx2(col, 0, m)];
        const double *vcol = V + col * ldv;
        for (int64_t row = 0; row < len; ++row) {
            out[row] += alpha * vcol[row];
        }
    }
    const double out_norm = norm2(len, out);
    if (!(out_norm > 0.0) || !std::isfinite(out_norm)) {
        set_error("compressed restart dominant vector has invalid norm");
        return false;
    }
    scal(len, 1.0 / out_norm, out);

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
    restore_krylovkit_arnoldi_form(k, Hwork.data(), k + 1, C.data(), m);
    const double tail_norm = std::abs(Hwork[idx2(k, k - 1, k + 1)]);

    std::fill(restart_V, restart_V + len * (max_k + 1), 0.0);
    blas_gemm('N', 'N', len, k, m, 1.0, V, ldv, C.data(), m, 0.0,
              restart_V, len);
    std::fill(restart_H, restart_H + (max_k + 1) * max_k, 0.0);
    for (int64_t col = 0; col < k; ++col) {
        for (int64_t row = 0; row < k + 1; ++row) {
            restart_H[idx2(row, col, max_k + 1)] =
                Hwork[idx2(row, col, k + 1)];
        }
    }
    *completed_cols = k;
    if (tail_norm > breakdown_tol) {
        std::copy(V + m * ldv, V + m * ldv + len, restart_V + k * len);
        *restart_cols = k + 1;
    } else {
        *restart_cols = k;
    }
    return true;
}

template <typename Apply>
int dominant_arnoldi_vector(int64_t len, int64_t max_k, double breakdown_tol,
                            const double *x0, double *out, Apply apply,
                            RitzTarget target = RitzTarget::LargestMagnitude) {
    std::vector<double> V(static_cast<std::size_t>(len) *
                          static_cast<std::size_t>(max_k + 1));
    std::vector<double> H(static_cast<std::size_t>(max_k + 1) *
                          static_cast<std::size_t>(max_k));
    const int64_t thick_keep = dominant_thick_keep_count(max_k);
    std::vector<double> restart_single(static_cast<std::size_t>(len));
    std::vector<double> restart_V(static_cast<std::size_t>(len) *
                                  static_cast<std::size_t>(max_k + 1));
    std::vector<double> restart_H(static_cast<std::size_t>(max_k + 1) *
                                  static_cast<std::size_t>(max_k));
    const double *seed = x0;
    int64_t seed_cols = 1;
    bool have_compressed_restart = false;
    int64_t restart_cols = 0;
    int64_t completed_cols = 0;
    const int max_blocks = dominant_restart_blocks(len, max_k);
    const double eig_tol = dominant_residual_tol(breakdown_tol);
    const int64_t convergence_nvalues = dominant_convergence_nvalues();
    last_dominant_relres = std::numeric_limits<double>::infinity();
    for (int block = 0; block < max_blocks; ++block) {
        double beta = 0.0;
        int64_t m = 0;
        double final_resnorm = 0.0;
        const int status =
            have_compressed_restart
                ? arnoldi_driver_prefilled(
                      len, max_k, breakdown_tol, restart_V.data(),
                      restart_cols, len, restart_H.data(), max_k + 1,
                      completed_cols, V.data(), len, H.data(), max_k + 1,
                      &beta, &m, &final_resnorm, apply)
                : arnoldi_driver_seeded(len, max_k, breakdown_tol, seed,
                                        seed_cols, len, V.data(), len,
                                        H.data(), max_k + 1, &beta, &m,
                                        &final_resnorm, apply);
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        const bool want_thick_restart = thick_keep > 1 && block + 1 < max_blocks;
        double ritz_resnorm = std::numeric_limits<double>::infinity();
        const bool selected = select_ritz(
            len, m, beta, final_resnorm, V.data(), len, H.data(), max_k + 1,
            out, &ritz_resnorm, target);
        if (!selected) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        if (!std::isfinite(ritz_resnorm)) {
            set_error("dominant Arnoldi Ritz residual is not finite");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        double convergence_resnorm = ritz_resnorm;
        if (convergence_nvalues > 1 &&
            !select_ritz_convergence_residual(
                m, beta, final_resnorm, H.data(), max_k + 1,
                convergence_nvalues, &convergence_resnorm, target)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        last_dominant_relres = convergence_resnorm;
        if (convergence_resnorm <= eig_tol || block + 1 == max_blocks) {
            return TENET_NATIVE_SUCCESS;
        }
        if (want_thick_restart &&
            build_compressed_restart(len, max_k, m, V.data(), len, H.data(),
                                     max_k + 1, final_resnorm, thick_keep,
                                     breakdown_tol, out, restart_V.data(),
                                     restart_H.data(), &restart_cols,
                                     &completed_cols, target)) {
            have_compressed_restart = true;
            seed = nullptr;
            seed_cols = 0;
        } else {
            std::copy(out, out + len, restart_single.begin());
            seed = restart_single.data();
            seed_cols = 1;
            have_compressed_restart = false;
            restart_cols = 0;
            completed_cols = 0;
        }
    }
    return TENET_NATIVE_SUCCESS;
}

template <typename Apply>
int restarted_arnoldi_ritz_values(int64_t len, int64_t max_k,
                                  double breakdown_tol, const double *x0,
                                  int64_t nvalues, double *lambda_real,
                                  double *lambda_imag, int64_t *m_out,
                                  Apply apply,
                                  RitzTarget target =
                                      RitzTarget::LargestMagnitude) {
    if (len <= 0 || max_k <= 0 || max_k > len || breakdown_tol < 0.0 ||
        x0 == nullptr || nvalues <= 0 || lambda_real == nullptr ||
        lambda_imag == nullptr || m_out == nullptr) {
        set_error("invalid restarted Ritz inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    std::vector<double> V(static_cast<std::size_t>(len) *
                          static_cast<std::size_t>(max_k + 1));
    std::vector<double> H(static_cast<std::size_t>(max_k + 1) *
                          static_cast<std::size_t>(max_k));
    std::vector<double> out(static_cast<std::size_t>(len));
    const int64_t thick_keep = dominant_thick_keep_count(max_k);
    std::vector<double> restart_single(static_cast<std::size_t>(len));
    std::vector<double> restart_V(static_cast<std::size_t>(len) *
                                  static_cast<std::size_t>(max_k + 1));
    std::vector<double> restart_H(static_cast<std::size_t>(max_k + 1) *
                                  static_cast<std::size_t>(max_k));
    const double *seed = x0;
    int64_t seed_cols = 1;
    bool have_compressed_restart = false;
    int64_t restart_cols = 0;
    int64_t completed_cols = 0;
    const int max_blocks = dominant_restart_blocks(len, max_k);
    last_dominant_relres = std::numeric_limits<double>::infinity();
    for (int block = 0; block < max_blocks; ++block) {
        double beta = 0.0;
        int64_t m = 0;
        double final_resnorm = 0.0;
        const int status =
            have_compressed_restart
                ? arnoldi_driver_prefilled(
                      len, max_k, breakdown_tol, restart_V.data(),
                      restart_cols, len, restart_H.data(), max_k + 1,
                      completed_cols, V.data(), len, H.data(), max_k + 1,
                      &beta, &m, &final_resnorm, apply)
                : arnoldi_driver_seeded(len, max_k, breakdown_tol, seed,
                                        seed_cols, len, V.data(), len,
                                        H.data(), max_k + 1, &beta, &m,
                                        &final_resnorm, apply);
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        double ritz_resnorm = std::numeric_limits<double>::infinity();
        if (!select_ritz(len, m, beta, final_resnorm, V.data(), len,
                         H.data(), max_k + 1, out.data(), &ritz_resnorm,
                         target) ||
            !select_ritz_values(m, H.data(), max_k + 1, nvalues,
                                lambda_real, lambda_imag, target)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        *m_out = m;
        double convergence_resnorm = ritz_resnorm;
        if (nvalues > 1 &&
            !select_ritz_convergence_residual(
                m, beta, final_resnorm, H.data(), max_k + 1, nvalues,
                &convergence_resnorm, target)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        last_dominant_relres = convergence_resnorm;
        if (block + 1 == max_blocks) {
            return TENET_NATIVE_SUCCESS;
        }
        const bool want_thick_restart = thick_keep > 1;
        if (want_thick_restart &&
            build_compressed_restart(len, max_k, m, V.data(), len, H.data(),
                                     max_k + 1, final_resnorm, thick_keep,
                                     breakdown_tol, out.data(),
                                     restart_V.data(), restart_H.data(),
                                     &restart_cols, &completed_cols, target)) {
            have_compressed_restart = true;
            seed = nullptr;
            seed_cols = 0;
        } else {
            std::copy(out.begin(), out.end(), restart_single.begin());
            seed = restart_single.data();
            seed_cols = 1;
            have_compressed_restart = false;
            restart_cols = 0;
            completed_cols = 0;
        }
    }
    return TENET_NATIVE_SUCCESS;
}

void alc_to_ac(int64_t chi, int64_t phys, const double *AL, const double *C,
               double *AC) {
    const int64_t tensor_ld = chi * phys;
    for (int64_t s = 0; s < phys; ++s) {
        const double *ALs = AL + chi * s;
        double *ACs = AC + chi * s;
        blas_gemm('N', 'N', chi, chi, chi, 1.0, ALs, tensor_ld, C, chi,
                  0.0, ACs, tensor_ld);
    }
}

bool acc_to_alar(int64_t chi, int64_t phys, const double *AC, const double *C,
                 double *AL, double *AR, double *err) {
    const int64_t tall = chi * phys;
    std::vector<double> QAC(static_cast<std::size_t>(tall) *
                            static_cast<std::size_t>(chi));
    std::vector<double> RAC(static_cast<std::size_t>(chi) *
                            static_cast<std::size_t>(chi));
    std::vector<double> QC(static_cast<std::size_t>(chi) *
                           static_cast<std::size_t>(chi));
    std::vector<double> RC(static_cast<std::size_t>(chi) *
                           static_cast<std::size_t>(chi));
    if (!lapack_qrpos(tall, chi, AC, QAC.data(), RAC.data()) ||
        !lapack_qrpos(chi, chi, C, QC.data(), RC.data())) {
        return false;
    }
    double errL = 0.0;
    for (int64_t i = 0; i < chi * chi; ++i) {
        const double d = RAC[i] - RC[i];
        errL += d * d;
    }
    errL = std::sqrt(errL);
    blas_gemm('N', 'T', tall, chi, chi, 1.0, QAC.data(), tall, QC.data(),
              chi, 0.0, AL, tall);

    std::vector<double> ACtail(static_cast<std::size_t>(chi) *
                               static_cast<std::size_t>(tall));
    for (int64_t j = 0; j < tall; ++j) {
        for (int64_t i = 0; i < chi; ++i) {
            ACtail[idx2(i, j, chi)] = AC[i + chi * j];
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
    if (!lapack_lqpos(chi, tall, ACtail.data(), LAC.data(), QLAC.data()) ||
        !lapack_lqpos(chi, chi, C, LC.data(), QLC.data())) {
        return false;
    }
    double errR = 0.0;
    for (int64_t i = 0; i < chi * chi; ++i) {
        const double d = LAC[i] - LC[i];
        errR += d * d;
    }
    errR = std::sqrt(errR);
    std::vector<double> ARtail(static_cast<std::size_t>(chi) *
                               static_cast<std::size_t>(tall));
    blas_gemm('T', 'N', chi, tall, chi, 1.0, QLC.data(), chi, QLAC.data(),
              chi, 0.0, ARtail.data(), chi);
    std::copy(ARtail.begin(), ARtail.end(), AR);
    *err = errL + errR;
    return true;
}

void permute_m_for_ac(int64_t phys, const double *M, double *Mp) {
    for (int64_t i = 0; i < phys; ++i) {
        for (int64_t j = 0; j < phys; ++j) {
            for (int64_t k = 0; k < phys; ++k) {
                for (int64_t l = 0; l < phys; ++l) {
                    Mp[idx4(i, j, k, l, phys, phys, phys)] =
                        M[idx4(l, k, j, i, phys, phys, phys)];
                }
            }
        }
    }
}

bool check_raw_transfer_op_inputs(int64_t phys, int64_t chi, const double *W,
                                 const double *O, const double *x, double *y) {
    if (W == nullptr || O == nullptr || x == nullptr || y == nullptr) {
        set_error("null raw transfer-op tensor pointer");
        return false;
    }
    if (phys <= 0 || chi <= 0) {
        set_error("invalid raw transfer-op dimensions");
        return false;
    }
    return true;
}

void raw_transfer_op_gemm(int64_t chi, int64_t phys, const double *W,
                          const double *O, const double *x, double *tmp,
                          double *work, double *y) {
    const Leg3ColMajorTwoLayerLayout layout(chi, phys);
    std::fill(y, y + chi * chi, 0.0);
    for (int64_t s = 0; s < phys; ++s) {
        const double *As = layout.slice(W, s);
        for (int64_t t = 0; t < phys; ++t) {
            const double coeff = O[s + t * phys];
            if (coeff == 0.0) {
                continue;
            }
            const double *At = layout.slice(W, t);
            std::fill(work, work + chi * chi, 0.0);
            layout.apply_forward_slice(At, As, x, tmp, work);
            for (int64_t i = 0; i < chi * chi; ++i) {
                y[i] += coeff * work[i];
            }
        }
    }
}

} // namespace

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

namespace {

using Complex64 = std::complex<double>;

Complex64 to_cpp_complex(tenet_native_complex64 z) {
    return Complex64(z.re, z.im);
}

tenet_native_complex64 to_c_complex(Complex64 z) {
    return tenet_native_complex64{std::real(z), std::imag(z)};
}

bool check_generic_krylov_common(int64_t n, int64_t max_k, double tol) {
    if (n <= 0 || max_k <= 0 || max_k > n || tol < 0.0) {
        set_error("invalid generic Krylov dimensions or tolerance");
        return false;
    }
    return true;
}

bool check_prefilled_krylov_common(int64_t n, int64_t initial_cols,
                                   int64_t completed_cols, int64_t max_k,
                                   double tol) {
    if (!check_generic_krylov_common(n, max_k, tol)) {
        return false;
    }
    if (initial_cols <= 0 || initial_cols > max_k + 1 ||
        completed_cols < 0 || completed_cols > max_k ||
        completed_cols > initial_cols) {
        set_error("invalid prefilled Arnoldi dimensions");
        return false;
    }
    return true;
}

void dense_matvec_real(int64_t n, const double *A, int64_t lda,
                       const double *x, double *y) {
    blas_gemv('N', n, n, 1.0, A, lda, x, 0.0, y);
}

void dense_matvec_complex(int64_t n, const Complex64 *A, int64_t lda,
                          const Complex64 *x, Complex64 *y) {
    std::fill(y, y + n, Complex64(0.0, 0.0));
    for (int64_t col = 0; col < n; ++col) {
        const Complex64 alpha = x[col];
        const Complex64 *Acol = A + col * lda;
        for (int64_t row = 0; row < n; ++row) {
            y[row] += Acol[row] * alpha;
        }
    }
}

} // namespace

extern "C" int tenet_native_krylov_arnoldi_d_cpu(
    int64_t n, const double *x0, int64_t max_k, double breakdown_tol,
    tenet_native_matvec_d_cpu_fn matvec, void *ctx, double *V, int64_t ldv,
    double *H, int64_t ldh, double *beta, int64_t *m,
    double *final_resnorm, int64_t *numops) {
    if (!check_generic_krylov_common(n, max_k, breakdown_tol) ||
        x0 == nullptr || matvec == nullptr || V == nullptr || H == nullptr ||
        beta == nullptr || m == nullptr || final_resnorm == nullptr ||
        numops == nullptr || ldv < n || ldh < max_k + 1) {
        set_error("invalid generic real Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::arnoldi<double>(
            n, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
            final_resnorm, numops,
            [&](const double *src, double *dst) {
                return matvec(n, src, dst, ctx);
            });
        if (status != 0) {
            set_error("generic real Arnoldi matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic real Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_z_cpu(
    int64_t n, const tenet_native_complex64 *x0, int64_t max_k,
    double breakdown_tol, tenet_native_matvec_z_cpu_fn matvec, void *ctx,
    tenet_native_complex64 *V, int64_t ldv, tenet_native_complex64 *H,
    int64_t ldh, double *beta, int64_t *m, double *final_resnorm,
    int64_t *numops) {
    if (!check_generic_krylov_common(n, max_k, breakdown_tol) ||
        x0 == nullptr || matvec == nullptr || V == nullptr || H == nullptr ||
        beta == nullptr || m == nullptr || final_resnorm == nullptr ||
        numops == nullptr || ldv < n || ldh < max_k + 1) {
        set_error("invalid generic complex Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> Vc(static_cast<std::size_t>(ldv) *
                                  static_cast<std::size_t>(max_k + 1));
        std::vector<Complex64> Hc(static_cast<std::size_t>(ldh) *
                                  static_cast<std::size_t>(max_k));
        std::vector<tenet_native_complex64> src(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> dst(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::arnoldi<Complex64>(
            n, max_k, breakdown_tol, x0c.data(), Vc.data(), ldv, Hc.data(),
            ldh, beta, m, final_resnorm, numops,
            [&](const Complex64 *src_cpp, Complex64 *dst_cpp) {
                for (int64_t i = 0; i < n; ++i) {
                    src[static_cast<std::size_t>(i)] =
                        to_c_complex(src_cpp[i]);
                }
                const int st = matvec(n, src.data(), dst.data(), ctx);
                if (st != 0) {
                    return st;
                }
                for (int64_t i = 0; i < n; ++i) {
                    dst_cpp[i] = to_cpp_complex(dst[static_cast<std::size_t>(i)]);
                }
                return 0;
            });
        if (status != 0) {
            set_error("generic complex Arnoldi matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t col = 0; col < max_k + 1; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                V[row + col * ldv] = to_c_complex(Vc[static_cast<std::size_t>(
                    row + col * ldv)]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                H[row + col * ldh] = to_c_complex(Hc[static_cast<std::size_t>(
                    row + col * ldh)]);
            }
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic complex Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_prefilled_d_cpu(
    int64_t n, const double *initial_V, int64_t initial_ldv,
    int64_t initial_cols, const double *initial_H, int64_t initial_ldh,
    int64_t completed_cols, int64_t max_k, double breakdown_tol,
    tenet_native_matvec_d_cpu_fn matvec, void *ctx, double *V, int64_t ldv,
    double *H, int64_t ldh, double *beta, int64_t *m,
    double *final_resnorm, int64_t *numops) {
    if (!check_prefilled_krylov_common(n, initial_cols, completed_cols, max_k,
                                       breakdown_tol) ||
        initial_V == nullptr || initial_H == nullptr || matvec == nullptr ||
        V == nullptr || H == nullptr || beta == nullptr || m == nullptr ||
        final_resnorm == nullptr || numops == nullptr || initial_ldv < n ||
        initial_ldh < max_k + 1 || ldv < n || ldh < max_k + 1) {
        set_error("invalid prefilled generic real Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::arnoldi_prefilled<double>(
            n, max_k, breakdown_tol, initial_V, initial_ldv, initial_cols,
            initial_H, initial_ldh, completed_cols, V, ldv, H, ldh, beta, m,
            final_resnorm, numops,
            [&](const double *src, double *dst) {
                return matvec(n, src, dst, ctx);
            });
        if (status != 0) {
            set_error("prefilled generic real Arnoldi matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown prefilled generic real Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_prefilled_z_cpu(
    int64_t n, const tenet_native_complex64 *initial_V, int64_t initial_ldv,
    int64_t initial_cols, const tenet_native_complex64 *initial_H,
    int64_t initial_ldh, int64_t completed_cols, int64_t max_k,
    double breakdown_tol, tenet_native_matvec_z_cpu_fn matvec, void *ctx,
    tenet_native_complex64 *V, int64_t ldv, tenet_native_complex64 *H,
    int64_t ldh, double *beta, int64_t *m, double *final_resnorm,
    int64_t *numops) {
    if (!check_prefilled_krylov_common(n, initial_cols, completed_cols, max_k,
                                       breakdown_tol) ||
        initial_V == nullptr || initial_H == nullptr || matvec == nullptr ||
        V == nullptr || H == nullptr || beta == nullptr || m == nullptr ||
        final_resnorm == nullptr || numops == nullptr || initial_ldv < n ||
        initial_ldh < max_k + 1 || ldv < n || ldh < max_k + 1) {
        set_error("invalid prefilled generic complex Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> initial_Vc(
            static_cast<std::size_t>(initial_ldv) *
            static_cast<std::size_t>(initial_cols));
        std::vector<Complex64> initial_Hc(
            static_cast<std::size_t>(initial_ldh) *
            static_cast<std::size_t>(max_k));
        std::vector<Complex64> Vc(static_cast<std::size_t>(ldv) *
                                  static_cast<std::size_t>(max_k + 1));
        std::vector<Complex64> Hc(static_cast<std::size_t>(ldh) *
                                  static_cast<std::size_t>(max_k));
        std::vector<tenet_native_complex64> src(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> dst(static_cast<std::size_t>(n));
        for (int64_t col = 0; col < initial_cols; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                initial_Vc[static_cast<std::size_t>(row +
                                                    col * initial_ldv)] =
                    to_cpp_complex(initial_V[row + col * initial_ldv]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                initial_Hc[static_cast<std::size_t>(row +
                                                    col * initial_ldh)] =
                    to_cpp_complex(initial_H[row + col * initial_ldh]);
            }
        }
        const int status = tenet_native_krylov::arnoldi_prefilled<Complex64>(
            n, max_k, breakdown_tol, initial_Vc.data(), initial_ldv,
            initial_cols, initial_Hc.data(), initial_ldh, completed_cols,
            Vc.data(), ldv, Hc.data(), ldh, beta, m, final_resnorm, numops,
            [&](const Complex64 *src_cpp, Complex64 *dst_cpp) {
                for (int64_t i = 0; i < n; ++i) {
                    src[static_cast<std::size_t>(i)] =
                        to_c_complex(src_cpp[i]);
                }
                const int st = matvec(n, src.data(), dst.data(), ctx);
                if (st != 0) {
                    return st;
                }
                for (int64_t i = 0; i < n; ++i) {
                    dst_cpp[i] = to_cpp_complex(dst[static_cast<std::size_t>(i)]);
                }
                return 0;
            });
        if (status != 0) {
            set_error("prefilled generic complex Arnoldi matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t col = 0; col < max_k + 1; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                V[row + col * ldv] = to_c_complex(
                    Vc[static_cast<std::size_t>(row + col * ldv)]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                H[row + col * ldh] = to_c_complex(
                    Hc[static_cast<std::size_t>(row + col * ldh)]);
            }
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown prefilled generic complex Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_gmres_d_cpu(
    int64_t n, const double *b, const double *x0, double a0, double a1,
    int64_t krylovdim, int64_t maxiter, double tol,
    tenet_native_matvec_d_cpu_fn matvec, void *ctx, double *x,
    double *residual, double *normres, int64_t *converged, int64_t *numops,
    int64_t *numiter) {
    if (n <= 0 || krylovdim <= 0 || maxiter < 0 || tol < 0.0 ||
        b == nullptr || x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic real GMRES inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::gmres<double>(
            n, krylovdim, maxiter, tol, b, x0, a0, a1, x, residual, normres,
            converged, numops, numiter,
            [&](const double *src, double *dst) {
                return matvec(n, src, dst, ctx);
            });
        if (status != 0) {
            set_error(status == 2 ? "generic real GMRES least-squares failed"
                                  : "generic real GMRES matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic real GMRES exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_gmres_z_cpu(
    int64_t n, const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0, tenet_native_complex64 a0,
    tenet_native_complex64 a1, int64_t krylovdim, int64_t maxiter,
    double tol, tenet_native_matvec_z_cpu_fn matvec, void *ctx,
    tenet_native_complex64 *x, tenet_native_complex64 *residual,
    double *normres, int64_t *converged, int64_t *numops,
    int64_t *numiter) {
    if (n <= 0 || krylovdim <= 0 || maxiter < 0 || tol < 0.0 ||
        b == nullptr || x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic complex GMRES inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> src(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> dst(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::gmres<Complex64>(
            n, krylovdim, maxiter, tol, bc.data(), x0c.data(),
            to_cpp_complex(a0), to_cpp_complex(a1), xc.data(), rc.data(),
            normres, converged, numops, numiter,
            [&](const Complex64 *src_cpp, Complex64 *dst_cpp) {
                for (int64_t i = 0; i < n; ++i) {
                    src[static_cast<std::size_t>(i)] =
                        to_c_complex(src_cpp[i]);
                }
                const int st = matvec(n, src.data(), dst.data(), ctx);
                if (st != 0) {
                    return st;
                }
                for (int64_t i = 0; i < n; ++i) {
                    dst_cpp[i] = to_cpp_complex(dst[static_cast<std::size_t>(i)]);
                }
                return 0;
            });
        if (status != 0) {
            set_error(status == 2 ? "generic complex GMRES least-squares failed"
                                  : "generic complex GMRES matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic complex GMRES exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_cg_d_cpu(
    int64_t n, const double *b, const double *x0, double a0, double a1,
    int64_t maxiter, double tol, tenet_native_matvec_d_cpu_fn matvec,
    void *ctx, double *x, double *residual, double *normres,
    int64_t *converged, int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || b == nullptr ||
        x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic real CG inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::cg<double>(
            n, maxiter, tol, b, x0, a0, a1, x, residual, normres, converged,
            numops, numiter,
            [&](const double *src, double *dst) {
                return matvec(n, src, dst, ctx);
            });
        if (status != 0) {
            set_error("generic real CG matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic real CG exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_cg_z_cpu(
    int64_t n, const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0, tenet_native_complex64 a0,
    tenet_native_complex64 a1, int64_t maxiter, double tol,
    tenet_native_matvec_z_cpu_fn matvec, void *ctx, tenet_native_complex64 *x,
    tenet_native_complex64 *residual, double *normres, int64_t *converged,
    int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || b == nullptr ||
        x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic complex CG inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> src(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> dst(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::cg<Complex64>(
            n, maxiter, tol, bc.data(), x0c.data(), to_cpp_complex(a0),
            to_cpp_complex(a1), xc.data(), rc.data(), normres, converged,
            numops, numiter,
            [&](const Complex64 *src_cpp, Complex64 *dst_cpp) {
                for (int64_t i = 0; i < n; ++i) {
                    src[static_cast<std::size_t>(i)] =
                        to_c_complex(src_cpp[i]);
                }
                const int st = matvec(n, src.data(), dst.data(), ctx);
                if (st != 0) {
                    return st;
                }
                for (int64_t i = 0; i < n; ++i) {
                    dst_cpp[i] = to_cpp_complex(dst[static_cast<std::size_t>(i)]);
                }
                return 0;
            });
        if (status != 0) {
            set_error("generic complex CG matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic complex CG exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_bicgstab_d_cpu(
    int64_t n, const double *b, const double *x0, double a0, double a1,
    int64_t maxiter, double tol, tenet_native_matvec_d_cpu_fn matvec,
    void *ctx, double *x, double *residual, double *normres,
    int64_t *converged, int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || b == nullptr ||
        x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic real BiCGStab inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::bicgstab<double>(
            n, maxiter, tol, b, x0, a0, a1, x, residual, normres, converged,
            numops, numiter,
            [&](const double *src, double *dst) {
                return matvec(n, src, dst, ctx);
            });
        if (status != 0) {
            set_error("generic real BiCGStab matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic real BiCGStab exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_bicgstab_z_cpu(
    int64_t n, const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0, tenet_native_complex64 a0,
    tenet_native_complex64 a1, int64_t maxiter, double tol,
    tenet_native_matvec_z_cpu_fn matvec, void *ctx, tenet_native_complex64 *x,
    tenet_native_complex64 *residual, double *normres, int64_t *converged,
    int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || b == nullptr ||
        x0 == nullptr || matvec == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr) {
        set_error("invalid generic complex BiCGStab inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> src(static_cast<std::size_t>(n));
        std::vector<tenet_native_complex64> dst(static_cast<std::size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::bicgstab<Complex64>(
            n, maxiter, tol, bc.data(), x0c.data(), to_cpp_complex(a0),
            to_cpp_complex(a1), xc.data(), rc.data(), normres, converged,
            numops, numiter,
            [&](const Complex64 *src_cpp, Complex64 *dst_cpp) {
                for (int64_t i = 0; i < n; ++i) {
                    src[static_cast<std::size_t>(i)] =
                        to_c_complex(src_cpp[i]);
                }
                const int st = matvec(n, src.data(), dst.data(), ctx);
                if (st != 0) {
                    return st;
                }
                for (int64_t i = 0; i < n; ++i) {
                    dst_cpp[i] = to_cpp_complex(dst[static_cast<std::size_t>(i)]);
                }
                return 0;
            });
        if (status != 0) {
            set_error("generic complex BiCGStab matvec failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown generic complex BiCGStab exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_dense_d_cpu(
    int64_t n, const double *A, int64_t lda, const double *x0, int64_t max_k,
    double breakdown_tol, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm, int64_t *numops) {
    if (!check_generic_krylov_common(n, max_k, breakdown_tol) ||
        A == nullptr || x0 == nullptr || V == nullptr || H == nullptr ||
        beta == nullptr || m == nullptr || final_resnorm == nullptr ||
        numops == nullptr || lda < n || ldv < n || ldh < max_k + 1) {
        set_error("invalid dense real Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::arnoldi<double>(
            n, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
            final_resnorm, numops,
            [&](const double *src, double *dst) {
                dense_matvec_real(n, A, lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense real Arnoldi failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense real Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_dense_z_cpu(
    int64_t n, const tenet_native_complex64 *A, int64_t lda,
    const tenet_native_complex64 *x0, int64_t max_k, double breakdown_tol,
    tenet_native_complex64 *V, int64_t ldv, tenet_native_complex64 *H,
    int64_t ldh, double *beta, int64_t *m, double *final_resnorm,
    int64_t *numops) {
    if (!check_generic_krylov_common(n, max_k, breakdown_tol) ||
        A == nullptr || x0 == nullptr || V == nullptr || H == nullptr ||
        beta == nullptr || m == nullptr || final_resnorm == nullptr ||
        numops == nullptr || lda < n || ldv < n || ldh < max_k + 1) {
        set_error("invalid dense complex Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> Ac(static_cast<std::size_t>(lda) *
                                  static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> Vc(static_cast<std::size_t>(ldv) *
                                  static_cast<std::size_t>(max_k + 1));
        std::vector<Complex64> Hc(static_cast<std::size_t>(ldh) *
                                  static_cast<std::size_t>(max_k));
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                Ac[static_cast<std::size_t>(row + col * lda)] =
                    to_cpp_complex(A[row + col * lda]);
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::arnoldi<Complex64>(
            n, max_k, breakdown_tol, x0c.data(), Vc.data(), ldv, Hc.data(),
            ldh, beta, m, final_resnorm, numops,
            [&](const Complex64 *src, Complex64 *dst) {
                dense_matvec_complex(n, Ac.data(), lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense complex Arnoldi failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t col = 0; col < max_k + 1; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                V[row + col * ldv] = to_c_complex(Vc[static_cast<std::size_t>(
                    row + col * ldv)]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                H[row + col * ldh] = to_c_complex(Hc[static_cast<std::size_t>(
                    row + col * ldh)]);
            }
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense complex Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_prefilled_dense_d_cpu(
    int64_t n, const double *A, int64_t lda, const double *initial_V,
    int64_t initial_ldv, int64_t initial_cols, const double *initial_H,
    int64_t initial_ldh, int64_t completed_cols, int64_t max_k,
    double breakdown_tol, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm, int64_t *numops) {
    if (!check_prefilled_krylov_common(n, initial_cols, completed_cols, max_k,
                                       breakdown_tol) ||
        A == nullptr || initial_V == nullptr || initial_H == nullptr ||
        V == nullptr || H == nullptr || beta == nullptr || m == nullptr ||
        final_resnorm == nullptr || numops == nullptr || lda < n ||
        initial_ldv < n || initial_ldh < max_k + 1 || ldv < n ||
        ldh < max_k + 1) {
        set_error("invalid prefilled dense real Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::arnoldi_prefilled<double>(
            n, max_k, breakdown_tol, initial_V, initial_ldv, initial_cols,
            initial_H, initial_ldh, completed_cols, V, ldv, H, ldh, beta, m,
            final_resnorm, numops,
            [&](const double *src, double *dst) {
                dense_matvec_real(n, A, lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("prefilled dense real Arnoldi failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown prefilled dense real Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_arnoldi_prefilled_dense_z_cpu(
    int64_t n, const tenet_native_complex64 *A, int64_t lda,
    const tenet_native_complex64 *initial_V, int64_t initial_ldv,
    int64_t initial_cols, const tenet_native_complex64 *initial_H,
    int64_t initial_ldh, int64_t completed_cols, int64_t max_k,
    double breakdown_tol, tenet_native_complex64 *V, int64_t ldv,
    tenet_native_complex64 *H, int64_t ldh, double *beta, int64_t *m,
    double *final_resnorm, int64_t *numops) {
    if (!check_prefilled_krylov_common(n, initial_cols, completed_cols, max_k,
                                       breakdown_tol) ||
        A == nullptr || initial_V == nullptr || initial_H == nullptr ||
        V == nullptr || H == nullptr || beta == nullptr || m == nullptr ||
        final_resnorm == nullptr || numops == nullptr || lda < n ||
        initial_ldv < n || initial_ldh < max_k + 1 || ldv < n ||
        ldh < max_k + 1) {
        set_error("invalid prefilled dense complex Arnoldi inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> Ac(static_cast<std::size_t>(lda) *
                                  static_cast<std::size_t>(n));
        std::vector<Complex64> initial_Vc(
            static_cast<std::size_t>(initial_ldv) *
            static_cast<std::size_t>(initial_cols));
        std::vector<Complex64> initial_Hc(
            static_cast<std::size_t>(initial_ldh) *
            static_cast<std::size_t>(max_k));
        std::vector<Complex64> Vc(static_cast<std::size_t>(ldv) *
                                  static_cast<std::size_t>(max_k + 1));
        std::vector<Complex64> Hc(static_cast<std::size_t>(ldh) *
                                  static_cast<std::size_t>(max_k));
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                Ac[static_cast<std::size_t>(row + col * lda)] =
                    to_cpp_complex(A[row + col * lda]);
            }
        }
        for (int64_t col = 0; col < initial_cols; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                initial_Vc[static_cast<std::size_t>(row +
                                                    col * initial_ldv)] =
                    to_cpp_complex(initial_V[row + col * initial_ldv]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                initial_Hc[static_cast<std::size_t>(row +
                                                    col * initial_ldh)] =
                    to_cpp_complex(initial_H[row + col * initial_ldh]);
            }
        }
        const int status = tenet_native_krylov::arnoldi_prefilled<Complex64>(
            n, max_k, breakdown_tol, initial_Vc.data(), initial_ldv,
            initial_cols, initial_Hc.data(), initial_ldh, completed_cols,
            Vc.data(), ldv, Hc.data(), ldh, beta, m, final_resnorm, numops,
            [&](const Complex64 *src, Complex64 *dst) {
                dense_matvec_complex(n, Ac.data(), lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("prefilled dense complex Arnoldi failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t col = 0; col < max_k + 1; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                V[row + col * ldv] = to_c_complex(
                    Vc[static_cast<std::size_t>(row + col * ldv)]);
            }
        }
        for (int64_t col = 0; col < max_k; ++col) {
            for (int64_t row = 0; row < max_k + 1; ++row) {
                H[row + col * ldh] = to_c_complex(
                    Hc[static_cast<std::size_t>(row + col * ldh)]);
            }
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown prefilled dense complex Arnoldi exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_gmres_dense_d_cpu(
    int64_t n, const double *A, int64_t lda, const double *b,
    const double *x0, double a0, double a1, int64_t krylovdim,
    int64_t maxiter, double tol, double *x, double *residual,
    double *normres, int64_t *converged, int64_t *numops,
    int64_t *numiter) {
    if (n <= 0 || krylovdim <= 0 || maxiter < 0 || tol < 0.0 ||
        A == nullptr || b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense real GMRES inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::gmres<double>(
            n, krylovdim, maxiter, tol, b, x0, a0, a1, x, residual, normres,
            converged, numops, numiter,
            [&](const double *src, double *dst) {
                dense_matvec_real(n, A, lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error(status == 2 ? "dense real GMRES least-squares failed"
                                  : "dense real GMRES failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense real GMRES exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_gmres_dense_z_cpu(
    int64_t n, const tenet_native_complex64 *A, int64_t lda,
    const tenet_native_complex64 *b, const tenet_native_complex64 *x0,
    tenet_native_complex64 a0, tenet_native_complex64 a1, int64_t krylovdim,
    int64_t maxiter, double tol, tenet_native_complex64 *x,
    tenet_native_complex64 *residual, double *normres, int64_t *converged,
    int64_t *numops, int64_t *numiter) {
    if (n <= 0 || krylovdim <= 0 || maxiter < 0 || tol < 0.0 ||
        A == nullptr || b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense complex GMRES inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> Ac(static_cast<std::size_t>(lda) *
                                  static_cast<std::size_t>(n));
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                Ac[static_cast<std::size_t>(row + col * lda)] =
                    to_cpp_complex(A[row + col * lda]);
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::gmres<Complex64>(
            n, krylovdim, maxiter, tol, bc.data(), x0c.data(),
            to_cpp_complex(a0), to_cpp_complex(a1), xc.data(), rc.data(),
            normres, converged, numops, numiter,
            [&](const Complex64 *src, Complex64 *dst) {
                dense_matvec_complex(n, Ac.data(), lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error(status == 2 ? "dense complex GMRES least-squares failed"
                                  : "dense complex GMRES failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense complex GMRES exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_cg_dense_d_cpu(
    int64_t n, const double *A, int64_t lda, const double *b,
    const double *x0, double a0, double a1, int64_t maxiter, double tol,
    double *x, double *residual, double *normres, int64_t *converged,
    int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || A == nullptr ||
        b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense real CG inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::cg<double>(
            n, maxiter, tol, b, x0, a0, a1, x, residual, normres, converged,
            numops, numiter,
            [&](const double *src, double *dst) {
                dense_matvec_real(n, A, lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense real CG failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense real CG exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_cg_dense_z_cpu(
    int64_t n, const tenet_native_complex64 *A, int64_t lda,
    const tenet_native_complex64 *b, const tenet_native_complex64 *x0,
    tenet_native_complex64 a0, tenet_native_complex64 a1, int64_t maxiter,
    double tol, tenet_native_complex64 *x, tenet_native_complex64 *residual,
    double *normres, int64_t *converged, int64_t *numops,
    int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || A == nullptr ||
        b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense complex CG inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> Ac(static_cast<std::size_t>(lda) *
                                  static_cast<std::size_t>(n));
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                Ac[static_cast<std::size_t>(row + col * lda)] =
                    to_cpp_complex(A[row + col * lda]);
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::cg<Complex64>(
            n, maxiter, tol, bc.data(), x0c.data(), to_cpp_complex(a0),
            to_cpp_complex(a1), xc.data(), rc.data(), normres, converged,
            numops, numiter,
            [&](const Complex64 *src, Complex64 *dst) {
                dense_matvec_complex(n, Ac.data(), lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense complex CG failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense complex CG exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_bicgstab_dense_d_cpu(
    int64_t n, const double *A, int64_t lda, const double *b,
    const double *x0, double a0, double a1, int64_t maxiter, double tol,
    double *x, double *residual, double *normres, int64_t *converged,
    int64_t *numops, int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || A == nullptr ||
        b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense real BiCGStab inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const int status = tenet_native_krylov::bicgstab<double>(
            n, maxiter, tol, b, x0, a0, a1, x, residual, normres, converged,
            numops, numiter,
            [&](const double *src, double *dst) {
                dense_matvec_real(n, A, lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense real BiCGStab failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense real BiCGStab exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_krylov_bicgstab_dense_z_cpu(
    int64_t n, const tenet_native_complex64 *A, int64_t lda,
    const tenet_native_complex64 *b, const tenet_native_complex64 *x0,
    tenet_native_complex64 a0, tenet_native_complex64 a1, int64_t maxiter,
    double tol, tenet_native_complex64 *x, tenet_native_complex64 *residual,
    double *normres, int64_t *converged, int64_t *numops,
    int64_t *numiter) {
    if (n <= 0 || maxiter < 0 || tol < 0.0 || A == nullptr ||
        b == nullptr || x0 == nullptr || x == nullptr ||
        residual == nullptr || normres == nullptr || converged == nullptr ||
        numops == nullptr || numiter == nullptr || lda < n) {
        set_error("invalid dense complex BiCGStab inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<Complex64> Ac(static_cast<std::size_t>(lda) *
                                  static_cast<std::size_t>(n));
        std::vector<Complex64> bc(static_cast<std::size_t>(n));
        std::vector<Complex64> x0c(static_cast<std::size_t>(n));
        std::vector<Complex64> xc(static_cast<std::size_t>(n));
        std::vector<Complex64> rc(static_cast<std::size_t>(n));
        for (int64_t col = 0; col < n; ++col) {
            for (int64_t row = 0; row < n; ++row) {
                Ac[static_cast<std::size_t>(row + col * lda)] =
                    to_cpp_complex(A[row + col * lda]);
            }
        }
        for (int64_t i = 0; i < n; ++i) {
            bc[static_cast<std::size_t>(i)] = to_cpp_complex(b[i]);
            x0c[static_cast<std::size_t>(i)] = to_cpp_complex(x0[i]);
        }
        const int status = tenet_native_krylov::bicgstab<Complex64>(
            n, maxiter, tol, bc.data(), x0c.data(), to_cpp_complex(a0),
            to_cpp_complex(a1), xc.data(), rc.data(), normres, converged,
            numops, numiter,
            [&](const Complex64 *src, Complex64 *dst) {
                dense_matvec_complex(n, Ac.data(), lda, src, dst);
                return 0;
            });
        if (status != 0) {
            set_error("dense complex BiCGStab failed");
            return TENET_NATIVE_BACKEND_ERROR;
        }
        for (int64_t i = 0; i < n; ++i) {
            x[i] = to_c_complex(xc[static_cast<std::size_t>(i)]);
            residual[i] = to_c_complex(rc[static_cast<std::size_t>(i)]);
        }
        set_error("success");
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown dense complex BiCGStab exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_raw_two_layer_apply_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x, int transpose, double *y) {
    const int64_t len = chi * chi;
    if (!check_raw_two_layer_inputs(chi, phys, Aup, Adn, x, transpose, y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        two_layer_apply_gemm(chi, phys, Aup, Adn, x, tmp.data(), y, transpose);
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_raw_transfer_op_d_cpu(
    int64_t phys, int64_t chi, const double *W, const double *O,
    const double *x, double *y) {
    const int64_t len = chi * chi;
    if (!check_raw_transfer_op_inputs(phys, chi, W, O, x, y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        std::vector<double> work(static_cast<std::size_t>(len));
        raw_transfer_op_gemm(chi, phys, W, O, x, tmp.data(), work.data(), y);
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_raw_rowmajor_transfer_d_cpu(int64_t d, int64_t D,
                                                        const double *W,
                                                        const double *x,
                                                        double *y) {
    const int64_t len = D * D;
    if (!check_raw_rowmajor_inputs(d, D, W, x, y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        std::fill(y, y + len, 0.0);
        for (int64_t s = 0; s < d; ++s) {
            const double *As = W + s * len;
            rowmajor_transfer_apply_forward_slice(D, As, x, tmp.data(), y);
        }
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_raw_rowmajor_transfer_adj_d_cpu(int64_t d,
                                                            int64_t D,
                                                            const double *W,
                                                            const double *x,
                                                            double *y) {
    const int64_t len = D * D;
    if (!check_raw_rowmajor_inputs(d, D, W, x, y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        std::fill(y, y + len, 0.0);
        for (int64_t s = 0; s < d; ++s) {
            const double *As = W + s * len;
            rowmajor_transfer_apply_adjoint_slice(D, As, x, tmp.data(), y);
        }
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_raw_rowmajor_transfer_op_d_cpu(
    int64_t d, int64_t D, const double *W, const double *O,
    const double *x, double *y) {
    const int64_t len = D * D;
    if (!check_raw_rowmajor_transfer_op_inputs(d, D, W, O, x, y)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        std::fill(y, y + len, 0.0);
        for (int64_t s = 0; s < d; ++s) {
            const double *As = W + s * len;
            for (int64_t t = 0; t < d; ++t) {
                const double coeff = O[s * d + t];
                if (coeff == 0.0) {
                    continue;
                }
                const double *At = W + t * len;
                std::fill(tmp.data(), tmp.data() + len, 0.0);
                for (int64_t i = 0; i < D; ++i) {
                    for (int64_t j = 0; j < D; ++j) {
                        double acc = 0.0;
                        for (int64_t k = 0; k < D; ++k) {
                            acc += At[i * D + k] * x[k * D + j];
                        }
                        tmp[i * D + j] = acc;
                    }
                }
                for (int64_t i = 0; i < D; ++i) {
                    for (int64_t l = 0; l < D; ++l) {
                        double acc = 0.0;
                        for (int64_t j = 0; j < D; ++j) {
                            acc += tmp[i * D + j] * As[l * D + j];
                        }
                        y[i * D + l] += coeff * acc;
                    }
                }
            }
        }
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_arnoldi_two_layer_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    double *V, int64_t ldv, double *H, int64_t ldh, double *beta, int64_t *m,
    double *final_resnorm) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) ||
        !check_arnoldi_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh,
                              beta, m, final_resnorm)) {
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(chi) *
                                static_cast<std::size_t>(chi));
        return arnoldi_driver(
            len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
            final_resnorm,
            [&](const double *src, double *dst) {
                two_layer_apply_gemm(chi, phys, Aup, Adn, src, tmp.data(),
                                     dst, transpose);
            });
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_arnoldi_two_layer_ritz_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    int64_t nvalues, double *lambda_real, double *lambda_imag, int64_t *m) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || x0 == nullptr ||
        max_k <= 0 || max_k > len || breakdown_tol < 0.0 ||
        nvalues <= 0 || lambda_real == nullptr || lambda_imag == nullptr ||
        m == nullptr) {
        set_error("invalid restarted two-layer Ritz inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(chi) *
                                static_cast<std::size_t>(chi));
        return restarted_arnoldi_ritz_values(
            len, max_k, breakdown_tol, x0, nvalues, lambda_real, lambda_imag,
            m,
            [&](const double *src, double *dst) {
                two_layer_apply_gemm(chi, phys, Aup, Adn, src, tmp.data(),
                                     dst, transpose);
            });
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_arnoldi_projected_two_layer_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *rho, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || rho == nullptr ||
        !check_arnoldi_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh,
                              beta, m, final_resnorm)) {
        if (rho == nullptr) {
            set_error("null rho pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(chi) *
                                static_cast<std::size_t>(chi));
        return arnoldi_driver(
            len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta,
            m, final_resnorm,
            [&](const double *src, double *dst) {
                const Leg3ColMajorTwoLayerLayout layout(chi, phys);
                const auto mode =
                    transpose == 0 ? tenet_native_shared::TwoLayerMode::Adjoint
                                   : tenet_native_shared::TwoLayerMode::Forward;
                tenet_native_shared::projected_two_layer_apply(
                    layout, phys, Aup, Adn, rho, src, tmp.data(), dst, mode);
            });
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

void qprojected_two_layer_apply(int64_t chi, int64_t phys,
                                const double *Aup, const double *Adn,
                                const double *rho, const double *x,
                                double *tmp, double *y, int transpose) {
    const int64_t len = chi * chi;
    std::vector<double> work(static_cast<std::size_t>(len));
    const Leg3ColMajorTwoLayerLayout layout(chi, phys);
    const auto mode = transpose == 0 ? tenet_native_shared::TwoLayerMode::Adjoint
                                     : tenet_native_shared::TwoLayerMode::Forward;
    tenet_native_shared::qprojected_two_layer_apply(
        layout, phys, Aup, Adn, rho, x, tmp, work.data(), y, mode);
}

extern "C" int tenet_native_arnoldi_qprojected_two_layer_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *rho, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || rho == nullptr ||
        !check_arnoldi_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh,
                              beta, m, final_resnorm)) {
        if (rho == nullptr) {
            set_error("null rho pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> tmp(static_cast<std::size_t>(len));
        return arnoldi_driver(
            len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta,
            m, final_resnorm,
            [&](const double *src, double *dst) {
                qprojected_two_layer_apply(chi, phys, Aup, Adn, rho, src,
                                           tmp.data(), dst, transpose);
            });
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_arnoldi_three_layer_leg4_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *M, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *V, int64_t ldv, double *H, int64_t ldh,
    double *beta, int64_t *m, double *final_resnorm) {
    const int64_t len = chi * phys * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || M == nullptr ||
        !check_arnoldi_common(len, max_k, breakdown_tol, x0, V, ldv, H, ldh,
                              beta, m, final_resnorm)) {
        if (M == nullptr) {
            set_error("null M pointer");
        }
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const std::size_t len2 =
            static_cast<std::size_t>(chi) * static_cast<std::size_t>(chi);
        const std::size_t blocks =
            static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys);
        std::vector<double> pair_work(blocks * len2);
        std::vector<double> accum_work(blocks * len2);
        return arnoldi_driver(
            len, max_k, breakdown_tol, x0, V, ldv, H, ldh, beta, m,
            final_resnorm,
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, Aup, Adn, M, src, pair_work.data(),
                    accum_work.data(), dst, transpose);
            });
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_dominant_two_layer_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    double *y, double *lambda) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || x0 == nullptr ||
        y == nullptr || lambda == nullptr || max_k <= 0 || max_k > len ||
        breakdown_tol < 0.0) {
        set_error("invalid dominant two-layer inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> work(static_cast<std::size_t>(len));
        std::vector<double> fy(static_cast<std::size_t>(len));
        int status = dominant_arnoldi_vector(
            len, max_k, breakdown_tol, x0, y,
            [&](const double *src, double *dst) {
                two_layer_apply_gemm(chi, phys, Aup, Adn, src, work.data(),
                                     dst, transpose);
            });
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        two_layer_apply_gemm(chi, phys, Aup, Adn, y, work.data(), fy.data(),
                             transpose);
        const double denom = dot(len, y, y);
        *lambda = denom > 0.0 ? dot(len, y, fy.data()) / denom : 0.0;
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_smallest_real_two_layer_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *x0, int64_t max_k, double breakdown_tol, int transpose,
    double *y, double *lambda) {
    const int64_t len = chi * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || x0 == nullptr ||
        y == nullptr || lambda == nullptr || max_k <= 0 || max_k > len ||
        breakdown_tol < 0.0) {
        set_error("invalid smallest-real two-layer inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> work(static_cast<std::size_t>(len));
        std::vector<double> fy(static_cast<std::size_t>(len));
        int status = dominant_arnoldi_vector(
            len, max_k, breakdown_tol, x0, y,
            [&](const double *src, double *dst) {
                two_layer_apply_gemm(chi, phys, Aup, Adn, src, work.data(),
                                     dst, transpose);
            },
            RitzTarget::SmallestReal);
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        two_layer_apply_gemm(chi, phys, Aup, Adn, y, work.data(), fy.data(),
                             transpose);
        const double denom = dot(len, y, y);
        *lambda = denom > 0.0 ? dot(len, y, fy.data()) / denom : 0.0;
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_dominant_three_layer_leg4_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *M, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *y, double *lambda) {
    const int64_t len = chi * phys * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || M == nullptr ||
        x0 == nullptr || y == nullptr || lambda == nullptr || max_k <= 0 ||
        max_k > len || breakdown_tol < 0.0) {
        set_error("invalid dominant three-layer inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const std::size_t len2 =
            static_cast<std::size_t>(chi) * static_cast<std::size_t>(chi);
        const std::size_t blocks =
            static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys);
        std::vector<double> pair_work(blocks * len2);
        std::vector<double> accum_work(blocks * len2);
        std::vector<double> fy(static_cast<std::size_t>(len));
        int status = dominant_arnoldi_vector(
            len, max_k, breakdown_tol, x0, y,
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, Aup, Adn, M, src, pair_work.data(),
                    accum_work.data(), dst, transpose);
            });
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        three_layer_apply_gemm_factored(chi, phys, Aup, Adn, M, y,
                                        pair_work.data(), accum_work.data(),
                                        fy.data(), transpose);
        const double denom = dot(len, y, y);
        *lambda = denom > 0.0 ? dot(len, y, fy.data()) / denom : 0.0;
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_smallest_real_three_layer_leg4_d_cpu(
    int64_t chi, int64_t phys, const double *Aup, const double *Adn,
    const double *M, const double *x0, int64_t max_k, double breakdown_tol,
    int transpose, double *y, double *lambda) {
    const int64_t len = chi * phys * chi;
    if (!check_tensor_common(chi, phys, Aup, Adn) || M == nullptr ||
        x0 == nullptr || y == nullptr || lambda == nullptr || max_k <= 0 ||
        max_k > len || breakdown_tol < 0.0) {
        set_error("invalid smallest-real three-layer inputs");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        const std::size_t len2 =
            static_cast<std::size_t>(chi) * static_cast<std::size_t>(chi);
        const std::size_t blocks =
            static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys);
        std::vector<double> pair_work(blocks * len2);
        std::vector<double> accum_work(blocks * len2);
        std::vector<double> fy(static_cast<std::size_t>(len));
        int status = dominant_arnoldi_vector(
            len, max_k, breakdown_tol, x0, y,
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, Aup, Adn, M, src, pair_work.data(),
                    accum_work.data(), dst, transpose);
            },
            RitzTarget::SmallestReal);
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        three_layer_apply_gemm_factored(chi, phys, Aup, Adn, M, y,
                                        pair_work.data(), accum_work.data(),
                                        fy.data(), transpose);
        const double denom = dot(len, y, y);
        *lambda = denom > 0.0 ? dot(len, y, fy.data()) / denom : 0.0;
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

int tenet_native_ising_vumps_step_impl_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double *err, int check_residual, double residual_tol) {
    if (!check_tensor_common(chi, phys, AL, AR) || M == nullptr ||
        C == nullptr || FL == nullptr || FR == nullptr || err == nullptr) {
        set_error("null pointer in native Ising VUMPS step");
        return TENET_NATIVE_INVALID_VALUE;
    }
    const int64_t len2 = chi * chi;
    const int64_t len3 = chi * phys * chi;
    if (max_k <= 0 || max_k > std::max(len2, len3) ||
        breakdown_tol < 0.0 || len3 > INT_MAX ||
        (check_residual && residual_tol < 0.0)) {
        set_error("invalid native Ising VUMPS step dimensions");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        std::vector<double> AC(static_cast<std::size_t>(len3));
        std::vector<double> FL_new(static_cast<std::size_t>(len3));
        std::vector<double> FR_new(static_cast<std::size_t>(len3));
        std::vector<double> C_new(static_cast<std::size_t>(len2));
        std::vector<double> tmp(static_cast<std::size_t>(len2));
        const std::size_t three_blocks =
            static_cast<std::size_t>(phys) * static_cast<std::size_t>(phys);
        std::vector<double> three_pair(three_blocks *
                                       static_cast<std::size_t>(len2));
        std::vector<double> three_accum(three_blocks *
                                        static_cast<std::size_t>(len2));
        std::vector<double> Mp(static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys) *
                               static_cast<std::size_t>(phys));
        permute_m_for_ac(phys, M, Mp.data());

        alc_to_ac(chi, phys, AL, C, AC.data());

        auto phase_start = SteadyClock::now();
        int status = dominant_arnoldi_vector(
            len3, std::min(max_k, len3), breakdown_tol, FL, FL_new.data(),
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, AL, AL, M, src, three_pair.data(),
                    three_accum.data(), dst, 0);
            });
        profile_phase("FL", chi, phys, max_k, seconds_since(phase_start));
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        if (check_residual &&
            !check_last_dominant_residual("FLmap", residual_tol)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        phase_start = SteadyClock::now();
        status = dominant_arnoldi_vector(
            len3, std::min(max_k, len3), breakdown_tol, FR, FR_new.data(),
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, AR, AR, M, src, three_pair.data(),
                    three_accum.data(), dst, 1);
            });
        profile_phase("FR", chi, phys, max_k, seconds_since(phase_start));
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        if (check_residual &&
            !check_last_dominant_residual("FRmap", residual_tol)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        phase_start = SteadyClock::now();
        status = dominant_arnoldi_vector(
            len3, std::min(max_k, len3), breakdown_tol, AC.data(), AC.data(),
            [&](const double *src, double *dst) {
                three_layer_apply_gemm_factored(
                    chi, phys, FL_new.data(), FR_new.data(), Mp.data(), src,
                    three_pair.data(), three_accum.data(), dst, 0);
            });
        profile_phase("AC", chi, phys, max_k, seconds_since(phase_start));
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        if (check_residual &&
            !check_last_dominant_residual("ACmap", residual_tol)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        phase_start = SteadyClock::now();
        status = dominant_arnoldi_vector(
            len2, std::min(max_k, len2), breakdown_tol, C, C_new.data(),
            [&](const double *src, double *dst) {
                two_layer_apply_gemm(chi, phys, FL_new.data(), FR_new.data(),
                                     src, tmp.data(), dst, 0);
            });
        profile_phase("C", chi, phys, max_k, seconds_since(phase_start));
        if (status != TENET_NATIVE_SUCCESS) {
            return status;
        }
        if (check_residual &&
            !check_last_dominant_residual("Cmap", residual_tol)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        phase_start = SteadyClock::now();
        if (!acc_to_alar(chi, phys, AC.data(), C_new.data(), AL, AR, err)) {
            profile_phase("ACCtoALAR", chi, phys, max_k,
                          seconds_since(phase_start));
            return TENET_NATIVE_BACKEND_ERROR;
        }
        profile_phase("ACCtoALAR", chi, phys, max_k,
                      seconds_since(phase_start));
        std::copy(C_new.begin(), C_new.end(), C);
        std::copy(FL_new.begin(), FL_new.end(), FL);
        std::copy(FR_new.begin(), FR_new.end(), FR);
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}

extern "C" int tenet_native_ising_vumps_step_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double *err) {
    return tenet_native_ising_vumps_step_impl_d_cpu(
        chi, phys, M, AL, AR, C, FL, FR, max_k, breakdown_tol, err, 0, 0.0);
}

extern "C" int tenet_native_ising_vumps_step_checked_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t max_k, double breakdown_tol,
    double residual_tol, double *err) {
    return tenet_native_ising_vumps_step_impl_d_cpu(
        chi, phys, M, AL, AR, C, FL, FR, max_k, breakdown_tol, err, 1,
        residual_tol);
}

int tenet_native_ising_vumps_run_impl_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double *err, int64_t *iterations, int *converged, int check_residual,
    double residual_tol) {
    if (err == nullptr || iterations == nullptr || converged == nullptr) {
        set_error("null pointer in native Ising VUMPS run");
        return TENET_NATIVE_INVALID_VALUE;
    }
    if (miniter < 0 || maxiter < 0 || miniter > maxiter || tol < 0.0 ||
        (check_residual && residual_tol < 0.0)) {
        set_error("invalid native Ising VUMPS run iteration controls");
        return TENET_NATIVE_INVALID_VALUE;
    }
    *err = 0.0;
    *iterations = 0;
    *converged = maxiter == 0 ? 1 : 0;
    for (int64_t iter = 1; iter <= maxiter; ++iter) {
        const int status = tenet_native_ising_vumps_step_impl_d_cpu(
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

extern "C" int tenet_native_ising_vumps_run_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double *err, int64_t *iterations, int *converged) {
    return tenet_native_ising_vumps_run_impl_d_cpu(
        chi, phys, M, AL, AR, C, FL, FR, arnoldi_max_k, breakdown_tol, tol,
        miniter, maxiter, err, iterations, converged, 0, 0.0);
}

extern "C" int tenet_native_ising_vumps_run_checked_d_cpu(
    int64_t chi, int64_t phys, const double *M, double *AL, double *AR,
    double *C, double *FL, double *FR, int64_t arnoldi_max_k,
    double breakdown_tol, double tol, int64_t miniter, int64_t maxiter,
    double residual_tol, double *err, int64_t *iterations, int *converged) {
    return tenet_native_ising_vumps_run_impl_d_cpu(
        chi, phys, M, AL, AR, C, FL, FR, arnoldi_max_k, breakdown_tol, tol,
        miniter, maxiter, err, iterations, converged, 1, residual_tol);
}

extern "C" int tenet_native_acc_to_alar_d_cpu(
    int64_t chi, int64_t phys, const double *AC, const double *C, double *AL,
    double *AR, double *err) {
    if (chi <= 0 || phys <= 0 || AC == nullptr || C == nullptr ||
        AL == nullptr || AR == nullptr || err == nullptr) {
        set_error("invalid native ACCtoALAR pointer or dimensions");
        return TENET_NATIVE_INVALID_VALUE;
    }
    try {
        if (!acc_to_alar(chi, phys, AC, C, AL, AR, err)) {
            return TENET_NATIVE_BACKEND_ERROR;
        }
        return TENET_NATIVE_SUCCESS;
    } catch (const std::bad_alloc &) {
        set_error("allocation failed");
        return TENET_NATIVE_ALLOCATION_FAILED;
    } catch (...) {
        set_error("unknown backend exception");
        return TENET_NATIVE_BACKEND_ERROR;
    }
}
