#ifndef TENET_NATIVE_KRYLOV_CORE_HPP
#define TENET_NATIVE_KRYLOV_CORE_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace tenet_native_krylov {

template <typename T> inline T conj_value(const T &x) { return x; }

template <typename T>
inline std::complex<T> conj_value(const std::complex<T> &x) {
    return std::conj(x);
}

template <typename Scalar>
inline double scalar_abs(const Scalar &x) {
    using std::abs;
    return abs(x);
}

template <typename Scalar>
Scalar dot(int64_t n, const Scalar *x, const Scalar *y) {
    Scalar acc = Scalar(0);
    for (int64_t i = 0; i < n; ++i) {
        acc += conj_value(x[i]) * y[i];
    }
    return acc;
}

template <typename Scalar>
double norm2(int64_t n, const Scalar *x) {
    const Scalar d = dot(n, x, x);
    return std::sqrt(std::max(0.0, static_cast<double>(std::real(d))));
}

template <typename Scalar>
void scal(int64_t n, Scalar alpha, Scalar *x) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

template <typename Scalar>
void axpy(int64_t n, Scalar alpha, const Scalar *x, Scalar *y) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

template <typename Scalar>
void copy(int64_t n, const Scalar *x, Scalar *y) {
    std::copy(x, x + n, y);
}

template <typename Scalar>
void fill_zero(int64_t n, Scalar *x) {
    std::fill(x, x + n, Scalar(0));
}

template <typename Scalar>
bool solve_dense_linear(int64_t n, std::vector<Scalar> &A,
                        std::vector<Scalar> &b,
                        std::vector<Scalar> &x) {
    x.assign(static_cast<std::size_t>(n), Scalar(0));
    if (n == 0) {
        return true;
    }
    const auto at = [n](int64_t row, int64_t col) -> std::size_t {
        return static_cast<std::size_t>(row + n * col);
    };
    const double eps = 64.0 * std::numeric_limits<double>::epsilon();
    for (int64_t k = 0; k < n; ++k) {
        int64_t pivot = k;
        double pivot_abs = scalar_abs(A[at(k, k)]);
        for (int64_t row = k + 1; row < n; ++row) {
            const double candidate = scalar_abs(A[at(row, k)]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot = row;
            }
        }
        if (pivot_abs <= eps) {
            A[at(k, k)] += Scalar(eps);
            pivot_abs = scalar_abs(A[at(k, k)]);
        }
        if (pivot_abs <= 0.0) {
            return false;
        }
        if (pivot != k) {
            for (int64_t col = k; col < n; ++col) {
                std::swap(A[at(k, col)], A[at(pivot, col)]);
            }
            std::swap(b[static_cast<std::size_t>(k)],
                      b[static_cast<std::size_t>(pivot)]);
        }
        const Scalar diag = A[at(k, k)];
        for (int64_t row = k + 1; row < n; ++row) {
            const Scalar factor = A[at(row, k)] / diag;
            A[at(row, k)] = Scalar(0);
            for (int64_t col = k + 1; col < n; ++col) {
                A[at(row, col)] -= factor * A[at(k, col)];
            }
            b[static_cast<std::size_t>(row)] -=
                factor * b[static_cast<std::size_t>(k)];
        }
    }
    for (int64_t i = n - 1; i >= 0; --i) {
        Scalar acc = b[static_cast<std::size_t>(i)];
        for (int64_t j = i + 1; j < n; ++j) {
            acc -= A[at(i, j)] * x[static_cast<std::size_t>(j)];
        }
        const Scalar diag = A[at(i, i)];
        if (scalar_abs(diag) <= 0.0) {
            return false;
        }
        x[static_cast<std::size_t>(i)] = acc / diag;
    }
    return true;
}

template <typename Scalar>
bool least_squares_qr_work(int64_t rows, int64_t cols, const Scalar *H,
                           int64_t ldh, Scalar beta,
                           std::vector<Scalar> &y,
                           std::vector<Scalar> &Q,
                           std::vector<Scalar> &R,
                           std::vector<Scalar> &rhs) {
    y.assign(static_cast<std::size_t>(cols), Scalar(0));
    if (cols == 0) {
        return true;
    }
    Q.assign(static_cast<std::size_t>(rows * cols), Scalar(0));
    R.assign(static_cast<std::size_t>(cols * cols), Scalar(0));
    rhs.assign(static_cast<std::size_t>(cols), Scalar(0));
    const auto ridx = [cols](int64_t row, int64_t col) -> std::size_t {
        return static_cast<std::size_t>(row + cols * col);
    };
    const auto qidx = [rows](int64_t row, int64_t col) -> std::size_t {
        return static_cast<std::size_t>(row + rows * col);
    };
    const double rank_tol = 64.0 * std::numeric_limits<double>::epsilon();
    for (int64_t j = 0; j < cols; ++j) {
        for (int64_t r = 0; r < rows; ++r) {
            Q[qidx(r, j)] = H[r + j * ldh];
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t i = 0; i < j; ++i) {
                Scalar rij = Scalar(0);
                for (int64_t r = 0; r < rows; ++r) {
                    rij += conj_value(Q[qidx(r, i)]) * Q[qidx(r, j)];
                }
                R[ridx(i, j)] += rij;
                for (int64_t r = 0; r < rows; ++r) {
                    Q[qidx(r, j)] -= rij * Q[qidx(r, i)];
                }
            }
        }
        const double rjj = norm2(rows, Q.data() + qidx(0, j));
        if (!(rjj > rank_tol) || !std::isfinite(rjj)) {
            return false;
        }
        R[ridx(j, j)] = Scalar(rjj);
        for (int64_t r = 0; r < rows; ++r) {
            Q[qidx(r, j)] /= Scalar(rjj);
        }
        rhs[static_cast<std::size_t>(j)] = conj_value(Q[qidx(0, j)]) * beta;
    }
    for (int64_t i = cols - 1; i >= 0; --i) {
        Scalar acc = rhs[static_cast<std::size_t>(i)];
        for (int64_t j = i + 1; j < cols; ++j) {
            acc -= R[ridx(i, j)] * y[static_cast<std::size_t>(j)];
        }
        const Scalar diag = R[ridx(i, i)];
        if (!(scalar_abs(diag) > rank_tol) || !std::isfinite(scalar_abs(diag))) {
            return false;
        }
        y[static_cast<std::size_t>(i)] = acc / diag;
    }
    return true;
}

template <typename Scalar>
bool least_squares_qr(int64_t rows, int64_t cols, const Scalar *H,
                      int64_t ldh, Scalar beta, std::vector<Scalar> &y) {
    std::vector<Scalar> Q;
    std::vector<Scalar> R;
    std::vector<Scalar> rhs;
    return least_squares_qr_work(rows, cols, H, ldh, beta, y, Q, R, rhs);
}

template <typename Scalar>
void gmres_small_residual(int64_t m, const Scalar *H, int64_t ldh,
                          Scalar beta, const std::vector<Scalar> &y,
                          std::vector<Scalar> &g) {
    g.assign(static_cast<std::size_t>(m + 1), Scalar(0));
    g[0] = beta;
    for (int64_t col = 0; col < m; ++col) {
        const Scalar yj = y[static_cast<std::size_t>(col)];
        for (int64_t row = 0; row <= m; ++row) {
            g[static_cast<std::size_t>(row)] -= H[row + col * ldh] * yj;
        }
    }
}

template <typename Scalar>
double gmres_small_resnorm(int64_t m, const Scalar *H, int64_t ldh,
                           Scalar beta, const std::vector<Scalar> &y,
                           std::vector<Scalar> &g) {
    gmres_small_residual(m, H, ldh, beta, y, g);
    return norm2(m + 1, g.data());
}

template <typename Scalar>
void gmres_project_residual(int64_t n, int64_t m, const Scalar *V,
                            int64_t ldv, const Scalar *H, int64_t ldh,
                            Scalar beta, const std::vector<Scalar> &y,
                            Scalar *residual, double *normres,
                            std::vector<Scalar> &g) {
    gmres_small_residual(m, H, ldh, beta, y, g);
    fill_zero(n, residual);
    for (int64_t col = 0; col <= m; ++col) {
        axpy(n, g[static_cast<std::size_t>(col)],
             V + static_cast<std::size_t>(col) * static_cast<std::size_t>(ldv),
             residual);
    }
    *normres = norm2(n, residual);
}

template <typename Scalar, typename Apply>
int arnoldi(int64_t n, int64_t max_k, double breakdown_tol, const Scalar *x0,
            Scalar *V, int64_t ldv, Scalar *H, int64_t ldh, double *beta,
            int64_t *m, double *final_resnorm, int64_t *numops,
            Apply apply) {
    if (numops != nullptr) {
        *numops = 0;
    }
    std::fill(V, V + static_cast<std::size_t>(ldv) *
                       static_cast<std::size_t>(max_k + 1),
              Scalar(0));
    std::fill(H, H + static_cast<std::size_t>(ldh) *
                       static_cast<std::size_t>(max_k),
              Scalar(0));
    *beta = norm2(n, x0);
    *m = 0;
    *final_resnorm = *beta;
    if (*beta <= breakdown_tol) {
        return 0;
    }
    for (int64_t i = 0; i < n; ++i) {
        V[i] = x0[i] / Scalar(*beta);
    }
    std::vector<Scalar> w(static_cast<std::size_t>(n));
    for (int64_t j = 0; j < max_k; ++j) {
        const Scalar *vj = V + static_cast<std::size_t>(j) *
                                   static_cast<std::size_t>(ldv);
        int status = apply(vj, w.data());
        if (status != 0) {
            return status;
        }
        if (numops != nullptr) {
            *numops += 1;
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t i = 0; i <= j; ++i) {
                const Scalar *vi = V + static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(ldv);
                const Scalar hij = dot(n, vi, w.data());
                H[i + j * ldh] += hij;
                axpy(n, -hij, vi, w.data());
            }
        }
        const double hnext = norm2(n, w.data());
        H[j + 1 + j * ldh] = Scalar(hnext);
        *m = j + 1;
        *final_resnorm = hnext;
        if (hnext > breakdown_tol && std::isfinite(hnext)) {
            Scalar *vnext = V + static_cast<std::size_t>(j + 1) *
                                    static_cast<std::size_t>(ldv);
            for (int64_t i = 0; i < n; ++i) {
                vnext[i] = w[static_cast<std::size_t>(i)] / Scalar(hnext);
            }
        }
        if (hnext <= breakdown_tol || j + 1 == max_k) {
            return 0;
        }
    }
    return 0;
}

template <typename Scalar, typename Apply>
int arnoldi_prefilled(int64_t n, int64_t max_k, double breakdown_tol,
                      const Scalar *initial_V, int64_t initial_ldv,
                      int64_t initial_cols, const Scalar *initial_H,
                      int64_t initial_ldh, int64_t completed_cols, Scalar *V,
                      int64_t ldv, Scalar *H, int64_t ldh, double *beta,
                      int64_t *m, double *final_resnorm, int64_t *numops,
                      Apply apply) {
    if (numops != nullptr) {
        *numops = 0;
    }
    std::fill(V, V + static_cast<std::size_t>(ldv) *
                       static_cast<std::size_t>(max_k + 1),
              Scalar(0));
    std::fill(H, H + static_cast<std::size_t>(ldh) *
                       static_cast<std::size_t>(max_k),
              Scalar(0));
    for (int64_t col = 0; col < initial_cols; ++col) {
        copy(n,
             initial_V + static_cast<std::size_t>(col) *
                             static_cast<std::size_t>(initial_ldv),
             V + static_cast<std::size_t>(col) *
                     static_cast<std::size_t>(ldv));
    }
    for (int64_t col = 0; col < completed_cols; ++col) {
        for (int64_t row = 0; row < max_k + 1; ++row) {
            H[row + col * ldh] = initial_H[row + col * initial_ldh];
        }
    }
    *beta = 1.0;
    *m = completed_cols;
    *final_resnorm =
        completed_cols > 0
            ? scalar_abs(H[completed_cols + (completed_cols - 1) * ldh])
            : 0.0;
    if (completed_cols >= max_k || initial_cols <= completed_cols) {
        return 0;
    }

    std::vector<Scalar> w(static_cast<std::size_t>(n));
    int64_t basis_cols = initial_cols;
    for (int64_t j = completed_cols; j < max_k && j < basis_cols; ++j) {
        const Scalar *vj = V + static_cast<std::size_t>(j) *
                                   static_cast<std::size_t>(ldv);
        int status = apply(vj, w.data());
        if (status != 0) {
            return status;
        }
        if (numops != nullptr) {
            *numops += 1;
        }
        Scalar *hj = H + static_cast<std::size_t>(j) *
                             static_cast<std::size_t>(ldh);
        for (int pass = 0; pass < 2; ++pass) {
            for (int64_t i = 0; i < basis_cols; ++i) {
                const Scalar *vi = V + static_cast<std::size_t>(i) *
                                           static_cast<std::size_t>(ldv);
                const Scalar hij = dot(n, vi, w.data());
                hj[i] += hij;
                axpy(n, -hij, vi, w.data());
            }
        }
        const double hnext = norm2(n, w.data());
        *m = j + 1;
        *final_resnorm = hnext;
        if (basis_cols <= max_k) {
            hj[basis_cols] = Scalar(hnext);
            if (hnext > breakdown_tol && std::isfinite(hnext)) {
                Scalar *vnext = V + static_cast<std::size_t>(basis_cols) *
                                        static_cast<std::size_t>(ldv);
                for (int64_t i = 0; i < n; ++i) {
                    vnext[i] = w[static_cast<std::size_t>(i)] / Scalar(hnext);
                }
                ++basis_cols;
            }
        }
        if (hnext <= breakdown_tol || !std::isfinite(hnext)) {
            return 0;
        }
    }
    return 0;
}

template <typename Apply>
int gmres_real_givens(int64_t n, int64_t krylovdim, int64_t maxiter,
                      double tol, const double *b, const double *x0,
                      double a0, double a1, double *x, double *residual,
                      double *normres, int64_t *converged, int64_t *numops,
                      int64_t *numiter, Apply apply_A) {
    copy(n, x0, x);
    *converged = 0;
    *numops = 0;
    *numiter = 0;
    std::vector<double> Ax(static_cast<std::size_t>(n));
    const auto apply_shifted = [&](const double *src, double *dst) -> int {
        int status = apply_A(src, dst);
        if (status != 0) {
            return status;
        }
        *numops += 1;
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = a0 * src[i] + a1 * dst[i];
        }
        return 0;
    };
    const auto recompute_residual = [&]() -> int {
        int status = apply_shifted(x, Ax.data());
        if (status != 0) {
            return status;
        }
        for (int64_t i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[static_cast<std::size_t>(i)];
        }
        *normres = norm2(n, residual);
        return 0;
    };

    const double threshold = tol;
    int status = recompute_residual();
    if (status != 0) {
        return status;
    }
    if (*normres <= threshold) {
        *converged = 1;
        return 0;
    }

    const int64_t restart = std::max<int64_t>(1, std::min(krylovdim, n));
    std::vector<double> V(static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(restart + 1));
    std::vector<double> H(static_cast<std::size_t>(restart + 1) *
                          static_cast<std::size_t>(restart));
    std::vector<double> w(static_cast<std::size_t>(n));
    std::vector<double> cs(static_cast<std::size_t>(restart), 0.0);
    std::vector<double> sn(static_cast<std::size_t>(restart), 0.0);
    std::vector<double> g(static_cast<std::size_t>(restart + 1), 0.0);
    std::vector<double> y(static_cast<std::size_t>(restart), 0.0);

    while (*numiter < maxiter) {
        std::fill(H.begin(), H.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        const double beta = *normres;
        if (beta <= threshold) {
            *converged = 1;
            return 0;
        }
        g[0] = beta;
        for (int64_t i = 0; i < n; ++i) {
            V[static_cast<std::size_t>(i)] = residual[i] / beta;
        }

        int64_t m = 0;
        bool estimated_converged = false;
        bool arnoldi_breakdown = false;
        for (int64_t j = 0; j < restart; ++j) {
            const double *vj = V.data() + static_cast<std::size_t>(j) *
                                             static_cast<std::size_t>(n);
            status = apply_shifted(vj, w.data());
            if (status != 0) {
                return status;
            }
            for (int pass = 0; pass < 2; ++pass) {
                for (int64_t i = 0; i <= j; ++i) {
                    const double *vi =
                        V.data() + static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(n);
                    const double hij = dot(n, vi, w.data());
                    H[static_cast<std::size_t>(i + j * (restart + 1))] += hij;
                    axpy(n, -hij, vi, w.data());
                }
            }

            const double hnext = norm2(n, w.data());
            H[static_cast<std::size_t>(j + 1 + j * (restart + 1))] = hnext;
            m = j + 1;
            if (hnext > 0.0 && std::isfinite(hnext)) {
                double *vnext =
                    V.data() + static_cast<std::size_t>(j + 1) *
                                   static_cast<std::size_t>(n);
                for (int64_t i = 0; i < n; ++i) {
                    vnext[i] = w[static_cast<std::size_t>(i)] / hnext;
                }
            }

            for (int64_t i = 0; i < j; ++i) {
                const double h_i =
                    H[static_cast<std::size_t>(i + j * (restart + 1))];
                const double h_ip1 =
                    H[static_cast<std::size_t>(i + 1 + j * (restart + 1))];
                H[static_cast<std::size_t>(i + j * (restart + 1))] =
                    cs[static_cast<std::size_t>(i)] * h_i +
                    sn[static_cast<std::size_t>(i)] * h_ip1;
                H[static_cast<std::size_t>(i + 1 + j * (restart + 1))] =
                    -sn[static_cast<std::size_t>(i)] * h_i +
                    cs[static_cast<std::size_t>(i)] * h_ip1;
            }

            const double h_jj =
                H[static_cast<std::size_t>(j + j * (restart + 1))];
            const double h_j1j =
                H[static_cast<std::size_t>(j + 1 + j * (restart + 1))];
            const double rho = std::hypot(h_jj, h_j1j);
            if (!(rho > 0.0) || !std::isfinite(rho)) {
                arnoldi_breakdown = true;
                break;
            }
            const double c = h_jj / rho;
            const double s = h_j1j / rho;
            cs[static_cast<std::size_t>(j)] = c;
            sn[static_cast<std::size_t>(j)] = s;
            H[static_cast<std::size_t>(j + j * (restart + 1))] = rho;
            H[static_cast<std::size_t>(j + 1 + j * (restart + 1))] = 0.0;

            const double g_j = g[static_cast<std::size_t>(j)];
            const double g_j1 = g[static_cast<std::size_t>(j + 1)];
            g[static_cast<std::size_t>(j)] = c * g_j + s * g_j1;
            g[static_cast<std::size_t>(j + 1)] = -s * g_j + c * g_j1;
            if (std::abs(g[static_cast<std::size_t>(j + 1)]) <= threshold) {
                estimated_converged = true;
                break;
            }
            if (!(hnext > 0.0) || !std::isfinite(hnext)) {
                arnoldi_breakdown = true;
                break;
            }
        }

        if (m <= 0) {
            break;
        }
        std::fill(y.begin(), y.end(), 0.0);
        bool triangular_ok = true;
        for (int64_t row = m - 1; row >= 0; --row) {
            double acc = g[static_cast<std::size_t>(row)];
            for (int64_t col = row + 1; col < m; ++col) {
                acc -= H[static_cast<std::size_t>(row +
                                                  col * (restart + 1))] *
                       y[static_cast<std::size_t>(col)];
            }
            const double diag =
                H[static_cast<std::size_t>(row + row * (restart + 1))];
            if (!(std::abs(diag) > 0.0) || !std::isfinite(diag)) {
                triangular_ok = false;
                break;
            }
            y[static_cast<std::size_t>(row)] = acc / diag;
        }
        if (!triangular_ok) {
            break;
        }
        for (int64_t j = 0; j < m; ++j) {
            axpy(n, y[static_cast<std::size_t>(j)],
                 V.data() + static_cast<std::size_t>(j) *
                                static_cast<std::size_t>(n),
                 x);
        }
        *numiter += 1;

        status = recompute_residual();
        if (status != 0) {
            return status;
        }
        if (*normres <= threshold) {
            *converged = 1;
            return 0;
        }
        (void)estimated_converged;
        (void)arnoldi_breakdown;
    }
    if (*normres <= threshold) {
        *converged = 1;
    }
    return 0;
}

template <typename Scalar, typename Apply>
int gmres(int64_t n, int64_t krylovdim, int64_t maxiter, double tol,
          const Scalar *b, const Scalar *x0, Scalar a0, Scalar a1, Scalar *x,
          Scalar *residual, double *normres, int64_t *converged,
          int64_t *numops, int64_t *numiter, Apply apply_A) {
    if constexpr (std::is_same<Scalar, double>::value) {
        return gmres_real_givens(n, krylovdim, maxiter, tol, b, x0, a0, a1,
                                 x, residual, normres, converged, numops,
                                 numiter, apply_A);
    }
    copy(n, x0, x);
    *converged = 0;
    *numops = 0;
    *numiter = 0;
    std::vector<Scalar> Ax(static_cast<std::size_t>(n));
    const auto apply_shifted = [&](const Scalar *src, Scalar *dst) -> int {
        int status = apply_A(src, dst);
        if (status != 0) {
            return status;
        }
        *numops += 1;
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = a0 * src[i] + a1 * dst[i];
        }
        return 0;
    };
    const auto recompute_residual = [&]() -> int {
        int status = apply_shifted(x, Ax.data());
        if (status != 0) {
            return status;
        }
        for (int64_t i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[static_cast<std::size_t>(i)];
        }
        *normres = norm2(n, residual);
        return 0;
    };
    const double threshold = tol;
    int status = recompute_residual();
    if (status != 0) {
        return status;
    }
    if (*normres <= threshold) {
        *converged = 1;
        return 0;
    }
    const int64_t restart = std::max<int64_t>(1, std::min(krylovdim, n));
    std::vector<Scalar> V(static_cast<std::size_t>(n) *
                          static_cast<std::size_t>(restart + 1));
    std::vector<Scalar> H(static_cast<std::size_t>(restart + 1) *
                          static_cast<std::size_t>(restart));
    std::vector<Scalar> w(static_cast<std::size_t>(n));
    std::vector<Scalar> y;
    std::vector<Scalar> g;
    std::vector<Scalar> Qwork;
    std::vector<Scalar> Rwork;
    std::vector<Scalar> rhswork;
    while (*numiter < maxiter) {
        std::fill(H.begin(), H.end(), Scalar(0));
        const double beta = *normres;
        if (beta <= threshold) {
            *converged = 1;
            return 0;
        }
        for (int64_t i = 0; i < n; ++i) {
            V[static_cast<std::size_t>(i)] = residual[i] / Scalar(beta);
        }

        int64_t m = 0;
        bool have_y = false;
        bool arnoldi_breakdown = false;
        bool estimated_converged = false;
        for (int64_t j = 0; j < restart; ++j) {
            const Scalar *vj = V.data() + static_cast<std::size_t>(j) *
                                              static_cast<std::size_t>(n);
            status = apply_shifted(vj, w.data());
            if (status != 0) {
                return status;
            }
            for (int pass = 0; pass < 2; ++pass) {
                for (int64_t i = 0; i <= j; ++i) {
                    const Scalar *vi =
                        V.data() + static_cast<std::size_t>(i) *
                                       static_cast<std::size_t>(n);
                    const Scalar hij = dot(n, vi, w.data());
                    H[static_cast<std::size_t>(i + j * (restart + 1))] += hij;
                    axpy(n, -hij, vi, w.data());
                }
            }
            const double hnext = norm2(n, w.data());
            H[static_cast<std::size_t>(j + 1 + j * (restart + 1))] =
                Scalar(hnext);
            m = j + 1;
            if (hnext > 0.0 && std::isfinite(hnext)) {
                Scalar *vnext =
                    V.data() + static_cast<std::size_t>(j + 1) *
                                   static_cast<std::size_t>(n);
                for (int64_t i = 0; i < n; ++i) {
                    vnext[i] = w[static_cast<std::size_t>(i)] /
                               Scalar(hnext);
                }
            }
            if (least_squares_qr_work(m + 1, m, H.data(), restart + 1,
                                      Scalar(beta), y, Qwork, Rwork,
                                      rhswork)) {
                have_y = true;
                const double estimate =
                    gmres_small_resnorm(m, H.data(), restart + 1,
                                        Scalar(beta), y, g);
                if (estimate <= threshold) {
                    estimated_converged = true;
                    break;
                }
            } else {
                break;
            }
            if (!(hnext > 0.0) || !std::isfinite(hnext)) {
                arnoldi_breakdown = true;
                break;
            }
        }

        if (m <= 0 || !have_y) {
            break;
        }
        for (int64_t j = 0; j < m; ++j) {
            axpy(n, y[static_cast<std::size_t>(j)],
                 V.data() + static_cast<std::size_t>(j) *
                                static_cast<std::size_t>(n),
                 x);
        }
        *numiter += 1;
        if (estimated_converged) {
            status = recompute_residual();
            if (status != 0) {
                return status;
            }
            if (*normres <= threshold) {
                *converged = 1;
                return 0;
            }
        } else {
            gmres_project_residual(n, m, V.data(), n, H.data(), restart + 1,
                                   Scalar(beta), y, residual, normres, g);
            if (*normres <= threshold) {
                status = recompute_residual();
                if (status != 0) {
                    return status;
                }
                if (*normres <= threshold) {
                    *converged = 1;
                    return 0;
                }
            }
        }
        if (arnoldi_breakdown || m < restart) {
            break;
        }
    }
    status = recompute_residual();
    if (status != 0) {
        return status;
    }
    if (*normres <= threshold) {
        *converged = 1;
    }
    return 0;
}

template <typename Scalar, typename Apply>
int cg(int64_t n, int64_t maxiter, double tol, const Scalar *b,
       const Scalar *x0, Scalar a0, Scalar a1, Scalar *x, Scalar *residual,
       double *normres, int64_t *converged, int64_t *numops,
       int64_t *numiter, Apply apply_A) {
    copy(n, x0, x);
    *converged = 0;
    *numops = 0;
    *numiter = 0;
    std::vector<Scalar> Ap(static_cast<std::size_t>(n));
    const auto apply_shifted = [&](const Scalar *src, Scalar *dst) -> int {
        int status = apply_A(src, dst);
        if (status != 0) {
            return status;
        }
        *numops += 1;
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = a0 * src[i] + a1 * dst[i];
        }
        return 0;
    };
    const auto recompute_residual = [&]() -> int {
        int status = apply_shifted(x, Ap.data());
        if (status != 0) {
            return status;
        }
        for (int64_t i = 0; i < n; ++i) {
            residual[i] = b[i] - Ap[static_cast<std::size_t>(i)];
        }
        *normres = norm2(n, residual);
        return 0;
    };
    int status = recompute_residual();
    if (status != 0) {
        return status;
    }
    const double threshold = tol;
    if (*normres <= threshold) {
        *converged = 1;
        return 0;
    }
    std::vector<Scalar> p(static_cast<std::size_t>(n));
    copy(n, residual, p.data());
    Scalar rsold = dot(n, residual, residual);
    const double eps = 64.0 * std::numeric_limits<double>::epsilon();
    while (*numiter < maxiter) {
        status = apply_shifted(p.data(), Ap.data());
        if (status != 0) {
            return status;
        }
        const Scalar denom = dot(n, p.data(), Ap.data());
        const double denom_abs = scalar_abs(denom);
        const double denom_scale = norm2(n, p.data()) * norm2(n, Ap.data());
        if (!(denom_scale > 0.0) || !std::isfinite(denom_scale) ||
            !(denom_abs > eps * denom_scale) || !std::isfinite(denom_abs)) {
            break;
        }
        const Scalar alpha = rsold / denom;
        axpy(n, alpha, p.data(), x);
        axpy(n, -alpha, Ap.data(), residual);
        *numiter += 1;
        *normres = norm2(n, residual);
        if (*normres <= threshold) {
            status = recompute_residual();
            if (status != 0) {
                return status;
            }
            if (*normres <= threshold) {
                *converged = 1;
            }
            return 0;
        }
        const Scalar rsnew = dot(n, residual, residual);
        const double rsold_abs = scalar_abs(rsold);
        const double rsnew_abs = scalar_abs(rsnew);
        if (!(rsold_abs > 0.0) || !std::isfinite(rsold_abs) ||
            !std::isfinite(rsnew_abs)) {
            break;
        }
        const Scalar beta = rsnew / rsold;
        for (int64_t i = 0; i < n; ++i) {
            p[static_cast<std::size_t>(i)] =
                residual[i] + beta * p[static_cast<std::size_t>(i)];
        }
        rsold = rsnew;
    }
    status = recompute_residual();
    if (status != 0) {
        return status;
    }
    if (*normres <= threshold) {
        *converged = 1;
    }
    return 0;
}

template <typename Scalar, typename Apply>
int bicgstab(int64_t n, int64_t maxiter, double tol, const Scalar *b,
             const Scalar *x0, Scalar a0, Scalar a1, Scalar *x,
             Scalar *residual, double *normres, int64_t *converged,
             int64_t *numops, int64_t *numiter, Apply apply_A) {
    copy(n, x0, x);
    *converged = 0;
    *numops = 0;
    *numiter = 0;
    std::vector<Scalar> work(static_cast<std::size_t>(n));
    const auto apply_shifted = [&](const Scalar *src, Scalar *dst) -> int {
        int status = apply_A(src, dst);
        if (status != 0) {
            return status;
        }
        *numops += 1;
        for (int64_t i = 0; i < n; ++i) {
            dst[i] = a0 * src[i] + a1 * dst[i];
        }
        return 0;
    };
    const auto recompute_residual = [&]() -> int {
        int status = apply_shifted(x, work.data());
        if (status != 0) {
            return status;
        }
        for (int64_t i = 0; i < n; ++i) {
            residual[i] = b[i] - work[static_cast<std::size_t>(i)];
        }
        *normres = norm2(n, residual);
        return 0;
    };
    int status = recompute_residual();
    if (status != 0) {
        return status;
    }
    const double threshold = tol;
    if (*normres <= threshold) {
        *converged = 1;
        return 0;
    }
    if (maxiter <= 0) {
        return 0;
    }

    std::vector<Scalar> r_hat(static_cast<std::size_t>(n));
    std::vector<Scalar> p(static_cast<std::size_t>(n), Scalar(0));
    std::vector<Scalar> v(static_cast<std::size_t>(n), Scalar(0));
    std::vector<Scalar> s(static_cast<std::size_t>(n));
    std::vector<Scalar> t(static_cast<std::size_t>(n));
    std::vector<Scalar> xhalf(static_cast<std::size_t>(n));
    copy(n, residual, r_hat.data());

    Scalar rho_prev = Scalar(1);
    Scalar alpha = Scalar(1);
    Scalar omega = Scalar(1);
    for (int64_t iter = 1; iter <= maxiter; ++iter) {
        const Scalar rho = dot(n, r_hat.data(), residual);
        const double rho_abs = scalar_abs(rho);
        if (!(rho_abs > 0.0) || !std::isfinite(rho_abs)) {
            break;
        }
        if (iter == 1) {
            copy(n, residual, p.data());
        } else {
            const double rho_prev_abs = scalar_abs(rho_prev);
            const double omega_abs = scalar_abs(omega);
            if (!(rho_prev_abs > 0.0) || !std::isfinite(rho_prev_abs) ||
                !(omega_abs > 0.0) || !std::isfinite(omega_abs)) {
                break;
            }
            const Scalar beta = (rho / rho_prev) * (alpha / omega);
            for (int64_t i = 0; i < n; ++i) {
                p[static_cast<std::size_t>(i)] =
                    residual[i] + beta *
                                      (p[static_cast<std::size_t>(i)] -
                                       omega * v[static_cast<std::size_t>(i)]);
            }
        }

        status = apply_shifted(p.data(), v.data());
        if (status != 0) {
            return status;
        }
        const Scalar sigma = dot(n, r_hat.data(), v.data());
        const double sigma_abs = scalar_abs(sigma);
        if (!(sigma_abs > 0.0) || !std::isfinite(sigma_abs)) {
            break;
        }
        alpha = rho / sigma;
        copy(n, residual, s.data());
        axpy(n, -alpha, v.data(), s.data());
        copy(n, x, xhalf.data());
        axpy(n, alpha, p.data(), xhalf.data());

        double s_norm = norm2(n, s.data());
        if (s_norm <= threshold) {
            copy(n, xhalf.data(), x);
            status = recompute_residual();
            if (status != 0) {
                return status;
            }
            *numiter = iter;
            if (*normres <= threshold) {
                *converged = 1;
                return 0;
            }
        }

        status = apply_shifted(s.data(), t.data());
        if (status != 0) {
            return status;
        }
        const Scalar tt = dot(n, t.data(), t.data());
        const double tt_abs = scalar_abs(tt);
        if (!(tt_abs > 0.0) || !std::isfinite(tt_abs)) {
            break;
        }
        omega = dot(n, t.data(), s.data()) / tt;
        const double omega_abs = scalar_abs(omega);
        if (!(omega_abs > 0.0) || !std::isfinite(omega_abs)) {
            break;
        }
        copy(n, xhalf.data(), x);
        axpy(n, omega, s.data(), x);
        copy(n, s.data(), residual);
        axpy(n, -omega, t.data(), residual);
        *numiter = iter;
        *normres = norm2(n, residual);
        if (*normres <= threshold) {
            status = recompute_residual();
            if (status != 0) {
                return status;
            }
            if (*normres <= threshold) {
                *converged = 1;
                return 0;
            }
        }
        rho_prev = rho;
    }

    status = recompute_residual();
    if (status != 0) {
        return status;
    }
    if (*normres <= threshold) {
        *converged = 1;
    }
    return 0;
}

} // namespace tenet_native_krylov

#endif
