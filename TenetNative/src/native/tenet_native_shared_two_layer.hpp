#pragma once

// Canonical shared real-valued two-layer operator helpers consumed by
// TenetNative and TorchExactLRuMPS CPU paths.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tenet_native_shared {

enum class TwoLayerMode { Forward, Adjoint };

inline double dense_dot(int64_t n, const double *x, const double *y) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        acc += x[i] * y[i];
    }
    return acc;
}

inline double dense_trace(int64_t dim, const double *x) {
    double acc = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        acc += x[i * dim + i];
    }
    return acc;
}

inline void project_q(int64_t dim, const double *rho, const double *x,
                      double *y) {
    const int64_t len = dim * dim;
    const double tr_x = dense_trace(dim, x);
    for (int64_t i = 0; i < len; ++i) {
        y[i] = x[i] - tr_x * rho[i];
    }
}

inline void project_q_adj(int64_t dim, const double *rho, const double *x,
                          double *y) {
    const int64_t len = dim * dim;
    const double alpha = dense_dot(len, rho, x);
    std::copy(x, x + len, y);
    for (int64_t i = 0; i < dim; ++i) {
        y[i * dim + i] -= alpha;
    }
}

template <class Layout>
inline void two_layer_apply(const Layout &layout, int64_t phys,
                            const double *Aup, const double *Adn,
                            const double *x, double *tmp, double *y,
                            TwoLayerMode mode) {
    std::fill(y, y + layout.matrix_len(), 0.0);
    for (int64_t s = 0; s < phys; ++s) {
        const double *A = layout.slice(Aup, s);
        const double *B = layout.slice(Adn, s);
        if (mode == TwoLayerMode::Adjoint) {
            layout.apply_adjoint_slice(A, B, x, tmp, y);
        } else {
            layout.apply_forward_slice(A, B, x, tmp, y);
        }
    }
}

template <class Layout>
inline void projected_two_layer_apply(const Layout &layout, int64_t phys,
                                      const double *Aup, const double *Adn,
                                      const double *rho, const double *x,
                                      double *tmp, double *y,
                                      TwoLayerMode mode) {
    const int64_t len = layout.matrix_len();
    two_layer_apply(layout, phys, Aup, Adn, x, tmp, y, mode);
    const double projection = dense_dot(len, x, rho);
    for (int64_t i = 0; i < len; ++i) {
        y[i] = x[i] - y[i];
    }
    for (int64_t i = 0; i < layout.dim(); ++i) {
        y[i * layout.dim() + i] += projection;
    }
}

template <class Layout>
inline void qprojected_two_layer_apply(const Layout &layout, int64_t phys,
                                       const double *Aup, const double *Adn,
                                       const double *rho, const double *x,
                                       double *project_work,
                                       double *gemm_work, double *y,
                                       TwoLayerMode mode) {
    if (mode == TwoLayerMode::Adjoint) {
        project_q_adj(layout.dim(), rho, x, project_work);
        two_layer_apply(layout, phys, Aup, Adn, project_work, gemm_work, y,
                        TwoLayerMode::Adjoint);
        project_q_adj(layout.dim(), rho, y, y);
    } else {
        project_q(layout.dim(), rho, x, project_work);
        two_layer_apply(layout, phys, Aup, Adn, project_work, gemm_work, y,
                        TwoLayerMode::Forward);
        project_q(layout.dim(), rho, y, y);
    }
}

}  // namespace tenet_native_shared
