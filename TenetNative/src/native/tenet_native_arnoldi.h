#ifndef TENET_NATIVE_ARNOLDI_H
#define TENET_NATIVE_ARNOLDI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TENET_NATIVE_SUCCESS = 0,
    TENET_NATIVE_INVALID_VALUE = 1,
    TENET_NATIVE_ALLOCATION_FAILED = 2,
    TENET_NATIVE_BACKEND_ERROR = 3
};

/*
 * Legacy Arnoldi ABI metadata stays at v3 so existing exact-v3 loaders can
 * keep using the preserved v3 entrypoints. The additive generic Krylov symbols
 * advertise their own v4 capability metadata below.
 */
#define TENET_NATIVE_ABI_VERSION 3
#define TENET_NATIVE_ABI_VERSION_STRING "tenet_native_arnoldi_abi_v3"
#define TENET_NATIVE_KRYLOV_ABI_VERSION 4
#define TENET_NATIVE_KRYLOV_ABI_VERSION_STRING "tenet_native_krylov_abi_v4"

typedef struct {
    double re;
    double im;
} tenet_native_complex64;

typedef int (*tenet_native_matvec_d_cpu_fn)(
    int64_t n,
    const double *x,
    double *y,
    void *ctx);

typedef int (*tenet_native_matvec_z_cpu_fn)(
    int64_t n,
    const tenet_native_complex64 *x,
    tenet_native_complex64 *y,
    void *ctx);

int tenet_native_abi_version(void);
const char *tenet_native_abi_version_string(void);
int tenet_native_krylov_abi_version(void);
const char *tenet_native_krylov_abi_version_string(void);
const char *tenet_native_status_string(int status);
const char *tenet_native_last_error(void);

int tenet_native_krylov_arnoldi_d_cpu(
    int64_t n,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_matvec_d_cpu_fn matvec,
    void *ctx,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_z_cpu(
    int64_t n,
    const tenet_native_complex64 *x0,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_matvec_z_cpu_fn matvec,
    void *ctx,
    tenet_native_complex64 *V,
    int64_t ldv,
    tenet_native_complex64 *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_prefilled_d_cpu(
    int64_t n,
    const double *initial_V,
    int64_t initial_ldv,
    int64_t initial_cols,
    const double *initial_H,
    int64_t initial_ldh,
    int64_t completed_cols,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_matvec_d_cpu_fn matvec,
    void *ctx,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_prefilled_z_cpu(
    int64_t n,
    const tenet_native_complex64 *initial_V,
    int64_t initial_ldv,
    int64_t initial_cols,
    const tenet_native_complex64 *initial_H,
    int64_t initial_ldh,
    int64_t completed_cols,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_matvec_z_cpu_fn matvec,
    void *ctx,
    tenet_native_complex64 *V,
    int64_t ldv,
    tenet_native_complex64 *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_gmres_d_cpu(
    int64_t n,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t krylovdim,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_d_cpu_fn matvec,
    void *ctx,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_gmres_z_cpu(
    int64_t n,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t krylovdim,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_z_cpu_fn matvec,
    void *ctx,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_cg_d_cpu(
    int64_t n,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_d_cpu_fn matvec,
    void *ctx,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_cg_z_cpu(
    int64_t n,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_z_cpu_fn matvec,
    void *ctx,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_bicgstab_d_cpu(
    int64_t n,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_d_cpu_fn matvec,
    void *ctx,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_bicgstab_z_cpu(
    int64_t n,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t maxiter,
    double tol,
    tenet_native_matvec_z_cpu_fn matvec,
    void *ctx,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_arnoldi_dense_d_cpu(
    int64_t n,
    const double *A,
    int64_t lda,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_dense_z_cpu(
    int64_t n,
    const tenet_native_complex64 *A,
    int64_t lda,
    const tenet_native_complex64 *x0,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_complex64 *V,
    int64_t ldv,
    tenet_native_complex64 *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_prefilled_dense_d_cpu(
    int64_t n,
    const double *A,
    int64_t lda,
    const double *initial_V,
    int64_t initial_ldv,
    int64_t initial_cols,
    const double *initial_H,
    int64_t initial_ldh,
    int64_t completed_cols,
    int64_t max_k,
    double breakdown_tol,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_arnoldi_prefilled_dense_z_cpu(
    int64_t n,
    const tenet_native_complex64 *A,
    int64_t lda,
    const tenet_native_complex64 *initial_V,
    int64_t initial_ldv,
    int64_t initial_cols,
    const tenet_native_complex64 *initial_H,
    int64_t initial_ldh,
    int64_t completed_cols,
    int64_t max_k,
    double breakdown_tol,
    tenet_native_complex64 *V,
    int64_t ldv,
    tenet_native_complex64 *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm,
    int64_t *numops);

int tenet_native_krylov_gmres_dense_d_cpu(
    int64_t n,
    const double *A,
    int64_t lda,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t krylovdim,
    int64_t maxiter,
    double tol,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_gmres_dense_z_cpu(
    int64_t n,
    const tenet_native_complex64 *A,
    int64_t lda,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t krylovdim,
    int64_t maxiter,
    double tol,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_cg_dense_d_cpu(
    int64_t n,
    const double *A,
    int64_t lda,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t maxiter,
    double tol,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_cg_dense_z_cpu(
    int64_t n,
    const tenet_native_complex64 *A,
    int64_t lda,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t maxiter,
    double tol,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_bicgstab_dense_d_cpu(
    int64_t n,
    const double *A,
    int64_t lda,
    const double *b,
    const double *x0,
    double a0,
    double a1,
    int64_t maxiter,
    double tol,
    double *x,
    double *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_krylov_bicgstab_dense_z_cpu(
    int64_t n,
    const tenet_native_complex64 *A,
    int64_t lda,
    const tenet_native_complex64 *b,
    const tenet_native_complex64 *x0,
    tenet_native_complex64 a0,
    tenet_native_complex64 a1,
    int64_t maxiter,
    double tol,
    tenet_native_complex64 *x,
    tenet_native_complex64 *residual,
    double *normres,
    int64_t *converged,
    int64_t *numops,
    int64_t *numiter);

int tenet_native_raw_two_layer_apply_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x,
    int transpose,
    double *y);

int tenet_native_raw_transfer_op_d_cpu(
    int64_t phys,
    int64_t chi,
    const double *W,
    const double *O,
    const double *x,
    double *y);

int tenet_native_raw_rowmajor_transfer_d_cpu(
    int64_t d,
    int64_t D,
    const double *W,
    const double *x,
    double *y);

int tenet_native_raw_rowmajor_transfer_adj_d_cpu(
    int64_t d,
    int64_t D,
    const double *W,
    const double *x,
    double *y);

int tenet_native_raw_rowmajor_transfer_op_d_cpu(
    int64_t d,
    int64_t D,
    const double *W,
    const double *O,
    const double *x,
    double *y);

int tenet_native_arnoldi_two_layer_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_arnoldi_two_layer_ritz_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    int64_t nvalues,
    double *lambda_real,
    double *lambda_imag,
    int64_t *m);

int tenet_native_arnoldi_projected_two_layer_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *rho,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_arnoldi_qprojected_two_layer_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *rho,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_arnoldi_three_layer_leg4_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *M,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_dominant_two_layer_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_smallest_real_two_layer_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_dominant_three_layer_leg4_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *M,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_smallest_real_three_layer_leg4_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *M,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_ising_vumps_step_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t max_k,
    double breakdown_tol,
    double *err);

int tenet_native_ising_vumps_step_checked_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t max_k,
    double breakdown_tol,
    double residual_tol,
    double *err);

int tenet_native_ising_vumps_run_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t arnoldi_max_k,
    double breakdown_tol,
    double tol,
    int64_t miniter,
    int64_t maxiter,
    double *err,
    int64_t *iterations,
    int *converged);

int tenet_native_ising_vumps_run_checked_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t arnoldi_max_k,
    double breakdown_tol,
    double tol,
    int64_t miniter,
    int64_t maxiter,
    double residual_tol,
    double *err,
    int64_t *iterations,
    int *converged);

int tenet_native_acc_to_alar_d_cpu(
    int64_t chi,
    int64_t phys,
    const double *AC,
    const double *C,
    double *AL,
    double *AR,
    double *err);

int tenet_native_arnoldi_two_layer_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_arnoldi_two_layer_ritz_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    int64_t nvalues,
    double *lambda_real,
    double *lambda_imag,
    int64_t *m);

int tenet_native_arnoldi_projected_two_layer_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *rho,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_arnoldi_qprojected_two_layer_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *rho,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_two_layer_apply_batch_d_cuda(
    int64_t batch,
    int64_t chi,
    int64_t phys,
    const double *Aup,
    int64_t stride_Aup,
    const double *Adn,
    int64_t stride_Adn,
    const double *X,
    int64_t stride_X,
    int transpose,
    double *Y,
    int64_t stride_Y);

int tenet_native_raw_two_layer_apply_batch_d_cuda(
    int64_t batch,
    int64_t chi,
    int64_t phys,
    const double *Aup,
    int64_t stride_Aup,
    const double *Adn,
    int64_t stride_Adn,
    const double *X,
    int64_t stride_X,
    int transpose,
    double *Y,
    int64_t stride_Y);

int tenet_native_projected_two_layer_apply_batch_d_cuda(
    int64_t batch,
    int64_t chi,
    int64_t phys,
    const double *Aup,
    int64_t stride_Aup,
    const double *Adn,
    int64_t stride_Adn,
    const double *rho,
    int64_t stride_rho,
    const double *X,
    int64_t stride_X,
    int transpose,
    double *Y,
    int64_t stride_Y);

int tenet_native_qprojected_two_layer_apply_batch_d_cuda(
    int64_t batch,
    int64_t chi,
    int64_t phys,
    const double *Aup,
    int64_t stride_Aup,
    const double *Adn,
    int64_t stride_Adn,
    const double *rho,
    int64_t stride_rho,
    const double *X,
    int64_t stride_X,
    int transpose,
    double *Y,
    int64_t stride_Y);

int tenet_native_arnoldi_three_layer_leg4_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *M,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *V,
    int64_t ldv,
    double *H,
    int64_t ldh,
    double *beta,
    int64_t *m,
    double *final_resnorm);

int tenet_native_dominant_two_layer_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_dominant_three_layer_leg4_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *Aup,
    const double *Adn,
    const double *M,
    const double *x0,
    int64_t max_k,
    double breakdown_tol,
    int transpose,
    double *y,
    double *lambda);

int tenet_native_ising_vumps_step_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t max_k,
    double breakdown_tol,
    double *err);

int tenet_native_ising_vumps_step_checked_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t max_k,
    double breakdown_tol,
    double residual_tol,
    double *err);

int tenet_native_ising_vumps_run_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t arnoldi_max_k,
    double breakdown_tol,
    double tol,
    int64_t miniter,
    int64_t maxiter,
    double *err,
    int64_t *iterations,
    int *converged);

int tenet_native_ising_vumps_run_checked_d_cuda(
    int64_t chi,
    int64_t phys,
    const double *M,
    double *AL,
    double *AR,
    double *C,
    double *FL,
    double *FR,
    int64_t arnoldi_max_k,
    double breakdown_tol,
    double tol,
    int64_t miniter,
    int64_t maxiter,
    double residual_tol,
    double *err,
    int64_t *iterations,
    int *converged);

#ifdef __cplusplus
}
#endif

#endif
