#include "tenet_native_arnoldi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dgels_64_(char *trans, int64_t *m, int64_t *n, int64_t *nrhs, double *a,
               int64_t *lda, double *b, int64_t *ldb, double *work,
               int64_t *lwork, int64_t *info);
void dgemm_64_(char *transa, char *transb, int64_t *m, int64_t *n,
               int64_t *k, double *alpha, const double *a, int64_t *lda,
               const double *b, int64_t *ldb, double *beta, double *c,
               int64_t *ldc);
}

namespace {

int64_t idx2(int64_t i, int64_t j, int64_t n1) { return i + n1 * j; }

int64_t idx3(int64_t i, int64_t j, int64_t k, int64_t n1, int64_t n2) {
    return i + n1 * (j + n2 * k);
}

int64_t idx4(int64_t i, int64_t j, int64_t k, int64_t l, int64_t n1,
             int64_t n2, int64_t n3) {
    return i + n1 * (j + n2 * (k + n3 * l));
}

struct Options {
    double field = 1.0;
    int64_t chi = 8;
    int64_t maxiter = 1;
    int64_t miniter = 1;
    int64_t max_k = 30;
    int64_t arnoldi_restarts = 0;
    double eig_tol = 1e-10;
    double env_tol = 1e-11;
    double tol = 1e-10;
    int64_t repetitions = 1;
    int64_t warmup = 0;
    int64_t seed = 1234;
    std::string init = "random";
    std::string dump_state;
    std::string load_initial_state;
};

struct State {
    std::vector<double> AL;
    std::vector<double> AR;
    std::vector<double> C;
    std::vector<double> AC;
};

struct Env {
    std::vector<std::vector<double>> left;
    std::vector<std::vector<double>> right;
};

struct StepResult {
    State state;
    double err = 0.0;
    double ac_seconds = 0.0;
    double c_seconds = 0.0;
    double acc_seconds = 0.0;
    double acnext_seconds = 0.0;
};

struct RunResult {
    State state;
    Env env;
    double energy = 0.0;
    double err = 0.0;
    double env_seconds = 0.0;
    double step_seconds = 0.0;
    double ac_seconds = 0.0;
    double c_seconds = 0.0;
    double acc_seconds = 0.0;
    double acnext_seconds = 0.0;
    double energy_seconds = 0.0;
    int64_t iterations = 0;
    int converged = 0;
};

void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--field h] [--chi n] [--maxiter n] [--miniter n]\n"
        << "       [--krylovdim n|--max-k n] [--arnoldi-restarts n]\n"
        << "       [--eig-tol x] [--env-tol x] [--tol x]\n"
        << "       [--warmup n] [--repetitions n] [--seed n]\n"
        << "       [--init random|product] [--dump-state path]\n"
        << "       [--load-initial-state path]\n";
}

bool consume_arg(int &i, int argc, char **argv, const char *name,
                 std::string &out) {
    if (std::strcmp(argv[i], name) != 0) {
        return false;
    }
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    out = argv[++i];
    return true;
}

Options parse_options(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string value;
        if (consume_arg(i, argc, argv, "--field", value)) {
            opt.field = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--chi", value)) {
            opt.chi = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--maxiter", value)) {
            opt.maxiter = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--miniter", value)) {
            opt.miniter = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--max-k", value) ||
                   consume_arg(i, argc, argv, "--krylovdim", value)) {
            opt.max_k = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--arnoldi-restarts", value)) {
            opt.arnoldi_restarts = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--eig-tol", value)) {
            opt.eig_tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--env-tol", value)) {
            opt.env_tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--tol", value)) {
            opt.tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--warmup", value)) {
            opt.warmup = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--repetitions", value)) {
            opt.repetitions = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--seed", value)) {
            opt.seed = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--init", value)) {
            opt.init = value;
        } else if (consume_arg(i, argc, argv, "--dump-state", value)) {
            opt.dump_state = value;
        } else if (consume_arg(i, argc, argv, "--load-initial-state", value)) {
            opt.load_initial_state = value;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument(std::string("unknown argument: ") +
                                        argv[i]);
        }
    }
    if (opt.field < 0.0 || opt.chi <= 0 || opt.maxiter < 0 ||
        opt.miniter < 0 || opt.miniter > opt.maxiter || opt.max_k <= 0 ||
        opt.arnoldi_restarts < 0 || opt.eig_tol < 0.0 ||
        opt.env_tol < 0.0 || opt.tol < 0.0 ||
        opt.warmup < 0 || opt.repetitions <= 0) {
        throw std::invalid_argument("invalid dimensions, field, or tolerances");
    }
    if (opt.init != "random" && opt.init != "product") {
        throw std::invalid_argument("invalid --init; expected random or product");
    }
    return opt;
}

void apply_arnoldi_restarts(const Options &opt) {
    if (opt.arnoldi_restarts <= 0) {
        return;
    }
    const std::string value = std::to_string(opt.arnoldi_restarts);
    setenv("TENET_NATIVE_ARNOLDI_RESTARTS", value.c_str(), 1);
}

int64_t effective_arnoldi_restarts(const Options &opt) {
    if (opt.arnoldi_restarts > 0) {
        return opt.arnoldi_restarts;
    }
    if (const char *env = std::getenv("TENET_NATIVE_ARNOLDI_RESTARTS")) {
        const int64_t parsed = std::atoll(env);
        if (parsed > 0) {
            return std::min<int64_t>(parsed, 1024);
        }
    }
    return 100;
}

double norm2(const std::vector<double> &x) {
    double acc = 0.0;
    for (double v : x) acc += v * v;
    return std::sqrt(acc);
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
    return acc;
}

double sum(const std::vector<double> &x) {
    double acc = 0.0;
    for (double v : x) acc += v;
    return acc;
}

void blas_gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
               double alpha, const double *A, int64_t lda, const double *B,
               int64_t ldb, double beta, double *C, int64_t ldc) {
    dgemm_64_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta,
              C, &ldc);
}

std::vector<double> physical_slice(const std::vector<double> &A, int64_t chi,
                                   int64_t p) {
    const int64_t phys = 2;
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    for (int64_t col = 0; col < chi; ++col)
        for (int64_t row = 0; row < chi; ++row)
            out[idx2(row, col, chi)] = A[idx3(row, p, col, chi, phys)];
    return out;
}

void write_physical_slice(std::vector<double> &A, const std::vector<double> &S,
                          int64_t chi, int64_t p) {
    const int64_t phys = 2;
    for (int64_t col = 0; col < chi; ++col)
        for (int64_t row = 0; row < chi; ++row)
            A[idx3(row, p, col, chi, phys)] = S[idx2(row, col, chi)];
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void normalize(std::vector<double> &x) {
    const double n = norm2(x);
    if (!(n > 0.0) || !std::isfinite(n)) {
        throw std::runtime_error("cannot normalize invalid vector");
    }
    for (double &v : x) v /= n;
}

std::vector<double> identity(int64_t chi) {
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    for (int64_t i = 0; i < chi; ++i) out[idx2(i, i, chi)] = 1.0;
    return out;
}

void fill_tfising_mpo(double field, std::vector<double> &W) {
    W.assign(3 * 2 * 3 * 2, 0.0);
    const double id[2][2] = {{1.0, 0.0}, {0.0, 1.0}};
    const double x[2][2] = {{0.0, 1.0}, {1.0, 0.0}};
    const double z[2][2] = {{1.0, 0.0}, {0.0, -1.0}};
    for (int g = 0; g < 2; ++g) {
        for (int b = 0; b < 2; ++b) {
            W[idx4(0, g, 0, b, 3, 2, 3)] = id[g][b];
            W[idx4(1, g, 0, b, 3, 2, 3)] = z[g][b];
            W[idx4(2, g, 0, b, 3, 2, 3)] = -field * x[g][b];
            W[idx4(2, g, 1, b, 3, 2, 3)] = -z[g][b];
            W[idx4(2, g, 2, b, 3, 2, 3)] = id[g][b];
        }
    }
}

std::vector<double> wslice(const std::vector<double> &W, int64_t left,
                           int64_t right) {
    std::vector<double> O(4, 0.0);
    for (int64_t g = 0; g < 2; ++g) {
        for (int64_t b = 0; b < 2; ++b) {
            O[idx2(g, b, 2)] = W[idx4(left, g, right, b, 3, 2, 3)];
        }
    }
    return O;
}

std::vector<double> left_apply(const std::vector<double> &L,
                               const std::vector<double> &A,
                               const std::vector<double> &O, int64_t chi) {
    const int64_t phys = 2;
    const auto A0 = physical_slice(A, chi, 0);
    const auto A1 = physical_slice(A, chi, 1);
    const std::vector<const double *> As = {A0.data(), A1.data()};
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    std::vector<double> tmp(static_cast<std::size_t>(chi * chi), 0.0);
    for (int64_t g = 0; g < phys; ++g) {
        blas_gemm('N', 'N', chi, chi, chi, 1.0, L.data(), chi, As[g], chi,
                  0.0, tmp.data(), chi);
        for (int64_t b = 0; b < phys; ++b) {
            const double alpha = O[idx2(g, b, phys)];
            if (alpha == 0.0) continue;
            blas_gemm('T', 'N', chi, chi, chi, alpha, As[b], chi, tmp.data(),
                      chi, 1.0, out.data(), chi);
        }
    }
    return out;
}

std::vector<double> right_apply(const std::vector<double> &R,
                                const std::vector<double> &A,
                                const std::vector<double> &O, int64_t chi) {
    const int64_t phys = 2;
    const auto A0 = physical_slice(A, chi, 0);
    const auto A1 = physical_slice(A, chi, 1);
    const std::vector<const double *> As = {A0.data(), A1.data()};
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    std::vector<double> tmp(static_cast<std::size_t>(chi * chi), 0.0);
    for (int64_t g = 0; g < phys; ++g) {
        blas_gemm('N', 'T', chi, chi, chi, 1.0, R.data(), chi, As[g], chi,
                  0.0, tmp.data(), chi);
        for (int64_t b = 0; b < phys; ++b) {
            const double alpha = O[idx2(g, b, phys)];
            if (alpha == 0.0) continue;
            blas_gemm('N', 'N', chi, chi, chi, alpha, As[b], chi, tmp.data(),
                      chi, 1.0, out.data(), chi);
        }
    }
    return out;
}

std::vector<double> matmul_nt(const std::vector<double> &A,
                              const std::vector<double> &B, int64_t chi) {
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    blas_gemm('N', 'T', chi, chi, chi, 1.0, A.data(), chi, B.data(), chi,
              0.0, out.data(), chi);
    return out;
}

std::vector<double> matmul_tn(const std::vector<double> &A,
                              const std::vector<double> &B, int64_t chi) {
    std::vector<double> out(static_cast<std::size_t>(chi * chi), 0.0);
    blas_gemm('T', 'N', chi, chi, chi, 1.0, A.data(), chi, B.data(), chi,
              0.0, out.data(), chi);
    return out;
}

double trace(const std::vector<double> &A, int64_t chi) {
    double t = 0.0;
    for (int64_t i = 0; i < chi; ++i) t += A[idx2(i, i, chi)];
    return t;
}

std::vector<double> right_density(const std::vector<double> &C, int64_t chi) {
    auto rho = matmul_nt(C, C, chi);
    const double tr = trace(rho, chi);
    for (double &v : rho) v /= tr;
    return rho;
}

std::vector<double> left_density(const std::vector<double> &C, int64_t chi) {
    auto rho = matmul_tn(C, C, chi);
    const double tr = trace(rho, chi);
    for (double &v : rho) v /= tr;
    return rho;
}

std::vector<double> add(const std::vector<double> &a, const std::vector<double> &b) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

std::vector<double> sub_scaled_identity(const std::vector<double> &a, double scale,
                                        int64_t chi) {
    std::vector<double> out = a;
    for (int64_t i = 0; i < chi; ++i) out[idx2(i, i, chi)] -= scale;
    return out;
}

std::vector<double> apply_physical(const std::vector<double> &A,
                                   const std::vector<double> &O, int64_t chi) {
    const int64_t phys = 2;
    std::vector<double> out(static_cast<std::size_t>(chi * phys * chi), 0.0);
    for (int64_t d = 0; d < chi; ++d)
        for (int64_t b = 0; b < phys; ++b)
            for (int64_t e = 0; e < chi; ++e)
                for (int64_t g = 0; g < phys; ++g)
                    out[idx3(d, b, e, chi, phys)] +=
                        A[idx3(d, g, e, chi, phys)] * O[idx2(g, b, phys)];
    return out;
}

void solve_least_squares(std::vector<double> &A, int64_t rows, int64_t cols,
                         std::vector<double> &b) {
    char trans = 'N';
    int64_t nrhs = 1;
    int64_t lda = rows;
    int64_t ldb = std::max(rows, cols);
    b.resize(static_cast<std::size_t>(ldb), 0.0);
    int64_t info = 0;
    int64_t lwork = -1;
    double work_query = 0.0;
    dgels_64_(&trans, &rows, &cols, &nrhs, A.data(), &lda, b.data(), &ldb,
              &work_query, &lwork, &info);
    if (info != 0) throw std::runtime_error("LAPACK dgels workspace query failed");
    lwork = std::max<int64_t>(1, static_cast<int64_t>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dgels_64_(&trans, &rows, &cols, &nrhs, A.data(), &lda, b.data(), &ldb,
              work.data(), &lwork, &info);
    if (info != 0) throw std::runtime_error("LAPACK dgels failed");
}

std::vector<double> projected_solve(const std::vector<double> &A,
                                    const std::vector<double> &O,
                                    const std::vector<double> &rho,
                                    const std::vector<double> &rhs,
                                    int64_t chi, double tol, int64_t max_k,
                                    bool right_side) {
    std::vector<double> x0(static_cast<std::size_t>(chi * chi), 0.0);
    auto r0 = rhs;
    const double beta0 = norm2(r0);
    if (beta0 <= tol * std::max(norm2(rhs), 1.0)) return x0;
    const auto AO = apply_physical(A, O, chi);
    const int64_t len = chi * chi;
    const int64_t kmax = std::max<int64_t>(1, std::min(max_k, len));
    std::vector<double> V(static_cast<std::size_t>(len * (kmax + 1)));
    std::vector<double> H(static_cast<std::size_t>((kmax + 1) * kmax), 0.0);
    double beta = 0.0;
    int64_t m = 0;
    double final_resnorm = 0.0;
    const int status = tenet_native_arnoldi_projected_two_layer_d_cpu(
        chi, 2, A.data(), AO.data(), rho.data(), r0.data(), kmax, tol,
        right_side ? 1 : 0, V.data(), len, H.data(), kmax + 1, &beta, &m,
        &final_resnorm);
    if (status != TENET_NATIVE_SUCCESS) {
        throw std::runtime_error(std::string("native projected Arnoldi failed: ") +
                                 tenet_native_last_error());
    }
    if (m <= 0) return x0;
    const int64_t rows = m + 1;
    std::vector<double> A_ls(static_cast<std::size_t>(rows * m));
    for (int64_t j = 0; j < m; ++j)
        for (int64_t i = 0; i < rows; ++i)
            A_ls[idx2(i, j, rows)] = H[idx2(i, j, kmax + 1)];
    std::vector<double> b(static_cast<std::size_t>(rows), 0.0);
    b[0] = beta0;
    solve_least_squares(A_ls, rows, m, b);
    std::vector<double> x = x0;
    for (int64_t j = 0; j < m; ++j)
        for (int64_t i = 0; i < len; ++i)
            x[i] += V[idx2(i, j, len)] * b[j];
    return x;
}

std::vector<double> stack_leg3(const std::vector<std::vector<double>> &mats,
                               int64_t chi) {
    const int64_t phys = static_cast<int64_t>(mats.size());
    std::vector<double> out(static_cast<std::size_t>(chi * phys * chi), 0.0);
    for (int64_t p = 0; p < phys; ++p)
        for (int64_t k = 0; k < chi; ++k)
            for (int64_t i = 0; i < chi; ++i)
                out[idx3(i, p, k, chi, phys)] = mats[p][idx2(i, k, chi)];
    return out;
}

std::vector<double> pad_mpo_permuted(const std::vector<double> &W) {
    std::vector<double> Mpad(3 * 3 * 3 * 3, 0.0);
    for (int64_t l = 0; l < 3; ++l)
        for (int64_t g = 0; g < 2; ++g)
            for (int64_t r = 0; r < 3; ++r)
                for (int64_t b = 0; b < 2; ++b)
                    Mpad[idx4(l, g, r, b, 3, 3, 3)] =
                        W[idx4(l, g, r, b, 3, 2, 3)];
    std::vector<double> Mp(3 * 3 * 3 * 3, 0.0);
    for (int64_t i = 0; i < 3; ++i)
        for (int64_t j = 0; j < 3; ++j)
            for (int64_t k = 0; k < 3; ++k)
                for (int64_t l = 0; l < 3; ++l)
                    Mp[idx4(i, j, k, l, 3, 3, 3)] =
                        Mpad[idx4(l, k, j, i, 3, 3, 3)];
    return Mp;
}

std::vector<double> pad_ac(const std::vector<double> &AC, int64_t chi) {
    std::vector<double> out(static_cast<std::size_t>(chi * 3 * chi), 0.0);
    for (int64_t k = 0; k < chi; ++k)
        for (int64_t p = 0; p < 2; ++p)
            for (int64_t i = 0; i < chi; ++i)
                out[idx3(i, p, k, chi, 3)] = AC[idx3(i, p, k, chi, 2)];
    return out;
}

std::vector<double> unpad_ac(const std::vector<double> &AC, int64_t chi) {
    std::vector<double> out(static_cast<std::size_t>(chi * 2 * chi), 0.0);
    for (int64_t k = 0; k < chi; ++k)
        for (int64_t p = 0; p < 2; ++p)
            for (int64_t i = 0; i < chi; ++i)
                out[idx3(i, p, k, chi, 2)] = AC[idx3(i, p, k, chi, 3)];
    return out;
}

std::vector<double> native_ac_eig(const State &state, const Env &env,
                                  const std::vector<double> &W, int64_t chi,
                                  double tol, int64_t max_k) {
    const auto FL = stack_leg3(env.left, chi);
    const auto FR = stack_leg3(env.right, chi);
    const auto Mp = pad_mpo_permuted(W);
    const auto AC0 = pad_ac(state.AC, chi);
    const int64_t phys = 3;
    const int64_t len = chi * phys * chi;
    const int64_t kmax = std::max<int64_t>(1, std::min(max_k, len));
    std::vector<double> ACp(static_cast<std::size_t>(len), 0.0);
    double lambda = 0.0;
    const int status = tenet_native_smallest_real_three_layer_leg4_d_cpu(
        chi, phys, FL.data(), FR.data(), Mp.data(), AC0.data(), kmax, tol, 0,
        ACp.data(), &lambda);
    if (status != TENET_NATIVE_SUCCESS) {
        throw std::runtime_error(std::string("native AC smallest-real failed: ") +
                                 tenet_native_last_error());
    }
    return unpad_ac(ACp, chi);
}

std::vector<double> native_c_eig(const State &state, const Env &env, int64_t chi,
                                 double tol, int64_t max_k) {
    const auto FL = stack_leg3(env.left, chi);
    const auto FR = stack_leg3(env.right, chi);
    const int64_t phys = 3;
    const int64_t len = chi * chi;
    const int64_t kmax = std::max<int64_t>(1, std::min(max_k, len));
    std::vector<double> Cp(static_cast<std::size_t>(len), 0.0);
    double lambda = 0.0;
    const int status = tenet_native_smallest_real_two_layer_d_cpu(
        chi, phys, FL.data(), FR.data(), state.C.data(), kmax, tol, 0,
        Cp.data(), &lambda);
    if (status != TENET_NATIVE_SUCCESS) {
        throw std::runtime_error(std::string("native C smallest-real failed: ") +
                                 tenet_native_last_error());
    }
    return Cp;
}

Env environments(const State &state, const std::vector<double> &W, int64_t chi,
                 double env_tol, int64_t env_krylovdim) {
    Env env;
    env.left.resize(3);
    env.right.resize(3);
    auto I = identity(chi);
    env.left[2] = I;
    env.left[1] = left_apply(I, state.AL, wslice(W, 2, 1), chi);
    auto rawL1 = add(left_apply(env.left[1], state.AL, wslice(W, 1, 0), chi),
                    left_apply(I, state.AL, wslice(W, 2, 0), chi));
    const auto rhoR = right_density(state.C, chi);
    const double eL = dot(rawL1, rhoR);
    const auto rhsL = sub_scaled_identity(rawL1, eL, chi);
    env.left[0] = projected_solve(state.AL, wslice(W, 0, 0), rhoR, rhsL, chi,
                                  env_tol, env_krylovdim, false);

    env.right[0] = I;
    env.right[1] = right_apply(I, state.AR, wslice(W, 1, 0), chi);
    auto rawR3 = add(right_apply(env.right[1], state.AR, wslice(W, 2, 1), chi),
                    right_apply(I, state.AR, wslice(W, 2, 0), chi));
    const auto rhoL = left_density(state.C, chi);
    const double eR = dot(rawR3, rhoL);
    const auto rhsR = sub_scaled_identity(rawR3, eR, chi);
    env.right[2] = projected_solve(state.AR, wslice(W, 2, 2), rhoL, rhsR, chi,
                                   env_tol, env_krylovdim, true);
    return env;
}

double energy_density(const State &state, const Env &env,
                      const std::vector<double> &W, int64_t chi) {
    const auto raw = add(left_apply(env.left[1], state.AL, wslice(W, 1, 0), chi),
                         left_apply(env.left[2], state.AL, wslice(W, 2, 0), chi));
    return dot(raw, right_density(state.C, chi));
}

std::vector<double> alc_to_ac(const std::vector<double> &AL,
                              const std::vector<double> &C, int64_t chi) {
    const int64_t phys = 2;
    std::vector<double> AC(static_cast<std::size_t>(chi * phys * chi), 0.0);
    for (int64_t b = 0; b < phys; ++b) {
        const auto ALb = physical_slice(AL, chi, b);
        std::vector<double> ACb(static_cast<std::size_t>(chi * chi), 0.0);
        blas_gemm('N', 'N', chi, chi, chi, 1.0, ALb.data(), chi, C.data(),
                  chi, 0.0, ACb.data(), chi);
        write_physical_slice(AC, ACb, chi, b);
    }
    return AC;
}

StepResult vumps_step(const State &state, const Env &env,
                      const std::vector<double> &W, int64_t chi,
                      double eig_tol, int64_t eig_krylovdim) {
    StepResult result;
    auto start = std::chrono::steady_clock::now();
    auto AC = native_ac_eig(state, env, W, chi, eig_tol, eig_krylovdim);
    auto stop = std::chrono::steady_clock::now();
    result.ac_seconds = std::chrono::duration<double>(stop - start).count();
    start = std::chrono::steady_clock::now();
    auto C = native_c_eig(state, env, chi, eig_tol, eig_krylovdim);
    stop = std::chrono::steady_clock::now();
    result.c_seconds = std::chrono::duration<double>(stop - start).count();
    normalize(C);
    std::vector<double> AL(static_cast<std::size_t>(chi * 2 * chi), 0.0);
    std::vector<double> AR(static_cast<std::size_t>(chi * 2 * chi), 0.0);
    double err = 0.0;
    start = std::chrono::steady_clock::now();
    const int status = tenet_native_acc_to_alar_d_cpu(
        chi, 2, AC.data(), C.data(), AL.data(), AR.data(), &err);
    stop = std::chrono::steady_clock::now();
    result.acc_seconds = std::chrono::duration<double>(stop - start).count();
    if (status != TENET_NATIVE_SUCCESS) {
        throw std::runtime_error(std::string("native ACCtoALAR failed: ") +
                                 tenet_native_last_error());
    }
    result.state.AL = std::move(AL);
    result.state.AR = std::move(AR);
    result.state.C = std::move(C);
    start = std::chrono::steady_clock::now();
    result.state.AC = alc_to_ac(result.state.AL, result.state.C, chi);
    stop = std::chrono::steady_clock::now();
    result.acnext_seconds = std::chrono::duration<double>(stop - start).count();
    result.err = err;
    return result;
}

State product_initial_state(int64_t chi) {
    State state;
    const int64_t phys = 2;
    state.AL.assign(static_cast<std::size_t>(chi * phys * chi), 0.0);
    state.AR.assign(static_cast<std::size_t>(chi * phys * chi), 0.0);
    state.C.assign(static_cast<std::size_t>(chi * chi), 0.0);
    const double a = 1.0 / std::sqrt(2.0);
    const double c = 1.0 / std::sqrt(static_cast<double>(chi));
    for (int64_t i = 0; i < chi; ++i) {
        state.AL[idx3(i, 0, i, chi, phys)] = a;
        state.AL[idx3(i, 1, i, chi, phys)] = a;
        state.AR[idx3(i, 0, i, chi, phys)] = a;
        state.AR[idx3(i, 1, i, chi, phys)] = a;
        state.C[idx2(i, i, chi)] = c;
    }
    state.AC = alc_to_ac(state.AL, state.C, chi);
    return state;
}

State canonical_state_from_ac_c(const std::vector<double> &AC,
                                const std::vector<double> &C, int64_t chi) {
    State state;
    std::vector<double> AL(static_cast<std::size_t>(chi * 2 * chi), 0.0);
    std::vector<double> AR(static_cast<std::size_t>(chi * 2 * chi), 0.0);
    double err = 0.0;
    const int status = tenet_native_acc_to_alar_d_cpu(
        chi, 2, AC.data(), C.data(), AL.data(), AR.data(), &err);
    if (status != TENET_NATIVE_SUCCESS) {
        throw std::runtime_error(std::string("native initial ACCtoALAR failed: ") +
                                 tenet_native_last_error());
    }
    state.AL = std::move(AL);
    state.AR = std::move(AR);
    state.C = C;
    state.AC = alc_to_ac(state.AL, state.C, chi);
    return state;
}

State random_initial_state(int64_t chi, int64_t seed) {
    const int64_t phys = 2;
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> AC(static_cast<std::size_t>(chi * phys * chi), 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(phys * chi));
    for (double &v : AC) v = scale * normal(rng);
    std::vector<double> C(static_cast<std::size_t>(chi * chi), 0.0);
    const double c = 1.0 / std::sqrt(static_cast<double>(chi));
    for (int64_t i = 0; i < chi; ++i) C[idx2(i, i, chi)] = c;
    return canonical_state_from_ac_c(AC, C, chi);
}

template <typename T> T read_scalar(std::ifstream &in) {
    T x{};
    in.read(reinterpret_cast<char *>(&x), sizeof(T));
    if (!in) throw std::runtime_error("failed while reading initial-state scalar");
    return x;
}

std::vector<double> read_vector(std::ifstream &in, std::size_t n) {
    std::vector<double> x(n);
    in.read(reinterpret_cast<char *>(x.data()),
            static_cast<std::streamsize>(n * sizeof(double)));
    if (!in) throw std::runtime_error("failed while reading initial-state vector");
    return x;
}

State load_initial_state(const std::string &path, int64_t expected_chi) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open initial-state path: " + path);
    char magic[8];
    in.read(magic, sizeof(magic));
    if (!in) {
        throw std::runtime_error("bad initial-state magic");
    }
    const int64_t version = read_scalar<int64_t>(in);
    const int64_t chi = read_scalar<int64_t>(in);
    const int64_t phys = read_scalar<int64_t>(in);
    if (std::memcmp(magic, "FTNTTFI0", 8) == 0) {
        if (version != 1 || chi != expected_chi || phys != 2) {
            throw std::runtime_error("initial-state metadata mismatch");
        }
        State state;
        const std::size_t len3 = static_cast<std::size_t>(chi * phys * chi);
        const std::size_t len2 = static_cast<std::size_t>(chi * chi);
        state.AL = read_vector(in, len3);
        state.AR = read_vector(in, len3);
        state.C = read_vector(in, len2);
        state.AC = read_vector(in, len3);
        return state;
    }
    if (std::memcmp(magic, "FTNTTFI1", 8) != 0) {
        throw std::runtime_error("bad initial-state magic");
    }
    const int64_t mpo = read_scalar<int64_t>(in);
    (void)read_scalar<double>(in);  // field
    (void)read_scalar<int64_t>(in); // maxiter
    (void)read_scalar<int64_t>(in); // miniter
    (void)read_scalar<int64_t>(in); // max_k
    (void)read_scalar<double>(in);  // eig_tol
    (void)read_scalar<double>(in);  // env_tol
    (void)read_scalar<double>(in);  // tol
    (void)read_scalar<double>(in);  // energy
    (void)read_scalar<double>(in);  // err
    (void)read_scalar<int64_t>(in); // iterations
    (void)read_scalar<int64_t>(in); // converged
    if (version != 1 || chi != expected_chi || phys != 2 || mpo != 3) {
        throw std::runtime_error("full dump metadata mismatch");
    }
    (void)read_vector(in, static_cast<std::size_t>(mpo * phys * mpo * phys));
    const std::size_t len3 = static_cast<std::size_t>(chi * phys * chi);
    const std::size_t len2 = static_cast<std::size_t>(chi * chi);
    (void)read_vector(in, len3); // initial AL
    (void)read_vector(in, len3); // initial AR
    (void)read_vector(in, len2); // initial C
    (void)read_vector(in, len3); // initial AC
    State state;
    state.AL = read_vector(in, len3);
    state.AR = read_vector(in, len3);
    state.C = read_vector(in, len2);
    state.AC = read_vector(in, len3);
    return state;
}

RunResult run_tfising(const Options &opt, const std::vector<double> &W,
                      const State &initial) {
    RunResult result;
    result.state = initial;
    auto start = std::chrono::steady_clock::now();
    result.env = environments(result.state, W, opt.chi, opt.env_tol, opt.max_k);
    auto stop = std::chrono::steady_clock::now();
    result.env_seconds += std::chrono::duration<double>(stop - start).count();
    start = std::chrono::steady_clock::now();
    result.energy = energy_density(result.state, result.env, W, opt.chi);
    stop = std::chrono::steady_clock::now();
    result.energy_seconds += std::chrono::duration<double>(stop - start).count();
    result.err = std::numeric_limits<double>::infinity();
    for (int64_t iter = 1; iter <= opt.maxiter; ++iter) {
        start = std::chrono::steady_clock::now();
        auto step = vumps_step(result.state, result.env, W, opt.chi, opt.eig_tol,
                              opt.max_k);
        stop = std::chrono::steady_clock::now();
        result.step_seconds += std::chrono::duration<double>(stop - start).count();
        result.ac_seconds += step.ac_seconds;
        result.c_seconds += step.c_seconds;
        result.acc_seconds += step.acc_seconds;
        result.acnext_seconds += step.acnext_seconds;
        result.state = std::move(step.state);
        result.err = step.err;
        result.iterations = iter;
        start = std::chrono::steady_clock::now();
        result.env = environments(result.state, W, opt.chi, opt.env_tol, opt.max_k);
        stop = std::chrono::steady_clock::now();
        result.env_seconds += std::chrono::duration<double>(stop - start).count();
        start = std::chrono::steady_clock::now();
        result.energy = energy_density(result.state, result.env, W, opt.chi);
        stop = std::chrono::steady_clock::now();
        result.energy_seconds += std::chrono::duration<double>(stop - start).count();
        if (result.err < opt.tol && iter >= opt.miniter) {
            result.converged = 1;
            break;
        }
    }
    if (opt.maxiter == 0) result.converged = 1;
    return result;
}

template <typename T> void write_scalar(std::ofstream &out, const T &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(T));
}

void write_vector(std::ofstream &out, const std::vector<double> &x) {
    out.write(reinterpret_cast<const char *>(x.data()),
              static_cast<std::streamsize>(x.size() * sizeof(double)));
}

void dump_state(const Options &opt, const std::vector<double> &W,
                const State &initial, const RunResult &result) {
    if (opt.dump_state.empty()) return;
    std::ofstream out(opt.dump_state, std::ios::binary);
    if (!out) throw std::runtime_error("failed to open dump path: " + opt.dump_state);
    const char magic[8] = {'F', 'T', 'N', 'T', 'T', 'F', 'I', '1'};
    out.write(magic, sizeof(magic));
    const int64_t version = 1;
    const int64_t phys = 2;
    const int64_t mpo = 3;
    write_scalar(out, version);
    write_scalar(out, opt.chi);
    write_scalar(out, phys);
    write_scalar(out, mpo);
    write_scalar(out, opt.field);
    write_scalar(out, opt.maxiter);
    write_scalar(out, opt.miniter);
    write_scalar(out, opt.max_k);
    write_scalar(out, opt.eig_tol);
    write_scalar(out, opt.env_tol);
    write_scalar(out, opt.tol);
    write_scalar(out, result.energy);
    write_scalar(out, result.err);
    write_scalar(out, result.iterations);
    write_scalar(out, static_cast<int64_t>(result.converged));
    write_vector(out, W);
    write_vector(out, initial.AL);
    write_vector(out, initial.AR);
    write_vector(out, initial.C);
    write_vector(out, initial.AC);
    write_vector(out, result.state.AL);
    write_vector(out, result.state.AR);
    write_vector(out, result.state.C);
    write_vector(out, result.state.AC);
    if (!out) throw std::runtime_error("failed while writing dump path");
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_options(argc, argv);
        apply_arnoldi_restarts(opt);
        const int64_t arnoldi_restarts = effective_arnoldi_restarts(opt);
        std::vector<double> W;
        fill_tfising_mpo(opt.field, W);
        const State initial = opt.load_initial_state.empty()
                                  ? (opt.init == "product"
                                         ? product_initial_state(opt.chi)
                                         : random_initial_state(opt.chi,
                                                                opt.seed))
                                  : load_initial_state(opt.load_initial_state,
                                                       opt.chi);

        auto run_once = [&]() { return run_tfising(opt, W, initial); };
        for (int64_t rep = 0; rep < opt.warmup; ++rep) {
            (void)run_once();
        }
        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(opt.repetitions));
        RunResult result;
        for (int64_t rep = 0; rep < opt.repetitions; ++rep) {
            const auto start = std::chrono::steady_clock::now();
            result = run_once();
            const auto stop = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double>(stop - start).count());
        }
        dump_state(opt, W, initial, result);
        std::cout << std::setprecision(17)
                  << "status=success backend=cpu_tfising"
                  << " field=" << opt.field << " chi=" << opt.chi
                  << " maxiter=" << opt.maxiter << " max_k=" << opt.max_k
                  << " krylovdim=" << opt.max_k
                  << " arnoldi_restarts=" << arnoldi_restarts
                  << " init=" << (opt.load_initial_state.empty()
                                      ? opt.init
                                      : "loaded")
                  << " seed=" << opt.seed
                  << " warmup=" << opt.warmup
                  << " repetitions=" << opt.repetitions
                  << " native_run_seconds_min="
                  << *std::min_element(timings.begin(), timings.end())
                  << " native_run_seconds_median=" << median(timings)
                  << " native_run_seconds_last=" << timings.back()
                  << " iterations=" << result.iterations
                  << " converged=" << result.converged
                  << " err=" << result.err << " energy=" << result.energy
                  << " env_seconds=" << result.env_seconds
                  << " step_seconds=" << result.step_seconds
                  << " ac_seconds=" << result.ac_seconds
                  << " c_seconds=" << result.c_seconds
                  << " acc_seconds=" << result.acc_seconds
                  << " acnext_seconds=" << result.acnext_seconds
                  << " energy_seconds=" << result.energy_seconds
                  << " norm_AL=" << norm2(result.state.AL)
                  << " norm_AR=" << norm2(result.state.AR)
                  << " norm_C=" << norm2(result.state.C)
                  << " norm_AC=" << norm2(result.state.AC)
                  << " sum_AL=" << sum(result.state.AL)
                  << " sum_AR=" << sum(result.state.AR)
                  << " sum_C=" << sum(result.state.C)
                  << " sum_AC=" << sum(result.state.AC) << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        usage(argv[0]);
        return 64;
    }
}
