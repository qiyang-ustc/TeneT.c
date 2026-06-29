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

namespace {

int64_t idx2(int64_t i, int64_t j, int64_t n1) { return i + n1 * j; }

int64_t idx4(int64_t i, int64_t j, int64_t k, int64_t l, int64_t n1,
             int64_t n2, int64_t n3) {
    return i + n1 * (j + n2 * (k + n3 * l));
}

struct Options {
    double beta = std::log1p(std::sqrt(2.0)) / 2.0;
    int64_t chi = 8;
    int64_t maxiter = 1;
    int64_t miniter = 1;
    int64_t max_k = 30;
    int64_t arnoldi_restarts = 0;
    double arnoldi_tol = 1e-12;
    double tol = 1e-10;
    uint64_t seed = 20260625ULL;
    int64_t repetitions = 1;
    int64_t warmup = 0;
    std::string init = "random";
    int64_t init_relax = 0;
    double init_relax_arnoldi_tol = 1e-8;
    std::string dump_state;
    std::string load_initial_state;
};

void usage(const char *argv0) {
    std::cerr
        << "usage: " << argv0
        << " [--beta x] [--chi n] [--maxiter n] [--miniter n]\n"
        << "       [--krylovdim n|--max-k n] [--arnoldi-restarts n]\n"
        << "       [--arnoldi-tol x] [--tol x]\n"
        << "       [--seed n] [--init random|native-canonical]\n"
        << "       [--init-relax n]\n"
        << "       [--init-relax-arnoldi-tol x]\n"
        << "       [--warmup n] [--repetitions n]\n"
        << "       [--dump-state path] [--load-initial-state path]\n";
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
        if (consume_arg(i, argc, argv, "--beta", value)) {
            opt.beta = std::stod(value);
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
        } else if (consume_arg(i, argc, argv, "--arnoldi-tol", value)) {
            opt.arnoldi_tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--tol", value)) {
            opt.tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--seed", value)) {
            opt.seed = std::stoull(value);
        } else if (consume_arg(i, argc, argv, "--init", value)) {
            opt.init = value;
        } else if (consume_arg(i, argc, argv, "--init-relax", value)) {
            opt.init_relax = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--init-relax-arnoldi-tol", value)) {
            opt.init_relax_arnoldi_tol = std::stod(value);
        } else if (consume_arg(i, argc, argv, "--warmup", value)) {
            opt.warmup = std::stoll(value);
        } else if (consume_arg(i, argc, argv, "--repetitions", value)) {
            opt.repetitions = std::stoll(value);
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
    if (opt.chi <= 0 || opt.maxiter < 0 || opt.miniter < 0 ||
        opt.miniter > opt.maxiter || opt.arnoldi_tol < 0.0 ||
        opt.tol < 0.0 || opt.arnoldi_restarts < 0 || opt.init_relax < 0 ||
        opt.init_relax_arnoldi_tol < 0.0 || opt.warmup < 0 ||
        opt.repetitions <= 0) {
        throw std::invalid_argument("invalid dimensions or tolerances");
    }
    if (opt.init != "random" && opt.init != "native-canonical") {
        throw std::invalid_argument("unsupported --init value");
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

void fill_bulk_ising_tensor(double beta, std::vector<double> &M) {
    M.assign(16, 0.0);
    const double ham[2][2] = {{-1.0, 1.0}, {1.0, -1.0}};
    double wsq[2][2];
    for (int s = 0; s < 2; ++s) {
        for (int a = 0; a < 2; ++a) {
            wsq[s][a] = std::sqrt(std::exp(-beta * ham[s][a]));
        }
    }
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    double total = 0.0;
                    for (int s = 0; s < 2; ++s) {
                        total += wsq[s][a] * wsq[s][b] * wsq[s][c] *
                                 wsq[s][d];
                    }
                    M[idx4(a, b, c, d, 2, 2, 2)] = total;
                }
            }
        }
    }
}

double norm2(const std::vector<double> &x) {
    double acc = 0.0;
    for (double v : x) {
        acc += v * v;
    }
    return std::sqrt(acc);
}

double sum(const std::vector<double> &x) {
    double acc = 0.0;
    for (double v : x) {
        acc += v;
    }
    return acc;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void normalize(std::vector<double> &x) {
    const double n = norm2(x);
    if (!(n > 0.0) || !std::isfinite(n)) {
        throw std::runtime_error("cannot normalize invalid vector");
    }
    for (double &v : x) {
        v /= n;
    }
}

void fill_random(std::vector<double> &x, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    for (double &v : x) {
        v = dist(rng);
    }
    normalize(x);
}

void fill_identity_c(int64_t chi, std::vector<double> &C) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int64_t i = 0; i < chi; ++i) {
        C[idx2(i, i, chi)] = 1.0;
    }
    normalize(C);
}

void init_random_state(int64_t chi, int64_t phys, std::mt19937_64 &rng,
                       std::vector<double> &AL, std::vector<double> &AR,
                       std::vector<double> &C, std::vector<double> &FL,
                       std::vector<double> &FR) {
    (void)phys;
    fill_random(AL, rng);
    fill_random(AR, rng);
    fill_identity_c(chi, C);
    fill_random(FL, rng);
    fill_random(FR, rng);
}

void check_native_status(int status, const char *context) {
    if (status == TENET_NATIVE_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(context) + " failed: " +
                             tenet_native_status_string(status) + ": " +
                             tenet_native_last_error());
}

void init_native_canonical_state(int64_t chi, int64_t phys,
                                 const std::vector<double> &M,
                                 int64_t max_k, double arnoldi_tol,
                                 std::mt19937_64 &rng,
                                 std::vector<double> &AL,
                                 std::vector<double> &AR,
                                 std::vector<double> &C,
                                 std::vector<double> &FL,
                                 std::vector<double> &FR) {
    const int64_t len3 = chi * phys * chi;
    std::vector<double> AC(static_cast<std::size_t>(len3));
    fill_random(AC, rng);
    fill_identity_c(chi, C);
    double canonical_err = 0.0;
    check_native_status(
        tenet_native_acc_to_alar_d_cpu(chi, phys, AC.data(), C.data(),
                                       AL.data(), AR.data(), &canonical_err),
        "native ACCtoALAR initializer");

    const int64_t env_k = std::max<int64_t>(1, std::min<int64_t>(max_k, len3));
    std::vector<double> out(static_cast<std::size_t>(len3));
    double lambda = 0.0;
    fill_random(FL, rng);
    check_native_status(tenet_native_dominant_three_layer_leg4_d_cpu(
                            chi, phys, AL.data(), AL.data(), M.data(),
                            FL.data(), env_k, arnoldi_tol, 0, out.data(),
                            &lambda),
                        "native FL initializer");
    FL = out;
    fill_random(FR, rng);
    check_native_status(tenet_native_dominant_three_layer_leg4_d_cpu(
                            chi, phys, AR.data(), AR.data(), M.data(),
                            FR.data(), env_k, arnoldi_tol, 1, out.data(),
                            &lambda),
                        "native FR initializer");
    FR = out;
}

void relax_initial_state(const Options &opt, int64_t phys,
                         const std::vector<double> &M,
                         std::vector<double> &AL, std::vector<double> &AR,
                         std::vector<double> &C, std::vector<double> &FL,
                         std::vector<double> &FR) {
    if (opt.init_relax == 0) {
        return;
    }
    double err = 0.0;
    int64_t iterations = 0;
    int converged = 0;
    check_native_status(tenet_native_ising_vumps_run_d_cpu(
                            opt.chi, phys, M.data(), AL.data(), AR.data(),
                            C.data(), FL.data(), FR.data(), opt.max_k,
                            opt.init_relax_arnoldi_tol, opt.tol, opt.init_relax,
                            opt.init_relax, &err, &iterations, &converged),
                        "native init relaxation");
}

template <typename T> void write_scalar(std::ofstream &out, const T &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(T));
}

void write_vector(std::ofstream &out, const std::vector<double> &x) {
    out.write(reinterpret_cast<const char *>(x.data()),
              static_cast<std::streamsize>(x.size() * sizeof(double)));
}

template <typename T> T read_scalar(std::ifstream &in) {
    T x{};
    in.read(reinterpret_cast<char *>(&x), sizeof(T));
    if (!in) {
        throw std::runtime_error("failed while reading scalar");
    }
    return x;
}

void read_vector(std::ifstream &in, std::vector<double> &x) {
    in.read(reinterpret_cast<char *>(x.data()),
            static_cast<std::streamsize>(x.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("failed while reading vector");
    }
}

struct LoadedState {
    double beta = 0.0;
    uint64_t seed = 0;
    int64_t maxiter = 0;
    int64_t miniter = 0;
    int64_t max_k = 0;
    double arnoldi_tol = 0.0;
    double tol = 0.0;
    std::vector<double> M;
    std::vector<double> AL;
    std::vector<double> AR;
    std::vector<double> C;
    std::vector<double> FL;
    std::vector<double> FR;
};

LoadedState load_state(const std::string &path, int64_t expected_chi) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open load path: " + path);
    }
    char magic[8];
    in.read(magic, sizeof(magic));
    const char expected_magic[8] = {'F', 'T', 'N', 'T', 'I', 'S', 'G', '1'};
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid Ising state magic in " + path);
    }
    const int64_t version = read_scalar<int64_t>(in);
    const int64_t chi = read_scalar<int64_t>(in);
    const int64_t phys = read_scalar<int64_t>(in);
    if (version != 1 || phys != 2 || chi != expected_chi) {
        throw std::runtime_error("incompatible Ising state header in " + path);
    }
    LoadedState s;
    s.beta = read_scalar<double>(in);
    s.seed = read_scalar<uint64_t>(in);
    s.maxiter = read_scalar<int64_t>(in);
    s.miniter = read_scalar<int64_t>(in);
    s.max_k = read_scalar<int64_t>(in);
    s.arnoldi_tol = read_scalar<double>(in);
    s.tol = read_scalar<double>(in);
    (void)read_scalar<double>(in);  // err
    (void)read_scalar<int64_t>(in); // iterations
    (void)read_scalar<int>(in);     // converged
    const int64_t len2 = chi * chi;
    const int64_t len3 = chi * phys * chi;
    s.M.resize(16);
    s.AL.resize(static_cast<std::size_t>(len3));
    s.AR.resize(static_cast<std::size_t>(len3));
    s.C.resize(static_cast<std::size_t>(len2));
    s.FL.resize(static_cast<std::size_t>(len3));
    s.FR.resize(static_cast<std::size_t>(len3));
    read_vector(in, s.M);
    read_vector(in, s.AL);
    read_vector(in, s.AR);
    read_vector(in, s.C);
    read_vector(in, s.FL);
    read_vector(in, s.FR);
    return s;
}

void dump_state(const Options &opt, double err, int64_t iterations,
                int converged, const std::vector<double> &M,
                const std::vector<double> &AL0, const std::vector<double> &AR0,
                const std::vector<double> &C0, const std::vector<double> &FL0,
                const std::vector<double> &FR0, const std::vector<double> &AL,
                const std::vector<double> &AR, const std::vector<double> &C,
                const std::vector<double> &FL, const std::vector<double> &FR) {
    if (opt.dump_state.empty()) {
        return;
    }
    std::ofstream out(opt.dump_state, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open dump path: " + opt.dump_state);
    }
    const char magic[8] = {'F', 'T', 'N', 'T', 'I', 'S', 'G', '1'};
    out.write(magic, sizeof(magic));
    const int64_t version = 1;
    const int64_t phys = 2;
    write_scalar(out, version);
    write_scalar(out, opt.chi);
    write_scalar(out, phys);
    write_scalar(out, opt.beta);
    write_scalar(out, opt.seed);
    write_scalar(out, opt.maxiter);
    write_scalar(out, opt.miniter);
    write_scalar(out, opt.max_k);
    write_scalar(out, opt.arnoldi_tol);
    write_scalar(out, opt.tol);
    write_scalar(out, err);
    write_scalar(out, iterations);
    write_scalar(out, converged);
    write_vector(out, M);
    write_vector(out, AL0);
    write_vector(out, AR0);
    write_vector(out, C0);
    write_vector(out, FL0);
    write_vector(out, FR0);
    write_vector(out, AL);
    write_vector(out, AR);
    write_vector(out, C);
    write_vector(out, FL);
    write_vector(out, FR);
    if (!out) {
        throw std::runtime_error("failed while writing dump path: " +
                                 opt.dump_state);
    }
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opt = parse_options(argc, argv);
        apply_arnoldi_restarts(opt);
        const int64_t arnoldi_restarts = effective_arnoldi_restarts(opt);
        const int64_t phys = 2;
        const int64_t len2 = opt.chi * opt.chi;
        const int64_t len3 = opt.chi * phys * opt.chi;
        if (opt.max_k <= 0) {
            opt.max_k = std::max(len2, len3);
        }

        std::vector<double> M;
        fill_bulk_ising_tensor(opt.beta, M);

        std::mt19937_64 rng(opt.seed);
        std::vector<double> AL(len3), AR(len3), C(len2), FL(len3), FR(len3);
        std::string init_kind = opt.init;
        if (opt.init == "native-canonical") {
            init_native_canonical_state(opt.chi, phys, M, opt.max_k,
                                        opt.arnoldi_tol, rng, AL, AR, C, FL,
                                        FR);
        } else {
            init_random_state(opt.chi, phys, rng, AL, AR, C, FL, FR);
        }
        if (!opt.load_initial_state.empty()) {
            LoadedState loaded = load_state(opt.load_initial_state, opt.chi);
            opt.beta = loaded.beta;
            opt.seed = loaded.seed;
            M = std::move(loaded.M);
            AL = std::move(loaded.AL);
            AR = std::move(loaded.AR);
            C = std::move(loaded.C);
            FL = std::move(loaded.FL);
            FR = std::move(loaded.FR);
            init_kind = "loaded";
        }
        relax_initial_state(opt, phys, M, AL, AR, C, FL, FR);

        const std::vector<double> AL0 = AL;
        const std::vector<double> AR0 = AR;
        const std::vector<double> C0 = C;
        const std::vector<double> FL0 = FL;
        const std::vector<double> FR0 = FR;

        double err = 0.0;
        int64_t iterations = 0;
        int converged = 0;
        auto reset_state = [&]() {
            AL = AL0;
            AR = AR0;
            C = C0;
            FL = FL0;
            FR = FR0;
        };
        auto run_once = [&]() {
            return tenet_native_ising_vumps_run_d_cpu(
                opt.chi, phys, M.data(), AL.data(), AR.data(), C.data(),
                FL.data(), FR.data(), opt.max_k, opt.arnoldi_tol, opt.tol,
                opt.miniter, opt.maxiter, &err, &iterations, &converged);
        };
        auto report_status = [](int status) {
            std::cerr << "native status=" << status << " ("
                      << tenet_native_status_string(status)
                      << "): " << tenet_native_last_error() << "\n";
        };
        for (int64_t rep = 0; rep < opt.warmup; ++rep) {
            reset_state();
            const int status = run_once();
            if (status != TENET_NATIVE_SUCCESS) {
                report_status(status);
                return status;
            }
        }
        std::vector<double> timings;
        timings.reserve(static_cast<std::size_t>(opt.repetitions));
        for (int64_t rep = 0; rep < opt.repetitions; ++rep) {
            reset_state();
            const auto start = std::chrono::steady_clock::now();
            const int status = run_once();
            const auto stop = std::chrono::steady_clock::now();
            if (status != TENET_NATIVE_SUCCESS) {
                report_status(status);
                return status;
            }
            const std::chrono::duration<double> elapsed = stop - start;
            timings.push_back(elapsed.count());
        }
        const double min_seconds =
            *std::min_element(timings.begin(), timings.end());
        const double median_seconds = median(timings);
        const double last_seconds = timings.back();

        dump_state(opt, err, iterations, converged, M, AL0, AR0, C0, FL0, FR0,
                   AL, AR, C, FL, FR);

        std::cout << std::setprecision(17)
                  << "status=success"
                  << " beta=" << opt.beta << " chi=" << opt.chi
                  << " maxiter=" << opt.maxiter
                  << " max_k=" << opt.max_k
                  << " krylovdim=" << opt.max_k
                  << " arnoldi_restarts=" << arnoldi_restarts
                  << " init=" << init_kind
                  << " init_relax=" << opt.init_relax
                  << " init_relax_arnoldi_tol=" << opt.init_relax_arnoldi_tol
                  << " warmup=" << opt.warmup
                  << " repetitions=" << opt.repetitions
                  << " native_run_seconds_min=" << min_seconds
                  << " native_run_seconds_median=" << median_seconds
                  << " native_run_seconds_last=" << last_seconds
                  << " iterations=" << iterations
                  << " converged=" << converged << " err=" << err
                  << " norm_AL=" << norm2(AL) << " norm_AR=" << norm2(AR)
                  << " norm_C=" << norm2(C) << " norm_FL=" << norm2(FL)
                  << " norm_FR=" << norm2(FR) << " sum_AL=" << sum(AL)
                  << " sum_AR=" << sum(AR) << " sum_C=" << sum(C)
                  << " sum_FL=" << sum(FL) << " sum_FR=" << sum(FR) << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << "\n";
        usage(argv[0]);
        return 64;
    }
}
