#include <chrono>
#include <iostream>
#include "piqp/solver.hpp"

using namespace piqp;
using Clock = std::chrono::steady_clock;

template<typename Solver, typename Matrix>
void measure(const char* backend, const Matrix& P, const Matrix& G, int repetitions)
{
    const isize n = P.rows();
    Vec<double> c = Vec<double>::LinSpaced(n, -2, 2);
    Vec<double> lower = Vec<double>::Constant(n, -1), upper = -lower;
    Solver solver;
    solver.settings().eps_abs = solver.settings().eps_rel = 1e-10;
    solver.settings().eps_duality_gap_abs = solver.settings().eps_duality_gap_rel = 1e-10;
    solver.setup(P, c, nullopt, nullopt, G, lower, upper);
    for (int i = -10; i < repetitions; ++i)
    {
        const auto start = Clock::now();
        const Status status = solver.solve();
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        if (status != PIQP_SOLVED) throw std::runtime_error("QP regression solve failed");
        const Vec<double> expected = (-c).cwiseMax(-1).cwiseMin(1);
        if ((solver.result().x - expected).template lpNorm<Eigen::Infinity>() > 1e-6)
            throw std::runtime_error("QP regression solution failed");
        if (i >= 0) std::cout << backend << ',' << n << ',' << i << ',' << elapsed << '\n';
    }
}

int main(int argc, char** argv)
{
    const int repetitions = argc > 1 ? std::stoi(argv[1]) : 200;
    std::cout.precision(17);
    std::cout << "backend,n,repetition,solve_s\n";
    for (int n : {8, 32, 128})
    {
        Mat<double> dense = Mat<double>::Identity(n, n);
        SparseMat<double, int> sparse = dense.sparseView();
        measure<DenseSolver<double>>("dense", dense, dense, repetitions);
        measure<SparseSolver<double>>("sparse", sparse, sparse, repetitions);
    }
}
