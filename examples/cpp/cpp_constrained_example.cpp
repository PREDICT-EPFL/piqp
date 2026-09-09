#include <iostream>
#include "piqp/constrained/solver.hpp"

int main()
{
    piqp::Mat<double> P = piqp::Mat<double>::Identity(3, 3);
    piqp::Vec<double> c(3); c << -2, -2, -1;
    piqp::dense::Model<double> model(P, c);
    piqp::Mat<double> Q(1, 1); Q << 2;
    model.quadratic_constraints.emplace_back(Q, piqp::Vec<double>::Zero(1), 0.25,
                                             std::vector<piqp::isize>{0});
    piqp::Mat<double> F = piqp::Mat<double>::Zero(3, 2);
    F.bottomRows(2).setIdentity();
    piqp::Vec<double> f(3); f << 1, 0, 0;
    model.cone_constraints.emplace_back(F, f, piqp::ConeType::second_order,
                                       std::vector<piqp::isize>{0, 1});
    piqp::ConstrainedDenseSolver<double> solver;
    solver.setup(model);
    if (solver.solve() != piqp::PIQP_SOLVED) return 1;
    std::cout << solver.result().x.transpose() << '\n';
}
