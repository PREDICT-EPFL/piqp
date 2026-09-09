#ifndef PIQP_CONSTRAINED_CONSTRAINTS_HPP
#define PIQP_CONSTRAINED_CONSTRAINTS_HPP

#include <vector>
#include <utility>
#include "piqp/typedefs.hpp"

namespace piqp
{

enum class ConeType { second_order, rotated_second_order };

template<typename T, typename Matrix = Mat<T>>
struct QuadraticConstraint
{
    Matrix Q;
    Vec<T> q;
    T upper;
    std::vector<isize> indices;

    QuadraticConstraint(const Matrix& Q, const Vec<T>& q, T upper,
                        std::vector<isize> indices = {})
        : Q(Q), q(q), upper(upper), indices(std::move(indices)) {}
};

template<typename T, typename Matrix = Mat<T>>
struct ConeConstraint
{
    Matrix F;
    Vec<T> f;
    ConeType type;
    std::vector<isize> indices;

    ConeConstraint(const Matrix& F, const Vec<T>& f,
                   ConeType type = ConeType::second_order,
                   std::vector<isize> indices = {})
        : F(F), f(f), type(type), indices(std::move(indices)) {}
};

}

#endif
