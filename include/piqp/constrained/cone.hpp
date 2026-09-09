// This file is part of PIQP.
//
// This source code is licensed under the BSD 2-Clause License found in the
// LICENSE file in the root directory of this source tree.

#ifndef PIQP_CONSTRAINED_CONE_HPP
#define PIQP_CONSTRAINED_CONE_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "piqp/typedefs.hpp"

namespace piqp
{
namespace constrained
{

template<typename T>
void jordan(CVecRef<T> a, CVecRef<T> b, VecRef<T> out)
{
    const T a0 = a(0), b0 = b(0);
    out(0) = a.dot(b);
    out.tail(a.size() - 1) = a0 * b.tail(b.size() - 1) + b0 * a.tail(a.size() - 1);
}

template<typename T>
Vec<T> jordan(CVecRef<T> a, CVecRef<T> b)
{
    Vec<T> out(a.size());
    jordan<T>(a, b, out);
    return out;
}

template<typename T>
void jordan_solve(CVecRef<T> a, CVecRef<T> b, VecRef<T> out)
{
    const T scale = a.cwiseAbs().maxCoeff();
    const T a0 = a(0) / scale;
    const T norm = a.tail(a.size() - 1).stableNorm() / scale;
    T product = T(0);
    for (Eigen::Index i = 1; i < a.size(); ++i) product += (a(i) / scale) * (b(i) / scale);
    out(0) = (a0 * (b(0) / scale) - product) / ((a0 - norm) * (a0 + norm));
    for (Eigen::Index i = 1; i < a.size(); ++i) out(i) = (b(i) / scale - (a(i) / scale) * out(0)) / a0;
}

template<typename T>
Vec<T> jordan_solve(CVecRef<T> a, CVecRef<T> b)
{
    Vec<T> out(a.size());
    jordan_solve<T>(a, b, out);
    return out;
}

template<typename T>
bool interior(CVecRef<T> v)
{
    return v.size() > 0 && v.allFinite() && v(0) > v.tail(v.size() - 1).stableNorm();
}

template<typename T>
void repair(VecRef<T> v, T margin = T(1))
{
    const T norm = v.tail(v.size() - 1).stableNorm();
    v(0) = (std::max)(v(0), (std::max)(norm + margin, std::nextafter(norm, std::numeric_limits<T>::infinity())));
}

template<typename T>
void repair(Vec<T>& v, T margin = T(1))
{
    repair<T>(VecRef<T>(v), margin);
}

template<typename T>
T max_step(CVecRef<T> v, CVecRef<T> d)
{
    const T vs = v.cwiseAbs().maxCoeff();
    const T ds = d.cwiseAbs().maxCoeff();
    if (ds == T(0)) return T(1);
    const T v0 = v(0) / vs, d0 = d(0) / ds;
    const T vn = v.tail(v.size() - 1).stableNorm() / vs;
    const T dn = d.tail(d.size() - 1).stableNorm() / ds;
    const T a = (d0 - dn) * (d0 + dn);
    const T c = (v0 - vn) * (v0 + vn);
    T b = v0 * d0;
    for (Eigen::Index i = 1; i < v.size(); ++i) b -= (v(i) / vs) * (d(i) / ds);
    T root = std::numeric_limits<T>::infinity();
    if (d0 < T(0)) root = -v0 / d0;
    if (a == T(0))
    {
        if (b < T(0)) root = (std::min)(root, -c / (T(2) * b));
    }
    else
    {
        const T disc = b * b - a * c;
        if (disc >= T(0))
        {
            const T q = -b - std::copysign(std::sqrt(disc), b);
            const T r1 = q / a;
            const T r2 = q == T(0) ? root : c / q;
            if (r1 > T(0)) root = (std::min)(root, r1);
            if (r2 > T(0)) root = (std::min)(root, r2);
        }
    }
    return root < ds / vs ? (std::max)(T(0), root * (vs / ds)) : T(1);
}


} // namespace constrained
} // namespace piqp

#endif // PIQP_CONSTRAINED_CONE_HPP
