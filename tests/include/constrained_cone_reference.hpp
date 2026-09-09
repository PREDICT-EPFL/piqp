#ifndef PIQP_TEST_CONSTRAINED_CONE_REFERENCE_HPP
#define PIQP_TEST_CONSTRAINED_CONE_REFERENCE_HPP

#include "piqp/constrained/cone.hpp"

namespace piqp
{
namespace test
{

template<typename T>
class ConeScaling
{
    Vec<T> m_s, m_z, m_w;

public:
    Mat<T> W, Winv;
    Vec<T> lambda;

    explicit ConeScaling(Eigen::Index dimension)
        : m_s(dimension), m_z(dimension), m_w(dimension),
          W(dimension, dimension), Winv(dimension, dimension), lambda(dimension)
    {}

    bool compute(CVecRef<T> s, CVecRef<T> z)
    {
        if (!constrained::interior<T>(s) || !constrained::interior<T>(z)) return false;
        const T ss = s.cwiseAbs().maxCoeff(), zs = z.cwiseAbs().maxCoeff();
        m_s = s / ss;
        m_z = z / zs;
        const T sn = m_s.tail(s.size() - 1).stableNorm();
        const T zn = m_z.tail(z.size() - 1).stableNorm();
        const T sr = std::sqrt((m_s(0) - sn) * (m_s(0) + sn));
        const T zr = std::sqrt((m_z(0) - zn) * (m_z(0) + zn));
        if (!(sr > T(0)) || !(zr > T(0))) return false;
        m_s /= sr;
        m_z /= zr;
        const T denominator = std::sqrt(T(2) * (T(1) + m_s.dot(m_z)));
        m_w = m_s - m_z;
        m_w(0) = m_s(0) + m_z(0);
        m_w /= denominator;
        m_w(0) = std::hypot(T(1), m_w.tail(m_w.size() - 1).stableNorm());
        const T beta = (std::sqrt(ss) / std::sqrt(zs)) * std::sqrt(sr / zr);
        for (Eigen::Index j = 0; j < s.size(); ++j)
        {
            for (Eigen::Index i = 0; i < s.size(); ++i)
            {
                const T value = i == 0 ? m_w(j) : j == 0 ? m_w(i)
                    : (i == j ? T(1) : T(0)) + m_w(i) * m_w(j) / (T(1) + m_w(0));
                W(i, j) = beta * value;
                Winv(i, j) = ((i == 0) != (j == 0) ? -value : value) / beta;
            }
        }
        lambda.noalias() = W * z;
        return W.allFinite() && Winv.allFinite() && constrained::interior<T>(lambda);
    }
};


} // namespace test
} // namespace piqp

#endif
