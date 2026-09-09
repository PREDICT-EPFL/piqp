#ifndef PIQP_CONSTRAINED_CONE_TRANSFORM_HPP
#define PIQP_CONSTRAINED_CONE_TRANSFORM_HPP

#include "piqp/fwd.hpp"
#include "piqp/constrained/cone.hpp"

namespace piqp
{
namespace constrained
{

template<typename T>
class ConeTransform
{
public:
    Vec<T> lambda;

    explicit ConeTransform(Eigen::Index dimension)
        : lambda(dimension), m_s(dimension), m_z(dimension), m_u(dimension - 1) {}

    bool compute(CVecRef<T> s, CVecRef<T> z, T delta)
    {
        if (!interior<T>(s) || !interior<T>(z) || !(delta >= T(0)) || !std::isfinite(delta)) return false;
        const T ss = s.cwiseAbs().maxCoeff(), zs = z.cwiseAbs().maxCoeff();
        m_s = s / ss; m_z = z / zs;
        const T sn = m_s.tail(s.size() - 1).stableNorm();
        const T zn = m_z.tail(z.size() - 1).stableNorm();
        const T sr = std::sqrt((m_s(0) - sn) * (m_s(0) + sn));
        const T zr = std::sqrt((m_z(0) - zn) * (m_z(0) + zn));
        if (!(sr > T(0)) || !(zr > T(0))) return false;
        m_s /= sr; m_z /= zr;
        const T st = sn / sr, zt = zn / zr;
        // Avoid cancellation when normalized tails are large and opposite.
        T product = m_z(0) / (m_s(0) + st) + st / (m_z(0) + zt);
        if (st > T(0) && zt > T(0))
        {
            T angle = T(0);
            for (Eigen::Index i = 1; i < s.size(); ++i)
            {
                const T sum = m_s(i) / st + m_z(i) / zt;
                angle += sum * sum;
            }
            product += T(.5) * st * zt * angle;
        }
        const T denominator = std::sqrt(T(2) * (T(1) + product));
        m_u = (m_s.tail(m_u.size()) - m_z.tail(m_u.size())) / denominator;
        const T r = m_u.stableNorm();
        const T w0 = std::hypot(T(1), r);
        m_r = r; m_w0 = w0;
        if (r > T(0))
            for (Eigen::Index i = 0; i < m_u.size(); ++i) m_u(i) /= r;
        const T beta = (std::sqrt(ss) / std::sqrt(zs)) * std::sqrt(sr / zr);
        m_base = beta;
        const T plus = beta * (w0 + r);
        const T minus = beta / (w0 + r);
        const T sqrt_delta = std::sqrt(delta);
        m_inverse_base = T(1) / std::hypot(beta, sqrt_delta);
        const T hp = std::hypot(plus, sqrt_delta), hm = std::hypot(minus, sqrt_delta);
        m_inverse_plus = T(1) / hp; m_inverse_minus = T(1) / hm;
        m_inverse_a = T(.5) / hp + T(.5) / hm;
        // Rationalize the difference of inverse eigenvalues when r is small.
        m_inverse_b = -(beta / hp) * (beta / hm) * (w0 * r / (T(.5) * hp + T(.5) * hm));
        m_scaled_base = beta * m_inverse_base;
        m_scaled_plus = plus / hp; m_scaled_minus = minus / hm;
        m_scaled_a = T(.5) * m_scaled_plus + T(.5) * m_scaled_minus;
        m_scaled_b = m_scaled_a > T(0)
            ? (sqrt_delta / hp) * (beta / hp) * (sqrt_delta / hm) * (beta / hm) * (w0 * r / m_scaled_a) : T(0);
        if (!std::isfinite(m_scaled_b)) m_scaled_b = T(.5) * m_scaled_plus - T(.5) * m_scaled_minus;
        if (!(minus > T(0)) || !std::isfinite(plus) || !std::isfinite(T(1) / minus)) return false;
        for (T coefficient : {m_base, m_w0, m_r, m_inverse_base, m_inverse_a, m_inverse_b,
                              m_inverse_plus, m_inverse_minus, m_scaled_base, m_scaled_a,
                              m_scaled_b, m_scaled_plus, m_scaled_minus})
            if (!std::isfinite(coefficient)) return false;
        apply_W(z, lambda);
        return interior<T>(lambda);
    }

    void apply_W(CVecRef<T> x, VecRef<T> out) const
    {
        if (m_r > T(.5)) apply_spectral(x, out, m_base, m_base * (m_w0 + m_r), m_base / (m_w0 + m_r));
        else apply(x, out, m_base, m_base * m_w0, m_base * m_r, (m_base * m_r) * (m_r / (T(1) + m_w0)));
    }

    void apply_Winv(CVecRef<T> x, VecRef<T> out) const
    {
        if (m_r > T(.5)) apply_spectral(x, out, T(1) / m_base, (T(1) / m_base) / (m_w0 + m_r), (m_w0 + m_r) / m_base);
        else apply(x, out, T(1) / m_base, m_w0 / m_base, -m_r / m_base, (m_r / m_base) * (m_r / (T(1) + m_w0)));
    }

    void apply_W2(CVecRef<T> x, VecRef<T> out) const
    {
        apply_W(x, out);
        apply_W(out, out);
    }

    void apply_inverse_sqrt_D(CVecRef<T> x, VecRef<T> out) const
    {
        if (m_r > T(.5)) apply_spectral(x, out, m_inverse_base, m_inverse_plus, m_inverse_minus);
        else apply(x, out, m_inverse_base, m_inverse_a, m_inverse_b, m_inverse_a - m_inverse_base);
    }

    void apply_W_inverse_sqrt_D(CVecRef<T> x, VecRef<T> out) const
    {
        if (m_r > T(.5)) apply_spectral(x, out, m_scaled_base, m_scaled_plus, m_scaled_minus);
        else apply(x, out, m_scaled_base, m_scaled_a, m_scaled_b, m_scaled_a - m_scaled_base);
    }

private:
    Vec<T> m_s, m_z, m_u;
    T m_base = T(1), m_w0 = T(1), m_r = T(0);
    T m_inverse_base = T(1), m_inverse_a = T(1), m_inverse_b = T(0);
    T m_inverse_plus = T(1), m_inverse_minus = T(1);
    T m_scaled_base = T(1), m_scaled_a = T(1), m_scaled_b = T(0), m_scaled_plus = T(1), m_scaled_minus = T(1);

    void apply_spectral(CVecRef<T> x, VecRef<T> out, T base, T plus, T minus) const
    {
        const T x0 = x(0), projection = m_u.dot(x.tail(m_u.size()));
        const T positive = plus * (T(.5) * x0 + T(.5) * projection);
        const T negative = minus * (T(.5) * x0 - T(.5) * projection);
        for (Eigen::Index i = 0; i < m_u.size(); ++i)
            out(i + 1) = base * (x(i + 1) - m_u(i) * projection) + m_u(i) * (positive - negative);
        out(0) = positive + negative;
    }

    void apply(CVecRef<T> x, VecRef<T> out, T base, T a, T b, T offset) const
    {
        const T x0 = x(0), projection = m_u.dot(x.tail(m_u.size()));
        const T correction = b * x0 + offset * projection;
        for (Eigen::Index i = 0; i < m_u.size(); ++i)
            out(i + 1) = base * x(i + 1) + m_u(i) * correction;
        out(0) = a * x0 + b * projection;
    }
};

} // namespace constrained
} // namespace piqp

#endif
