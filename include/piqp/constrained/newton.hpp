#ifndef PIQP_CONSTRAINED_NEWTON_HPP
#define PIQP_CONSTRAINED_NEWTON_HPP

#include "piqp/fwd.hpp"
#include "piqp/kkt_system.tpp"

namespace piqp
{
namespace constrained
{

template<typename T, typename I, int MatrixType>
class Newton
{
public:
    using DataType = std::conditional_t<MatrixType == PIQP_DENSE, dense::Data<T>, sparse::Data<T, I>>;
    DataType data;
    T residual_norm = T(0);
    isize refinement_steps = 0;

    void set_refinement_settings(const Settings<T>& settings)
    {
        m_settings.iterative_refinement_eps_abs = settings.iterative_refinement_eps_abs;
        m_settings.iterative_refinement_eps_rel = settings.iterative_refinement_eps_rel;
        m_settings.iterative_refinement_max_iter = settings.iterative_refinement_max_iter;
    }

    bool setup(const DataType& initial_data, const Settings<T>& settings)
    {
        data = initial_data;
        m_settings = settings;
        m_x_reg.resize(data.n);
        m_z_reg.resize(data.m);
        m_ex.resize(data.n); m_ey.resize(data.p); m_ez.resize(data.m);
        m_dx.resize(data.n); m_dy.resize(data.p); m_dz.resize(data.m);
        m_backend.reset();
        return detail::init_kkt_solver_impl<T, I>(data, settings, m_backend,
                                                std::integral_constant<int, MatrixType>());
    }

    bool factor(T rho, T delta, const Vec<T>& z_reg, int update_options = KKT_UPDATE_P | KKT_UPDATE_G)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Newton::factor");
        m_x_reg.setConstant(rho);
        m_z_reg = z_reg;
        m_delta = delta;
        m_backend->update_data(data, update_options);
        return m_backend->update_scalings_and_factor(data, delta, m_x_reg, m_z_reg);
    }

    bool solve(const Vec<T>& rx, const Vec<T>& ry, const Vec<T>& rz,
               Vec<T>& dx, Vec<T>& dy, Vec<T>& dz)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Newton::solve");
        m_backend->solve(data, rx, ry, rz, dx, dy, dz);
        const T tolerance = m_settings.iterative_refinement_eps_abs
                          + m_settings.iterative_refinement_eps_rel * norm(rx, ry, rz);
        refinement_steps = 0;
        residual(rx, ry, rz, dx, dy, dz);
        while (residual_norm > tolerance && refinement_steps < m_settings.iterative_refinement_max_iter)
        {
            m_backend->solve(data, m_ex, m_ey, m_ez, m_dx, m_dy, m_dz);
            dx += m_dx; dy += m_dy; dz += m_dz;
            ++refinement_steps;
            residual(rx, ry, rz, dx, dy, dz);
        }
        return std::isfinite(residual_norm) && residual_norm <= tolerance;
    }

private:
    Settings<T> m_settings;
    std::unique_ptr<KKTSolverBase<T, I, MatrixType>> m_backend;
    T m_delta = T(0);
    Vec<T> m_x_reg, m_z_reg, m_ex, m_ey, m_ez, m_dx, m_dy, m_dz;

    static T norm(const Vec<T>& x, const Vec<T>& y, const Vec<T>& z)
    {
        if (!x.allFinite() || !y.allFinite() || !z.allFinite())
            return std::numeric_limits<T>::infinity();
        return (std::max)({x.template lpNorm<Eigen::Infinity>(),
                           y.template lpNorm<Eigen::Infinity>(),
                           z.template lpNorm<Eigen::Infinity>()});
    }

    void residual(const Vec<T>& rx, const Vec<T>& ry, const Vec<T>& rz,
                  const Vec<T>& dx, const Vec<T>& dy, const Vec<T>& dz)
    {
        m_ex = rx;
        m_ex.noalias() -= data.P_utri.template selfadjointView<Eigen::Upper>() * dx;
        m_ex.array() -= m_x_reg.array() * dx.array();
        m_ex.noalias() -= data.AT * dy;
        m_ex.noalias() -= data.GT * dz;
        m_ey = ry;
        m_ey.noalias() -= data.AT.transpose() * dx;
        m_ey.noalias() += m_delta * dy;
        m_ez = rz;
        m_ez.noalias() -= data.GT.transpose() * dx;
        m_ez.array() += m_z_reg.array() * dz.array();
        residual_norm = norm(m_ex, m_ey, m_ez);
    }
};

} // namespace constrained
} // namespace piqp

#endif
