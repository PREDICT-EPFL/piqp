#ifndef PIQP_CONSTRAINED_SOLVER_HPP
#define PIQP_CONSTRAINED_SOLVER_HPP

#include "piqp/fwd.hpp"
#include <chrono>
#include <stdexcept>
#include "piqp/constrained/constraints.hpp"
#include "piqp/constrained/cone.hpp"
#include "piqp/constrained/cone_transform.hpp"
#include "piqp/constrained/newton.hpp"
#include "piqp/results.hpp"

namespace piqp
{
namespace constrained
{

template<typename T, typename Function>
void entries(const Mat<T>& matrix, Function function)
{
    for (isize j = 0; j < matrix.cols(); ++j)
        for (isize i = 0; i < matrix.rows(); ++i) function(i, j, matrix(i, j));
}

template<typename T, typename I, typename Function>
void entries(const SparseMat<T, I>& matrix, Function function)
{
    for (I j = 0; j < matrix.outerSize(); ++j)
        for (typename SparseMat<T, I>::InnerIterator it(matrix, j); it; ++it)
            function(it.row(), it.col(), it.value());
}

template<typename T> void zero_values(Mat<T>& matrix) { matrix.setZero(); }
template<typename T, typename I> void zero_values(SparseMat<T, I>& matrix)
{
    std::fill(matrix.valuePtr(), matrix.valuePtr() + matrix.nonZeros(), T(0));
}

template<typename T, typename I>
void from_sparse(Mat<T>& out, const SparseMat<T, I>& matrix) { out = Mat<T>(matrix); }
template<typename T, typename I>
void from_sparse(SparseMat<T, I>& out, const SparseMat<T, I>& matrix) { out = matrix; }

template<typename T>
bool same_structure(const Mat<T>& a, const Mat<T>& b)
{ return a.rows() == b.rows() && a.cols() == b.cols(); }
template<typename T, typename I>
bool same_structure(const SparseMat<T, I>& a, const SparseMat<T, I>& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols() || a.nonZeros() != b.nonZeros()) return false;
    for (I j = 0; j < a.outerSize(); ++j)
    {
        typename SparseMat<T, I>::InnerIterator ia(a, j), ib(b, j);
        for (; ia && ib; ++ia, ++ib) if (ia.row() != ib.row()) return false;
        if (ia || ib) return false;
    }
    return true;
}

template<typename T>
struct Result
{
    Status status = PIQP_UNSOLVED;
    Vec<T> x, y, s_l, s_u, z_l, z_u, s_bl, s_bu, z_bl, z_bu;
    Vec<T> quadratic_slack, quadratic_dual, cone_slack, cone_dual;
    std::vector<isize> cone_offsets;
    isize iterations = 0, factor_retries = 0, refinement_steps = 0;
    T primal_residual = 0, dual_residual = 0, complementarity = 0, objective = 0;
    double setup_time = 0, update_time = 0, solve_time = 0, factor_time = 0;
};

template<typename T, typename I, int MatrixType>
class Solver
{
public:
    using Model = std::conditional_t<MatrixType == PIQP_DENSE, dense::Model<T>, sparse::Model<T, I>>;
    using Matrix = std::conditional_t<MatrixType == PIQP_DENSE, Mat<T>, SparseMat<T, I>>;
    using Data = typename Newton<T, I, MatrixType>::DataType;

    Solver() { m_settings.kkt_solver = MatrixType == PIQP_DENSE ? KKTSolver::dense_cholesky : KKTSolver::sparse_ldlt; }
    Settings<T>& settings() { return m_settings; }
    const Result<T>& result() const { return m_result; }

    void setup(const Model& model)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::setup");
        const auto start = timing_start();
        validate(model);
        m_setup_done = false;
        m_result.status = PIQP_UNSOLVED;
        m_model = std::make_unique<Model>(model);
        m_n = model.c.size(); m_p = model.b.size();
        m_blocks.clear();
        make_blocks();
        m_E = Vec<T>::Ones(m_n); m_eq_scale = Vec<T>::Ones(m_p); m_gamma = T(1);
        m_P = model.P; m_A = model.A; m_c = model.c; m_b = model.b;
        equilibrate();
        make_workspace();
        m_result.setup_time = seconds(start);
        m_backend_kind = m_settings.kkt_solver;
        m_setup_done = true;
    }

    void update_quadratic(isize index, T upper)
    {
        const auto start = timing_start();
        if (!m_setup_done || index < 0 || index >= static_cast<isize>(m_model->quadratic_constraints.size()) || !std::isfinite(upper))
            throw std::invalid_argument("invalid quadratic update");
        m_model->quadratic_constraints[usize(index)].upper = upper;
        auto& block = m_blocks[usize(m_quadratic_start + index)];
        block.f(0) = -block.scale * upper;
        m_result.status = PIQP_UNSOLVED;
        m_result.update_time = seconds(start);
    }

    void update(const Model& model)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::update");
        const auto start = timing_start();
        require(m_setup_done, "setup required before update");
        validate(model);
        require(same_structure(model.P, m_model->P) && same_structure(model.A, m_model->A) && same_structure(model.G, m_model->G), "matrix structure changed; call setup");
        require(model.quadratic_constraints.size() == m_model->quadratic_constraints.size() && model.cone_constraints.size() == m_model->cone_constraints.size(), "constraint count changed; call setup");
        for (const auto pair : {std::make_pair(&model.h_l, &m_model->h_l), std::make_pair(&model.h_u, &m_model->h_u), std::make_pair(&model.x_l, &m_model->x_l), std::make_pair(&model.x_u, &m_model->x_u)})
            for (isize i = 0; i < pair.first->size(); ++i)
                require(std::isfinite((*pair.first)(i)) == std::isfinite((*pair.second)(i)), "finite bound pattern changed; call setup");
        for (isize i = 0; i < static_cast<isize>(model.quadratic_constraints.size()); ++i)
        {
            const auto& a = model.quadratic_constraints[usize(i)]; const auto& b = m_model->quadratic_constraints[usize(i)];
            require(a.indices == b.indices && same_structure(a.Q, b.Q), "quadratic structure changed; call setup");
        }
        for (isize i = 0; i < static_cast<isize>(model.cone_constraints.size()); ++i)
        {
            const auto& a = model.cone_constraints[usize(i)]; const auto& b = m_model->cone_constraints[usize(i)];
            require(a.indices == b.indices && a.type == b.type && same_structure(a.F, b.F), "cone structure changed; call setup");
        }
        for (const auto pair : {std::make_pair(&m_model->P, &model.P), std::make_pair(&m_model->A, &model.A), std::make_pair(&m_model->G, &model.G)})
            entries(*pair.second, [&](isize i, isize j, T v) { pair.first->coeffRef(i, j) = v; });
        m_model->c = model.c; m_model->b = model.b;
        m_model->h_l = model.h_l; m_model->h_u = model.h_u;
        m_model->x_l = model.x_l; m_model->x_u = model.x_u;
        for (isize i = 0; i < static_cast<isize>(model.quadratic_constraints.size()); ++i)
        {
            auto& destination = m_model->quadratic_constraints[usize(i)]; const auto& source = model.quadratic_constraints[usize(i)];
            entries(source.Q, [&](isize r, isize c, T v) { destination.Q.coeffRef(r, c) = v; });
            destination.q = source.q; destination.upper = source.upper;
        }
        for (isize i = 0; i < static_cast<isize>(model.cone_constraints.size()); ++i)
        {
            auto& destination = m_model->cone_constraints[usize(i)]; const auto& source = model.cone_constraints[usize(i)];
            entries(source.F, [&](isize r, isize c, T v) { destination.F.coeffRef(r, c) = v; });
            destination.f = source.f;
        }
        entries(model.P, [&](isize i, isize j, T v) { m_P.coeffRef(i, j) = m_gamma * m_E(i) * m_E(j) * v; });
        entries(model.A, [&](isize i, isize j, T v) { m_A.coeffRef(i, j) = m_eq_scale(i) * m_E(j) * v; });
        m_c.array() = m_gamma * m_E.array() * model.c.array(); m_b.array() = m_eq_scale.array() * model.b.array();
        for (auto& block : m_blocks)
        {
            if (block.kind < 2)
            {
                const T sign = block.kind == 0 ? T(-1) : T(1);
                for (isize j = 0; j < block.F.cols(); ++j) block.F(0, j) = sign * block.scale * m_E(block.indices[usize(j)]) * model.G.coeff(block.input, block.indices[usize(j)]);
                block.f(0) = -sign * block.scale * (block.kind == 0 ? model.h_l(block.input) : model.h_u(block.input));
            }
            else if (block.kind < 4)
                block.f(0) = block.scale * (block.kind == 2 ? model.x_l(block.input) : -model.x_u(block.input));
            else if (block.quadratic)
            {
                const auto& source = model.quadratic_constraints[usize(block.input)];
                entries(source.Q, [&](isize i, isize j, T v) { block.Q.coeffRef(i, j) = block.scale * m_E(block.indices[usize(i)]) * m_E(block.indices[usize(j)]) * v; });
                for (isize j = 0; j < block.F.cols(); ++j) block.F(0, j) = block.scale * m_E(block.indices[usize(j)]) * source.q(j);
                block.f(0) = -block.scale * source.upper;
            }
            else
            {
                const auto& source = model.cone_constraints[usize(block.input)];
                for (isize j = 0; j < block.F.cols(); ++j)
                    for (isize i = 0; i < block.F.rows(); ++i)
                        block.F(i, j) = -block.scale * m_E(block.indices[usize(j)]) * source.F.coeff(i, block.columns[usize(j)]);
                block.f = -block.scale * source.f;
                if (source.type == ConeType::rotated_second_order) { rotate_matrix(block.F); rotate(block.f); }
            }
        }
        entries(m_A, [&](isize i, isize j, T v) { m_newton.data.AT.coeffRef(j, i) = v; });
        m_update_options = KKT_UPDATE_P | KKT_UPDATE_A | KKT_UPDATE_G;
        m_result.status = PIQP_UNSOLVED; m_result.update_time = seconds(start);
    }

    void update_cone(isize index, CVecRef<T> offset)
    {
        const auto start = timing_start();
        if (!m_setup_done || index < 0 || index >= static_cast<isize>(m_model->cone_constraints.size()) ||
            offset.size() != m_model->cone_constraints[usize(index)].f.size() || !offset.allFinite())
            throw std::invalid_argument("invalid cone update");
        auto& source = m_model->cone_constraints[usize(index)];
        source.f = offset;
        auto& block = m_blocks[usize(m_cone_start + index)];
        block.f = -block.scale * offset;
        if (source.type == ConeType::rotated_second_order) rotate(block.f);
        m_result.status = PIQP_UNSOLVED;
        m_result.update_time = seconds(start);
    }

    Status solve()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::solve");
        if (!m_setup_done) return m_result.status = PIQP_UNSOLVED;
        if (!m_settings.verify_settings() || m_settings.kkt_solver != m_backend_kind) return m_result.status = PIQP_INVALID_SETTINGS;
        m_newton.set_refinement_settings(m_settings);
        const auto start = timing_start();
        m_result.iterations = m_result.factor_retries = m_result.refinement_steps = 0;
        m_result.factor_time = 0;
        m_rho = m_settings.rho_init; m_delta = m_settings.delta_init;
        initialize();
        m_xi = m_x; m_eta = m_y; m_zeta = m_z;
        m_result.status = PIQP_MAX_ITER_REACHED;
        T previous_primal = std::numeric_limits<T>::infinity(), previous_dual = previous_primal;
        for (isize iteration = 0; iteration <= m_settings.max_iter; ++iteration)
        {
            m_result.iterations = iteration;
            residuals();
            report();
            if (m_settings.verbose) piqp_print("%3ld  primal %.3e  dual %.3e  complementarity %.3e\n", long(iteration), double(m_result.primal_residual), double(m_result.dual_residual), double(m_result.complementarity));
            if (converged()) { m_result.status = PIQP_SOLVED; break; }
            if (iteration == m_settings.max_iter) break;
            const T mu = complementarity(m_s, m_z);
            if (!std::isfinite(mu) || !m_x.allFinite() || !all_interior(m_s, m_z))
            { m_result.status = PIQP_NUMERICS; break; }
            if (m_result.dual_residual < T(.95) * previous_dual) m_xi = m_x;
            if (m_result.primal_residual < T(.95) * previous_primal) { m_eta = m_y; m_zeta = m_z; }
            previous_primal = m_result.primal_residual; previous_dual = m_result.dual_residual;
            m_rho = (std::max)(m_settings.reg_finetune_lower_limit, (std::min)(m_rho, T(.1) * mu));
            m_delta = (std::max)(m_settings.reg_finetune_lower_limit, (std::min)(m_delta, T(.1) * mu));
            bool factored = false;
            for (isize retry = 0; retry < m_settings.max_factor_retires; ++retry)
            {
                const auto factor_start = timing_start();
                factored = assemble() && m_newton.factor(m_rho, m_delta, m_z_reg);
                m_result.factor_time += seconds(factor_start);
                if (factored && direction(T(0), false)) break;
                factored = false;
                m_rho *= T(10); m_delta *= T(10); ++m_result.factor_retries;
            }
            if (!factored)
            {
                if (m_settings.verbose) piqp_print("Newton solve failed, residual %.3e\n", double(m_newton.residual_norm));
                m_result.status = PIQP_NUMERICS; break;
            }
            m_ax = m_dx; m_as = m_ds; m_az = m_dz;
            T alpha = step_length(m_s, m_z, m_ds, m_dz);
            m_trial_s = m_s + alpha * m_ds; m_trial_z = m_z + alpha * m_dz;
            T ratio = mu > T(0) ? complementarity(m_trial_s, m_trial_z) / mu : T(0);
            ratio = (std::max)(T(0), (std::min)(T(1), ratio));
            const T target = ratio * ratio * ratio * mu;
            if (!direction(target, true) || !take_step(target))
            {
                if (!direction(T(.5) * mu, false) || !take_step(T(.5) * mu))
                {
                    if (m_settings.verbose) piqp_print("Centered step failed, Newton residual %.3e\n", double(m_newton.residual_norm));
                    m_result.status = PIQP_NUMERICS; break;
                }
            }
        }
        m_result.solve_time = seconds(start);
        return m_result.status;
    }

private:
    using Clock = std::chrono::steady_clock;
    struct Block
    {
        std::vector<isize> indices, columns;
        Matrix Q;
        Mat<T> F, J;
        Vec<T> f, local, gradient, t, rc, work, work2;
        ConeTransform<T> scaling;
        isize offset = 0, kind = 0, input = 0;
        T scale = T(1);
        bool cone = false, quadratic = false;
        Block(isize dimension, isize support)
            : Q(0, 0), F(dimension, support), J(dimension, support),
              f(dimension), local(support), gradient(support), t(dimension),
              rc(dimension), work(dimension), work2(dimension), scaling(dimension) {}
    };
    Settings<T> m_settings;
    Result<T> m_result;
    std::unique_ptr<Model> m_model;
    std::vector<Block> m_blocks;
    Matrix m_P, m_A;
    Vec<T> m_c, m_b, m_E, m_eq_scale;
    T m_gamma = 1, m_rho = 0, m_delta = 0;
    isize m_n = 0, m_p = 0, m_m = 0, m_quadratic_start = 0, m_cone_start = 0;
    int m_update_options = KKT_UPDATE_P | KKT_UPDATE_G;
    KKTSolver m_backend_kind = KKTSolver::dense_cholesky;
    bool m_setup_done = false;
    Newton<T, I, MatrixType> m_newton;
    Vec<T> m_x, m_y, m_s, m_z, m_rd, m_re, m_rg, m_rx, m_ry, m_rz, m_z_reg;
    Vec<T> m_dx, m_dy, m_ds, m_dz, m_ax, m_as, m_az, m_xi, m_eta, m_zeta;
    Vec<T> m_trial_x, m_trial_y, m_trial_s, m_trial_z, m_original_rd, m_affine_dual, m_affine_value;

    Clock::time_point timing_start() const
    { return m_settings.compute_timings ? Clock::now() : Clock::time_point{}; }
    double seconds(Clock::time_point start) const
    { return m_settings.compute_timings ? std::chrono::duration<double>(Clock::now() - start).count() : 0; }
    static T norm(CVecRef<T> vector) { return vector.template lpNorm<Eigen::Infinity>(); }
    static void require(bool condition, const char* message)
    { if (!condition) throw std::invalid_argument(message); }
    static void rotate(VecRef<T> v)
    {
        const T a = v(0), b = v(1);
        v(0) = a + b; v(1) = a - b;
        v.tail(v.size() - 2) *= std::sqrt(T(2));
    }
    static void rotate_matrix(Mat<T>& matrix)
    {
        for (isize j = 0; j < matrix.cols(); ++j)
        {
            const T a = matrix(0, j), b = matrix(1, j);
            matrix(0, j) = a + b; matrix(1, j) = a - b;
        }
        matrix.bottomRows(matrix.rows() - 2) *= std::sqrt(T(2));
    }
    static void validate_indices(const std::vector<isize>& indices, isize size, isize n)
    {
        require(indices.empty() ? size == n : static_cast<isize>(indices.size()) == size, "constraint support dimension mismatch");
        for (isize j = 0; j < static_cast<isize>(indices.size()); ++j)
        {
            const isize i = indices[usize(j)];
            require(i >= 0 && i < n && std::find(indices.begin(), indices.begin() + j, i) == indices.begin() + j, "constraint indices must be valid and unique");
        }
    }
    static void validate(const Model& model)
    {
        const isize n = model.c.size();
        require(n > 0 && model.P.rows() == n && model.P.cols() == n, "invalid objective dimensions");
        require(model.A.cols() == n && model.A.rows() == model.b.size(), "invalid equality dimensions");
        require(model.G.cols() == n && model.G.rows() == model.h_l.size() && model.G.rows() == model.h_u.size(), "invalid inequality dimensions");
        require(model.x_l.size() == n && model.x_u.size() == n, "invalid bound dimensions");
        require(model.c.allFinite() && model.b.allFinite(), "nonfinite objective or equality data");
        for (const Matrix* matrix : {&model.P, &model.A, &model.G})
            entries(*matrix, [](isize, isize, T v) { require(std::isfinite(v), "nonfinite matrix data"); });
        for (const auto& block : model.quadratic_constraints)
        {
            require(block.Q.rows() == block.q.size() && block.Q.cols() == block.q.size() && block.q.allFinite() && std::isfinite(block.upper), "invalid quadratic data");
            validate_indices(block.indices, block.q.size(), n);
            entries(block.Q, [](isize, isize, T v) { require(std::isfinite(v), "nonfinite quadratic matrix"); });
        }
        for (const auto& block : model.cone_constraints)
        {
            require(block.F.rows() == block.f.size() && block.f.size() >= 2 && block.f.allFinite(), "invalid cone dimensions");
            require(block.type == ConeType::second_order || (block.type == ConeType::rotated_second_order && block.f.size() >= 3), "invalid cone type");
            validate_indices(block.indices, block.F.cols(), n);
            entries(block.F, [](isize, isize, T v) { require(std::isfinite(v), "nonfinite cone matrix"); });
        }
        for (isize i = 0; i < model.h_l.size(); ++i)
            require(model.h_l(i) < std::numeric_limits<T>::infinity() && model.h_u(i) > -std::numeric_limits<T>::infinity() && model.h_l(i) <= model.h_u(i), "invalid affine bounds");
        for (isize i = 0; i < n; ++i)
            require(model.x_l(i) < std::numeric_limits<T>::infinity() && model.x_u(i) > -std::numeric_limits<T>::infinity() && model.x_l(i) <= model.x_u(i), "invalid variable bounds");
    }

    void make_blocks()
    {
        const auto& model = *m_model;
        std::vector<Eigen::Triplet<T, I>> entries_G;
        entries(model.G, [&](isize i, isize j, T v) { entries_G.emplace_back(I(j), I(i), v); });
        SparseMat<T, I> GT(m_n, model.G.rows());
        GT.setFromTriplets(entries_G.begin(), entries_G.end());
        for (isize i = 0; i < model.G.rows(); ++i)
            for (isize side = 0; side < 2; ++side)
            {
                const T bound = side ? model.h_u(i) : model.h_l(i);
                if (!std::isfinite(bound)) continue;
                const T sign = side ? T(1) : T(-1);
                std::vector<isize> support;
                for (typename SparseMat<T, I>::InnerIterator it(GT, i); it; ++it) support.push_back(it.row());
                m_blocks.emplace_back(1, isize(support.size()));
                auto& block = m_blocks.back(); block.indices = std::move(support);
                block.kind = side; block.input = i; block.f(0) = -sign * bound;
                isize j = 0;
                for (typename SparseMat<T, I>::InnerIterator it(GT, i); it; ++it) block.F(0, j++) = sign * it.value();
            }
        for (isize i = 0; i < m_n; ++i)
            for (isize side = 0; side < 2; ++side)
            {
                const T bound = side ? model.x_u(i) : model.x_l(i);
                if (!std::isfinite(bound)) continue;
                const T sign = side ? T(1) : T(-1);
                m_blocks.emplace_back(1, 1);
                auto& block = m_blocks.back(); block.indices = {i};
                block.kind = side + 2; block.input = i; block.f(0) = -sign * bound; block.F(0, 0) = sign;
            }
        m_quadratic_start = isize(m_blocks.size());
        for (isize i = 0; i < static_cast<isize>(model.quadratic_constraints.size()); ++i)
        {
            const auto& source = model.quadratic_constraints[usize(i)];
            m_blocks.emplace_back(1, source.q.size());
            auto& block = m_blocks.back(); block.indices = source.indices;
            block.Q = source.Q; block.F.row(0) = source.q.transpose(); block.f(0) = -source.upper;
            block.quadratic = true; block.kind = 4; block.input = i;
        }
        m_cone_start = isize(m_blocks.size());
        for (isize i = 0; i < static_cast<isize>(model.cone_constraints.size()); ++i)
        {
            const auto& source = model.cone_constraints[usize(i)];
            std::vector<isize> columns(usize(source.F.cols()), -1), support;
            entries(source.F, [&](isize, isize j, T) { columns[usize(j)] = 0; });
            for (isize j = 0; j < source.F.cols(); ++j)
                if (columns[usize(j)] == 0) { columns[usize(j)] = isize(support.size()); support.push_back(source.indices.empty() ? j : source.indices[usize(j)]); }
            m_blocks.emplace_back(source.f.size(), isize(support.size()));
            auto& block = m_blocks.back(); block.indices = std::move(support);
            for (isize j = 0; j < source.F.cols(); ++j) if (columns[usize(j)] >= 0) block.columns.push_back(j);
            block.F.setZero();
            entries(source.F, [&](isize r, isize j, T v) { block.F(r, columns[usize(j)]) = -v; });
            block.f = -source.f;
            block.cone = true; block.kind = 5; block.input = i;
            if (source.type == ConeType::rotated_second_order) { rotate_matrix(block.F); rotate(block.f); }
        }
        m_m = 0;
        for (auto& block : m_blocks)
        {
            if (block.indices.empty() && block.F.cols() == m_n)
                for (isize j = 0; j < m_n; ++j) block.indices.push_back(j);
            block.offset = m_m; m_m += block.f.size();
        }
    }

    void equilibrate()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::equilibrate");
        Vec<T> norms(m_n), step(m_n);
        for (isize iteration = 0; iteration < m_settings.preconditioner_iter; ++iteration)
        {
            norms.setZero();
            entries(m_P, [&](isize i, isize j, T v) {
                if (i <= j) { norms(i) = (std::max)(norms(i), std::abs(v)); norms(j) = (std::max)(norms(j), std::abs(v)); }
            });
            entries(m_A, [&](isize, isize j, T v) { norms(j) = (std::max)(norms(j), std::abs(v)); });
            for (const auto& block : m_blocks)
            {
                for (isize j = 0; j < block.F.cols(); ++j)
                    norms(block.indices[usize(j)]) = (std::max)(norms(block.indices[usize(j)]), block.F.col(j).cwiseAbs().maxCoeff());
                if (block.quadratic) entries(block.Q, [&](isize i, isize j, T v) {
                    if (i <= j) { const T a = std::sqrt(std::abs(v));
                        norms(block.indices[usize(i)]) = (std::max)(norms(block.indices[usize(i)]), a);
                        norms(block.indices[usize(j)]) = (std::max)(norms(block.indices[usize(j)]), a); }
                });
            }
            for (isize j = 0; j < m_n; ++j) step(j) = norms(j) > T(1e-12) ? T(1) / std::sqrt(norms(j)) : T(1);
            m_E.array() *= step.array();
            entries(m_P, [&](isize i, isize j, T v) { m_P.coeffRef(i, j) = v * step(i) * step(j); });
            entries(m_A, [&](isize i, isize j, T v) { m_A.coeffRef(i, j) = v * step(j); });
            m_c.array() *= step.array();
            for (auto& block : m_blocks)
            {
                for (isize j = 0; j < block.F.cols(); ++j) block.F.col(j) *= step(block.indices[usize(j)]);
                if (block.quadratic) entries(block.Q, [&](isize i, isize j, T v) {
                    block.Q.coeffRef(i, j) = v * step(block.indices[usize(i)]) * step(block.indices[usize(j)]);
                });
                T magnitude = block.F.size() ? block.F.cwiseAbs().maxCoeff() : T(0);
                if (block.quadratic) entries(block.Q, [&](isize, isize, T v) { magnitude = (std::max)(magnitude, std::abs(v)); });
                const T scale = magnitude > T(1e-12) ? T(1) / std::sqrt(magnitude) : T(1);
                block.scale *= scale; block.F *= scale; block.f *= scale; block.Q *= scale;
            }
            Vec<T> rows = Vec<T>::Zero(m_p);
            entries(m_A, [&](isize i, isize, T v) { rows(i) = (std::max)(rows(i), std::abs(v)); });
            for (isize i = 0; i < m_p; ++i) rows(i) = rows(i) > T(1e-12) ? T(1) / std::sqrt(rows(i)) : T(1);
            entries(m_A, [&](isize i, isize j, T v) { m_A.coeffRef(i, j) = v * rows(i); });
            m_b.array() *= rows.array(); m_eq_scale.array() *= rows.array();
        }
        if (m_settings.preconditioner_scale_cost)
        {
            T magnitude = (std::max)(T(1), norm(m_c));
            entries(m_P, [&](isize, isize, T v) { magnitude = (std::max)(magnitude, std::abs(v)); });
            m_gamma = T(1) / magnitude; m_P *= m_gamma; m_c *= m_gamma;
        }
    }

    void make_workspace()
    {
        std::vector<Eigen::Triplet<T, I>> hessian, jacobian;
        entries(m_P, [&](isize i, isize j, T) { if (i <= j) hessian.emplace_back(I(i), I(j), T(1)); });
        for (isize j = 0; j < m_n; ++j) hessian.emplace_back(I(j), I(j), T(1));
        for (const auto& block : m_blocks)
        {
            if (block.quadratic) entries(block.Q, [&](isize i, isize j, T) {
                if (i <= j) hessian.emplace_back(I((std::min)(block.indices[usize(i)], block.indices[usize(j)])), I((std::max)(block.indices[usize(i)], block.indices[usize(j)])), T(1));
            });
            for (isize r = 0; r < block.f.size(); ++r)
                for (isize index : block.indices) jacobian.emplace_back(I(index), I(block.offset + r), T(1));
        }
        SparseMat<T, I> H(m_n, m_n), JT(m_n, m_m);
        H.setFromTriplets(hessian.begin(), hessian.end()); JT.setFromTriplets(jacobian.begin(), jacobian.end());
        Data data; data.resize(m_n, m_p, m_m);
        from_sparse(data.P_utri, H); from_sparse(data.GT, JT);
        data.AT = m_A.transpose(); data.c = m_c; data.b = m_b;
        data.n_h_l = data.n_h_u = data.n_x_l = data.n_x_u = 0;
        require(m_newton.setup(data, m_settings), "unsupported constrained backend");
        for (Vec<T>* vector : {&m_x, &m_rd, &m_rx, &m_dx, &m_ax, &m_xi, &m_trial_x, &m_original_rd}) vector->resize(m_n);
        for (Vec<T>* vector : {&m_y, &m_re, &m_ry, &m_dy, &m_eta, &m_trial_y}) vector->resize(m_p);
        for (Vec<T>* vector : {&m_s, &m_z, &m_rg, &m_rz, &m_z_reg, &m_ds, &m_dz, &m_as, &m_az, &m_zeta, &m_trial_s, &m_trial_z}) vector->resize(m_m);
        m_result.x.resize(m_n); m_result.y.resize(m_p);
        m_affine_dual.resize(m_model->G.rows());
        m_affine_value.resize(m_model->G.rows());
        for (Vec<T>* v : {&m_result.s_l, &m_result.s_u, &m_result.z_l, &m_result.z_u}) v->setZero(m_model->G.rows());
        for (Vec<T>* v : {&m_result.s_bl, &m_result.s_bu, &m_result.z_bl, &m_result.z_bu}) v->setZero(m_n);
        m_result.quadratic_slack.resize(isize(m_model->quadratic_constraints.size()));
        m_result.quadratic_dual.resize(isize(m_model->quadratic_constraints.size()));
        m_result.cone_offsets.clear(); m_result.cone_offsets.push_back(0);
        for (const auto& c : m_model->cone_constraints) m_result.cone_offsets.push_back(m_result.cone_offsets.back() + c.f.size());
        m_result.cone_slack.resize(m_result.cone_offsets.back()); m_result.cone_dual.resize(m_result.cone_offsets.back());
    }

    void initialize()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::initialize");
        m_x.setZero(); m_y.setZero(); m_s.setOnes(); m_z.setOnes();
        residuals();
        zero_values(m_newton.data.P_utri);
        entries(m_P, [&](isize i, isize j, T v) { if (i <= j) m_newton.data.P_utri.coeffRef(i, j) = v; });
        zero_values(m_newton.data.GT);
        m_rx = -m_c; m_ry = m_b; m_rz.setZero(); m_z_reg.setConstant(T(1) + m_delta);
        for (auto& block : m_blocks)
        {
            if (block.quadratic) continue;
            for (isize j = 0; j < block.F.cols(); ++j)
                for (isize i = 0; i < block.f.size(); ++i)
                    m_newton.data.GT.coeffRef(block.indices[usize(j)], block.offset + i) = block.F(i, j);
            m_rz.segment(block.offset, block.f.size()) = -block.f;
        }
        const bool initialized = m_newton.factor(m_rho, m_delta, m_z_reg, m_update_options) && m_newton.solve(m_rx, m_ry, m_rz, m_dx, m_dy, m_dz);
        m_update_options = KKT_UPDATE_P | KKT_UPDATE_G;
        if (initialized)
        { m_x = m_dx; m_y = m_dy; m_z = m_dz; m_s = -m_z; }
        else { m_x.setZero(); m_y.setZero(); m_s.setOnes(); m_z.setOnes(); }
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.cone)
            {
                repair<T>(m_s.segment(o, d)); repair<T>(m_z.segment(o, d));
            }
            else
            {
                if (block.quadratic)
                {
                    for (isize j = 0; j < block.local.size(); ++j) block.local(j) = m_x(block.indices[usize(j)]);
                    block.gradient.noalias() = block.Q.template selfadjointView<Eigen::Upper>() * block.local;
                    const T value = T(.5) * block.local.dot(block.gradient) + block.F.row(0).dot(block.local) + block.f(0);
                    m_s(o) = (std::max)(T(1), -value); m_z(o) = T(1) / m_s(o);
                }
                else { m_s(o) = (std::max)(T(1), m_s(o)); m_z(o) = (std::max)(T(1), m_z(o)); }
            }
        }
    }

    void residuals()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::residuals");
        m_rd.noalias() = m_P.template selfadjointView<Eigen::Upper>() * m_x;
        m_rd += m_c; m_rd.noalias() += m_A.transpose() * m_y;
        m_re.noalias() = m_A * m_x; m_re -= m_b;
        for (auto& block : m_blocks)
        {
            for (isize j = 0; j < block.local.size(); ++j) block.local(j) = m_x(block.indices[usize(j)]);
            auto rg = m_rg.segment(block.offset, block.f.size());
            rg.noalias() = block.F * block.local; rg += block.f; rg += m_s.segment(block.offset, block.f.size());
            block.J = block.F;
            if (block.quadratic)
            {
                block.gradient.noalias() = block.Q.template selfadjointView<Eigen::Upper>() * block.local;
                rg(0) += T(.5) * block.local.dot(block.gradient);
                block.J.row(0) += block.gradient.transpose();
            }
            block.gradient.noalias() = block.J.transpose() * m_z.segment(block.offset, block.f.size());
            for (isize j = 0; j < block.local.size(); ++j) m_rd(block.indices[usize(j)]) += block.gradient(j);
        }
    }

    bool assemble()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::assemble");
        zero_values(m_newton.data.P_utri);
        entries(m_P, [&](isize i, isize j, T v) { if (i <= j) m_newton.data.P_utri.coeffRef(i, j) = v; });
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.quadratic) entries(block.Q, [&](isize i, isize j, T v) {
                if (i <= j) m_newton.data.P_utri.coeffRef((std::min)(block.indices[usize(i)], block.indices[usize(j)]), (std::max)(block.indices[usize(i)], block.indices[usize(j)])) += m_z(o) * v;
            });
            if (block.cone)
            {
                if (!block.scaling.compute(m_s.segment(o, d), m_z.segment(o, d), m_delta)) return false;
                block.J = block.F;
                for (isize j = 0; j < block.J.cols(); ++j) block.scaling.apply_inverse_sqrt_D(block.J.col(j), block.J.col(j));
                m_z_reg.segment(o, d).setOnes();
            }
            else m_z_reg(o) = m_s(o) / m_z(o) + m_delta;
            for (isize j = 0; j < block.J.cols(); ++j)
                for (isize i = 0; i < d; ++i) m_newton.data.GT.coeffRef(block.indices[usize(j)], o + i) = block.J(i, j);
        }
        return true;
    }

    bool direction(T target, bool corrector)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::direction");
        m_rx = -m_rd - m_rho * (m_x - m_xi);
        m_ry = -m_re + m_delta * (m_y - m_eta);
        m_rz = -m_rg + m_delta * (m_z - m_zeta);
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.cone)
            {
                block.rc.setZero(); block.rc(0) = target;
                if (corrector)
                {
                    block.scaling.apply_Winv(m_as.segment(o, d), block.work);
                    block.scaling.apply_W(m_az.segment(o, d), block.work2);
                    jordan<T>(block.work, block.work2, block.t); block.rc -= block.t;
                }
                jordan_solve<T>(block.scaling.lambda, block.rc, block.work);
                block.work -= block.scaling.lambda;
                block.scaling.apply_inverse_sqrt_D(m_rz.segment(o, d), m_rz.segment(o, d));
                block.scaling.apply_W_inverse_sqrt_D(block.work, block.t);
                m_rz.segment(o, d) -= block.t;
            }
            else
            {
                block.t(0) = -m_s(o) + target / m_z(o);
                if (corrector) block.t(0) -= m_as(o) * m_az(o) / m_z(o);
                m_rz(o) -= block.t(0);
                if (corrector && block.quadratic)
                {
                    for (isize j = 0; j < block.local.size(); ++j) block.local(j) = m_ax(block.indices[usize(j)]);
                    block.gradient.noalias() = block.Q.template selfadjointView<Eigen::Upper>() * block.local;
                    m_rz(o) -= T(.5) * block.local.dot(block.gradient);
                    for (isize j = 0; j < block.local.size(); ++j) m_rx(block.indices[usize(j)]) -= m_az(o) * block.gradient(j);
                }
            }
        }
        const bool solved = m_newton.solve(m_rx, m_ry, m_rz, m_dx, m_dy, m_dz);
        m_result.refinement_steps += m_newton.refinement_steps;
        if (!solved) return false;
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.cone)
            {
                block.scaling.apply_inverse_sqrt_D(m_dz.segment(o, d), m_dz.segment(o, d));
                for (isize j = 0; j < block.local.size(); ++j) block.local(j) = m_dx(block.indices[usize(j)]);
                block.work.noalias() = block.F * block.local;
                m_ds.segment(o, d) = -m_rg.segment(o, d) - block.work
                    + m_delta * (m_z.segment(o, d) - m_zeta.segment(o, d) + m_dz.segment(o, d));
            }
            else m_ds(o) = block.t(0) - (m_s(o) / m_z(o)) * m_dz(o);
        }
        return m_dx.allFinite() && m_ds.allFinite() && m_dz.allFinite();
    }

    T complementarity(const Vec<T>& s, const Vec<T>& z) const
    { return m_blocks.empty() ? T(0) : s.dot(z) / T(m_blocks.size()); }
    bool all_interior(const Vec<T>& s, const Vec<T>& z) const
    {
        for (const auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.cone) { if (!interior<T>(s.segment(o, d)) || !interior<T>(z.segment(o, d))) return false; }
            else if (!(s(o) > 0 && z(o) > 0)) return false;
        }
        return true;
    }
    T step_length(const Vec<T>& s, const Vec<T>& z, const Vec<T>& ds, const Vec<T>& dz) const
    {
        T alpha = T(1);
        for (const auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            if (block.cone) alpha = (std::min)({alpha, max_step<T>(s.segment(o, d), ds.segment(o, d)), max_step<T>(z.segment(o, d), dz.segment(o, d))});
            else
            {
                if (ds(o) < T(0)) alpha = (std::min)(alpha, -s(o) / ds(o));
                if (dz(o) < T(0)) alpha = (std::min)(alpha, -z(o) / dz(o));
            }
        }
        return (std::min)(T(1), m_settings.tau * alpha);
    }

    T merit(const Vec<T>& x, const Vec<T>& y, const Vec<T>& s, const Vec<T>& z, T target)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::merit");
        m_original_rd.noalias() = m_P.template selfadjointView<Eigen::Upper>() * x;
        m_original_rd += m_c; m_original_rd.noalias() += m_A.transpose() * y;
        m_original_rd += m_rho * (x - m_xi);
        m_ry.noalias() = m_A * x; m_ry -= m_b; m_ry -= m_delta * (y - m_eta);
        T value = norm(m_ry);
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            for (isize j = 0; j < block.local.size(); ++j) block.local(j) = x(block.indices[usize(j)]);
            block.work.noalias() = block.F * block.local;
            block.work += block.f; block.work += s.segment(o, d); block.work -= m_delta * (z.segment(o, d) - m_zeta.segment(o, d));
            block.gradient.noalias() = block.F.transpose() * z.segment(o, d);
            if (block.quadratic)
            {
                block.gradient.noalias() = block.Q.template selfadjointView<Eigen::Upper>() * block.local;
                block.work(0) += T(.5) * block.local.dot(block.gradient);
                block.gradient *= z(o); block.gradient.noalias() += block.F.transpose() * z.segment(o, d);
            }
            for (isize j = 0; j < block.local.size(); ++j) m_original_rd(block.indices[usize(j)]) += block.gradient(j);
            value = (std::max)(value, norm(block.work));
            if (block.cone)
            {
                block.scaling.apply_Winv(s.segment(o, d), block.work);
                block.scaling.apply_W(z.segment(o, d), block.work2);
                jordan<T>(block.work, block.work2, block.rc); block.rc(0) -= target;
                value = (std::max)(value, norm(block.rc));
            }
            else value = (std::max)(value, std::abs(s(o) * z(o) - target));
        }
        return (std::max)(value, norm(m_original_rd));
    }

    bool take_step(T target)
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::take_step");
        T alpha = step_length(m_s, m_z, m_ds, m_dz);
        const T before = merit(m_x, m_y, m_s, m_z, target);
        for (isize backtrack = 0; backtrack < 30; ++backtrack)
        {
            m_trial_x = m_x + alpha * m_dx; m_trial_y = m_y + alpha * m_dy;
            m_trial_s = m_s + alpha * m_ds; m_trial_z = m_z + alpha * m_dz;
            const T after = merit(m_trial_x, m_trial_y, m_trial_s, m_trial_z, target);
            if (all_interior(m_trial_s, m_trial_z) && std::isfinite(after) && after <= (T(1) - T(1e-4) * alpha) * before)
            { m_x = m_trial_x; m_y = m_trial_y; m_s = m_trial_s; m_z = m_trial_z; return true; }
            alpha *= T(.5);
        }
        return false;
    }

    void report()
    {
        PIQP_TRACY_ZoneScopedN("piqp::constrained::Solver::report");
        m_result.x.array() = m_E.array() * m_x.array();
        m_result.y.array() = m_eq_scale.array() * m_y.array() / m_gamma;
        m_result.primal_residual = T(0); m_result.complementarity = T(0);
        for (auto& block : m_blocks)
        {
            const isize o = block.offset, d = block.f.size();
            block.work = m_s.segment(o, d) / block.scale;
            block.work2 = m_z.segment(o, d) * (block.scale / m_gamma);
            m_result.complementarity += block.work.dot(block.work2);
            if (block.kind == 0) { m_result.s_l(block.input) = block.work(0); m_result.z_l(block.input) = block.work2(0); }
            if (block.kind == 1) { m_result.s_u(block.input) = block.work(0); m_result.z_u(block.input) = block.work2(0); }
            if (block.kind == 2) { m_result.s_bl(block.input) = block.work(0); m_result.z_bl(block.input) = block.work2(0); }
            if (block.kind == 3) { m_result.s_bu(block.input) = block.work(0); m_result.z_bu(block.input) = block.work2(0); }
            if (block.kind == 4) { m_result.quadratic_slack(block.input) = block.work(0); m_result.quadratic_dual(block.input) = block.work2(0); }
            if (block.kind == 5)
            {
                if (m_model->cone_constraints[usize(block.input)].type == ConeType::rotated_second_order)
                { rotate(block.work); block.work *= T(.5); rotate(block.work2); }
                const isize offset = m_result.cone_offsets[usize(block.input)];
                m_result.cone_slack.segment(offset, d) = block.work;
                m_result.cone_dual.segment(offset, d) = block.work2;
            }
        }
        m_original_rd.noalias() = m_model->P.template selfadjointView<Eigen::Upper>() * m_result.x;
        m_result.objective = T(.5) * m_result.x.dot(m_original_rd) + m_model->c.dot(m_result.x);
        m_original_rd += m_model->c;
        m_original_rd.noalias() += m_model->A.transpose() * m_result.y;
        m_affine_dual = m_result.z_u - m_result.z_l;
        m_original_rd.noalias() += m_model->G.transpose() * m_affine_dual;
        m_original_rd += m_result.z_bu - m_result.z_bl;
        m_ry.noalias() = m_model->A * m_result.x; m_ry -= m_model->b;
        m_result.primal_residual = norm(m_ry);
        m_affine_value.noalias() = m_model->G * m_result.x;
        T cone_dual_violation = T(0);
        for (auto& block : m_blocks)
        {
            T residual = T(0);
            if (block.kind == 0) residual = m_affine_value(block.input) - m_model->h_l(block.input) - m_result.s_l(block.input);
            if (block.kind == 1) residual = m_affine_value(block.input) - m_model->h_u(block.input) + m_result.s_u(block.input);
            if (block.kind == 2) residual = m_result.x(block.input) - m_model->x_l(block.input) - m_result.s_bl(block.input);
            if (block.kind == 3) residual = m_result.x(block.input) - m_model->x_u(block.input) + m_result.s_bu(block.input);
            if (block.quadratic)
            {
                const auto& source = m_model->quadratic_constraints[usize(block.input)];
                for (isize j = 0; j < block.local.size(); ++j) block.local(j) = m_result.x(block.indices[usize(j)]);
                block.gradient.noalias() = source.Q.template selfadjointView<Eigen::Upper>() * block.local;
                residual = T(.5) * block.local.dot(block.gradient) + source.q.dot(block.local) - source.upper + m_result.quadratic_slack(block.input);
                block.gradient += source.q;
                for (isize j = 0; j < block.local.size(); ++j) m_original_rd(block.indices[usize(j)]) += m_result.quadratic_dual(block.input) * block.gradient(j);
            }
            if (block.cone)
            {
                const auto& source = m_model->cone_constraints[usize(block.input)];
                const isize o = m_result.cone_offsets[usize(block.input)], d = source.f.size();
                block.work = source.f;
                entries(source.F, [&](isize i, isize j, T v) {
                    const isize index = source.indices.empty() ? j : source.indices[usize(j)];
                    block.work(i) += v * m_result.x(index);
                    m_original_rd(index) -= v * m_result.cone_dual(o + i);
                });
                m_result.primal_residual = (std::max)(m_result.primal_residual, cone_violation(block.work, source.type));
                block.work -= m_result.cone_slack.segment(o, d);
                residual = norm(block.work);
                cone_dual_violation = (std::max)(cone_dual_violation, cone_violation(m_result.cone_dual.segment(o, d), source.type));
            }
            m_result.primal_residual = (std::max)(m_result.primal_residual, std::abs(residual));
        }
        m_result.dual_residual = (std::max)(norm(m_original_rd), cone_dual_violation);
    }
    static T cone_violation(CVecRef<T> v, ConeType type)
    {
        if (type == ConeType::second_order) return (std::max)(T(0), v.tail(v.size() - 1).stableNorm() - v(0));
        const T root_two = std::sqrt(T(2));
        return (std::max)(T(0), std::hypot((v(0) - v(1)) / root_two, v.tail(v.size() - 2).stableNorm()) - (v(0) + v(1)) / root_two);
    }
    bool converged() const
    {
        if (!m_result.x.allFinite() || !m_result.y.allFinite() || !std::isfinite(m_result.objective) ||
            !std::isfinite(m_result.primal_residual) || !std::isfinite(m_result.dual_residual) || !std::isfinite(m_result.complementarity)) return false;
        const T primal_tolerance = m_settings.eps_abs;
        const T dual_tolerance = m_settings.eps_abs + m_settings.eps_rel * (std::max)(T(1), norm(m_model->c));
        const T gap_tolerance = m_settings.eps_duality_gap_abs + m_settings.eps_duality_gap_rel * std::abs(m_result.objective);
        return m_result.primal_residual <= primal_tolerance && m_result.dual_residual <= dual_tolerance &&
               m_result.complementarity >= T(0) && m_result.complementarity <= gap_tolerance;
    }
};

}

template<typename T> using ConstrainedDenseSolver = constrained::Solver<T, int, PIQP_DENSE>;
template<typename T, typename I = int> using ConstrainedSparseSolver = constrained::Solver<T, I, PIQP_SPARSE>;

}
#endif
