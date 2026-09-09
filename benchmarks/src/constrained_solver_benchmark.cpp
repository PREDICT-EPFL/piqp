#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include "piqp/constrained/solver.hpp"

using namespace piqp;
using Matrix = Mat<double>;
using Vector = Vec<double>;
using Model = dense::Model<double>;

static Model ellipsoid(int n, bool conic)
{
    Matrix P = Matrix::Identity(n, n), Q = Matrix::Zero(n, n);
    Vector c(n);
    for (int i = 0; i < n; ++i) { c(i) = -2.0 / std::sqrt(n); Q(i, i) = 1.0 + 9.0 * i / (n - 1); }
    Model model(P, c);
    if (conic)
    {
        Matrix F = Matrix::Zero(n + 2, n);
        Vector f = Vector::Zero(n + 2); f(0) = .5; f(1) = 1;
        for (int i = 0; i < n; ++i) F(i + 2, i) = std::sqrt(Q(i, i));
        model.cone_constraints.emplace_back(F, f, ConeType::rotated_second_order);
    }
    else model.quadratic_constraints.emplace_back(Q, Vector::Zero(n), .5);
    return model;
}

static Model local_mixed(int stages)
{
    int n = 3 * stages;
    Matrix P = Matrix::Identity(n, n);
    Vector c(n);
    for (int i = 0; i < n; ++i) c(i) = -1.5 - .3 * std::sin(double(i + 1));
    Model model(P, c);
    for (int k = 0; k < stages; ++k)
    {
        std::vector<isize> ids{3*k, 3*k+1, 3*k+2};
        Matrix Q = Matrix::Identity(3, 3); Q(0, 1) = Q(1, 0) = .2;
        model.quadratic_constraints.emplace_back(Q, Vector::Zero(3), .7, ids);
        Matrix F = Matrix::Zero(3, 3); F(1, 0) = 1; F(2, 1) = 1;
        Vector f = Vector::Zero(3); f(0) = .8;
        model.cone_constraints.emplace_back(F, f, ConeType::second_order, ids);
        if (k + 1 < stages) { P(3*k+2, 3*k+3) = P(3*k+3, 3*k+2) = -.1; }
    }
    model.P = P;
    return model;
}

static Model control(int horizon)
{
    int n = 6 * horizon + 4;
    Matrix P = Matrix::Zero(n, n), A = Matrix::Zero(4 * (horizon + 1), n);
    Vector c = Vector::Zero(n), b = Vector::Zero(A.rows());
    const double dt = .2;
    Matrix dynamics = Matrix::Identity(4, 4); dynamics(0, 2) = dynamics(1, 3) = dt;
    Matrix input = Matrix::Zero(4, 2); input(0, 0) = input(1, 1) = .5 * dt * dt;
    input(2, 0) = input(3, 1) = dt;
    A.topLeftCorner(4, 4).setIdentity(); b(0) = 1; b(1) = .5;
    for (int k = 0; k < horizon; ++k)
    {
        P.block(6*k, 6*k, 4, 4).diagonal() << 1, 1, .1, .1;
        P(6*k+4, 6*k+4) = P(6*k+5, 6*k+5) = .05;
        A.block(4+4*k, 6*k, 4, 4) = -dynamics;
        A.block(4+4*k, 6*k+4, 4, 2) = -input;
        A.block(4+4*k, 6*k+6, 4, 4).setIdentity();
    }
    P.bottomRightCorner(4, 4).diagonal() << 10, 10, 1, 1;
    Model model(P, c, A, b);
    for (int k = 0; k < horizon; ++k)
    {
        Matrix F = Matrix::Zero(3, 2); F.bottomRows(2).setIdentity();
        Vector f = Vector::Zero(3); f(0) = 1;
        model.cone_constraints.emplace_back(F, f, ConeType::second_order, std::vector<isize>{6*k+4, 6*k+5});
    }
    Matrix Q = 2 * Matrix::Identity(4, 4);
    model.quadratic_constraints.emplace_back(Q, Vector::Zero(4), .25,
        std::vector<isize>{6*horizon, 6*horizon+1, 6*horizon+2, 6*horizon+3});
    return model;
}

static Model cone_dimension(int dimension)
{
    Model model(Matrix::Identity(2, 2), Vector::Constant(2, -2));
    Matrix F = Matrix::Zero(dimension, 2);
    for (int i = 1; i < dimension; ++i)
        F(i, (i-1)%2) = 1.0 / std::sqrt(double((dimension-1+(i%2))/2));
    Vector f = Vector::Zero(dimension); f(0) = 1;
    model.cone_constraints.emplace_back(F, f);
    return model;
}

static SparseMat<double, int> read_sparse(std::istream& in)
{
    int rows, cols, count; in >> rows >> cols >> count;
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(usize(count));
    for (int k=0; k<count; ++k)
    { int i,j; double v; in >> i >> j >> v; entries.emplace_back(i,j,v); }
    SparseMat<double, int> matrix(rows,cols); matrix.setFromTriplets(entries.begin(),entries.end());
    return matrix;
}

static Vector read_vector(std::istream& in, isize size)
{
    Vector out(size); for (isize i=0; i<size; ++i) in >> out(i); return out;
}

static sparse::Model<double, int> read_public(const std::string& path)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open instance " + path);
    int n; in >> n;
    Vector c=read_vector(in,n);
    auto A=read_sparse(in); Vector b=read_vector(in,A.rows());
    auto G=read_sparse(in); Vector h=read_vector(in,G.rows());
    sparse::Model<double, int> model(SparseMat<double,int>(n,n),c,A,b,G,nullopt,h);
    int count; in >> count;
    for (int k=0; k<count; ++k)
    {
        int rotated,support; in >> rotated >> support;
        std::vector<isize> ids(static_cast<usize>(support)); for (auto& i:ids) in >> i;
        auto F=read_sparse(in); Vector f=read_vector(in,F.rows());
        model.cone_constraints.emplace_back(F,f,rotated ? ConeType::rotated_second_order : ConeType::second_order,ids);
    }
    if (!in) throw std::runtime_error("invalid sparse instance " + path);
    return model;
}

static sparse::Model<double, int> sparse_model(const Model& m)
{
    sparse::Model<double, int> out(m.P.sparseView(), m.c, SparseMat<double, int>(m.A.sparseView()), m.b);
    for (const auto& q : m.quadratic_constraints)
        out.quadratic_constraints.emplace_back(q.Q.sparseView(), q.q, q.upper, q.indices);
    for (const auto& c : m.cone_constraints)
        out.cone_constraints.emplace_back(c.F.sparseView(), c.f, c.type, c.indices);
    return out;
}

static Vector local(const Vector& x, const std::vector<isize>& ids)
{
    if (ids.empty()) return x;
    Vector v(isize(ids.size())); for (isize i = 0; i < v.size(); ++i) v(i) = x(ids[usize(i)]);
    return v;
}

template<typename Input>
static double violation(const Input& m, const Vector& x)
{
    if (!x.allFinite()) return std::numeric_limits<double>::infinity();
    double v = m.b.size() ? (m.A*x-m.b).template lpNorm<Eigen::Infinity>() : 0;
    if (m.G.rows())
    {
        Vector gx=m.G*x;
        for (isize i=0; i<gx.size(); ++i) v=std::max({v,m.h_l(i)-gx(i),gx(i)-m.h_u(i)});
    }
    for (const auto& q : m.quadratic_constraints)
    {
        Vector u = local(x, q.indices);
        v = std::max(v, .5*u.dot(q.Q*u)+q.q.dot(u)-q.upper);
    }
    for (const auto& c : m.cone_constraints)
    {
        Vector s = c.F*local(x, c.indices)+c.f;
        if (c.type == ConeType::rotated_second_order)
        {
            double a=s(0), b=s(1); s(0)=a+b; s(1)=a-b; s.tail(s.size()-2)*=std::sqrt(2.0);
        }
        v=std::max(v, s.tail(s.size()-1).norm()-s(0));
    }
    return v;
}

static void write_matrix(std::ostream& out, const Matrix& a)
{
    out << a.rows() << ' ' << a.cols() << '\n';
    for (isize i=0; i<a.rows(); ++i) { for (isize j=0; j<a.cols(); ++j) out << a(i,j) << ' '; out << '\n'; }
}

static void export_model(const std::string& path, const Model& m)
{
    std::ofstream out(path); out << std::setprecision(17);
    if (!out) throw std::runtime_error("cannot create instance " + path);
    write_matrix(out,m.P); write_matrix(out,m.c); write_matrix(out,m.A); write_matrix(out,m.b);
    out << m.quadratic_constraints.size() << '\n';
    for (const auto& q:m.quadratic_constraints)
    {
        out << q.indices.size(); for (auto i:q.indices) out << ' ' << i; out << '\n';
        write_matrix(out,q.Q); write_matrix(out,q.q); out << q.upper << '\n';
    }
    out << m.cone_constraints.size() << '\n';
    for (const auto& c:m.cone_constraints)
    {
        out << int(c.type == ConeType::rotated_second_order) << ' ' << c.indices.size();
        for (auto i:c.indices) out << ' ' << i; out << '\n';
        write_matrix(out,c.F); write_matrix(out,c.f);
    }
}

static Vector analytic_ellipsoid(int n)
{
    double lower=0,upper=1;
    auto squared_norm=[&](double multiplier)
    {
        double value=0;
        for (int i=0; i<n; ++i)
        {
            double q=1.0+9.0*i/(n-1),x=2.0/(std::sqrt(n)*(1+multiplier*q));
            value+=q*x*x;
        }
        return value;
    };
    while (squared_norm(upper)>1) upper*=2;
    for (int k=0; k<80; ++k)
    {
        double mid=.5*(lower+upper);
        if (squared_norm(mid)>1) lower=mid; else upper=mid;
    }
    Vector x(n);
    for (int i=0; i<n; ++i) x(i)=2.0/(std::sqrt(n)*(1+upper*(1.0+9.0*i/(n-1))));
    return x;
}

struct Metrics
{
    double stationarity=0, dual_cone=0, slack_error=0, complementarity=0, roundoff=0;
    bool finite=true;
};

static double membership(Vector value, ConeType type)
{
    if (type==ConeType::rotated_second_order)
    {
        double a=value(0),b=value(1);
        value(0)=(a+b)/std::sqrt(2.0); value(1)=(a-b)/std::sqrt(2.0);
    }
    return std::max(0.0,value.tail(value.size()-1).norm()-value(0));
}

template<typename Input>
static Metrics metrics(const Input& model, const constrained::Result<double>& result)
{
    Metrics out;
    Vector rd=model.P.template selfadjointView<Eigen::Upper>()*result.x+model.c;
    rd.noalias()+=model.A.transpose()*result.y;
    rd.noalias()+=model.G.transpose()*(result.z_u-result.z_l);
    rd+=result.z_bu-result.z_bl;
    out.finite=result.x.allFinite() && result.y.allFinite();
    auto pair=[&](const Vector& slack,const Vector& dual,bool scalar)
    {
        out.finite=out.finite && slack.allFinite() && dual.allFinite();
        out.complementarity+=slack.dot(dual);
        out.roundoff+=(slack.array()*dual.array()).abs().sum();
        if (scalar && dual.size())
        {
            out.dual_cone=std::max(out.dual_cone,-dual.minCoeff());
            out.slack_error=std::max(out.slack_error,-slack.minCoeff());
        }
    };
    pair(result.s_l,result.z_l,true); pair(result.s_u,result.z_u,true);
    pair(result.s_bl,result.z_bl,true); pair(result.s_bu,result.z_bu,true);
    pair(result.quadratic_slack,result.quadratic_dual,true); pair(result.cone_slack,result.cone_dual,false);
    Vector gx=model.G*result.x;
    for (isize i=0; i<gx.size(); ++i)
    {
        if (std::isfinite(model.h_l(i))) out.slack_error=std::max(out.slack_error,std::abs(gx(i)-model.h_l(i)-result.s_l(i)));
        if (std::isfinite(model.h_u(i))) out.slack_error=std::max(out.slack_error,std::abs(gx(i)-model.h_u(i)+result.s_u(i)));
    }
    for (isize i=0; i<result.x.size(); ++i)
    {
        if (std::isfinite(model.x_l(i))) out.slack_error=std::max(out.slack_error,std::abs(result.x(i)-model.x_l(i)-result.s_bl(i)));
        if (std::isfinite(model.x_u(i))) out.slack_error=std::max(out.slack_error,std::abs(result.x(i)-model.x_u(i)+result.s_bu(i)));
    }
    for (isize k=0; k<static_cast<isize>(model.quadratic_constraints.size()); ++k)
    {
        const auto& q=model.quadratic_constraints[usize(k)];
        Vector u=local(result.x,q.indices);
        Vector gradient=q.Q.template selfadjointView<Eigen::Upper>()*u+q.q;
        double error=.5*u.dot(gradient+q.q)-q.upper+result.quadratic_slack(k);
        out.slack_error=std::max(out.slack_error,std::abs(error));
        for (isize j=0; j<u.size(); ++j) rd(q.indices.empty() ? j : q.indices[usize(j)])+=result.quadratic_dual(k)*gradient(j);
    }
    for (isize k=0; k<static_cast<isize>(model.cone_constraints.size()); ++k)
    {
        const auto& c=model.cone_constraints[usize(k)];
        isize offset=result.cone_offsets[usize(k)],size=c.f.size();
        Vector dual=result.cone_dual.segment(offset,size),u=local(result.x,c.indices);
        Vector gradient=c.F.transpose()*dual;
        for (isize j=0; j<u.size(); ++j) rd(c.indices.empty() ? j : c.indices[usize(j)])-=gradient(j);
        Vector error=c.F*u+c.f-result.cone_slack.segment(offset,size);
        out.slack_error=std::max(out.slack_error,error.lpNorm<Eigen::Infinity>());
        out.dual_cone=std::max(out.dual_cone,membership(dual,c.type));
        out.slack_error=std::max(out.slack_error,membership(result.cone_slack.segment(offset,size),c.type));
    }
    out.stationarity=rd.lpNorm<Eigen::Infinity>();
    out.roundoff=64*std::numeric_limits<double>::epsilon()*(1+out.roundoff);
    out.finite=out.finite && rd.allFinite() && std::isfinite(out.complementarity);
    return out;
}

struct Structure
{
    isize jacobian=-1, kkt=-1, factor=-1;
};

struct InspectKKT : sparse::KKT<double,int>
{
    using sparse::KKT<double,int>::KKT;
    isize factor_nonzeros() const { return ldlt.L_vals.size()+ldlt.D.size(); }
};

static Structure structure(const Model&) { return {}; }

static Structure structure(const sparse::Model<double,int>& model)
{
    using Triplet=Eigen::Triplet<double,int>;
    using Sparse=SparseMat<double,int>;
    std::vector<Triplet> hessian,jacobian;
    isize rows=0,n=model.c.size();
    auto add_row=[&](const std::vector<isize>& indices,isize dimension)
    {
        for (isize r=0; r<dimension; ++r)
            for (isize i:indices) jacobian.emplace_back(int(i),int(rows+r),1);
        rows+=dimension;
    };
    auto support=[](const std::vector<isize>& indices,isize size)
    {
        if (!indices.empty()) return indices;
        std::vector<isize> full(static_cast<usize>(size)); std::iota(full.begin(),full.end(),isize(0)); return full;
    };
    constrained::entries(model.P,[&](isize i,isize j,double) { if (i<=j) hessian.emplace_back(int(i),int(j),1); });
    for (isize i=0; i<n; ++i) hessian.emplace_back(int(i),int(i),1);
    Sparse GT=model.G.transpose();
    for (isize r=0; r<model.G.rows(); ++r)
    {
        std::vector<isize> indices;
        for (Sparse::InnerIterator it(GT,r); it; ++it) indices.push_back(it.row());
        if (std::isfinite(model.h_l(r))) add_row(indices,1);
        if (std::isfinite(model.h_u(r))) add_row(indices,1);
    }
    for (isize i=0; i<n; ++i)
    {
        if (std::isfinite(model.x_l(i))) add_row({i},1);
        if (std::isfinite(model.x_u(i))) add_row({i},1);
    }
    for (const auto& q:model.quadratic_constraints)
    {
        auto indices=support(q.indices,q.q.size()); add_row(indices,1);
        constrained::entries(q.Q,[&](isize i,isize j,double) {
            if (i<=j) hessian.emplace_back(int(std::min(indices[usize(i)],indices[usize(j)])),int(std::max(indices[usize(i)],indices[usize(j)])),1);
        });
    }
    for (const auto& cone:model.cone_constraints)
    {
        std::vector<isize> indices;
        for (isize j=0; j<cone.F.cols(); ++j)
            if (cone.F.outerIndexPtr()[j+1]>cone.F.outerIndexPtr()[j])
                indices.push_back(cone.indices.empty() ? j : cone.indices[usize(j)]);
        add_row(indices,cone.f.size());
    }
    sparse::Data<double,int> data; data.resize(n,model.b.size(),rows);
    data.P_utri.setFromTriplets(hessian.begin(),hessian.end());
    data.GT.setFromTriplets(jacobian.begin(),jacobian.end()); data.AT=model.A.transpose();
    data.n_h_l=data.n_h_u=data.n_x_l=data.n_x_u=0;
    InspectKKT kkt(data);
    return {data.GT.nonZeros(),kkt.internal_kkt_mat().nonZeros(),kkt.factor_nonzeros()};
}

template<typename Solver, typename Input, typename Evaluation>
static bool run(const std::string& name, const std::string& backend, const Evaluation& m,
                const Input& input, int repetitions, Vector& reference)
{
    Solver solver;
    if (backend == "multistage") solver.settings().kkt_solver = KKTSolver::sparse_multistage;
    solver.settings().eps_abs=solver.settings().eps_duality_gap_abs=1e-8;
    solver.settings().eps_rel=solver.settings().eps_duality_gap_rel=1e-8;
    solver.settings().max_iter=150;
    solver.settings().compute_timings=true;
    solver.setup(input);
    Structure counts=backend=="sparse" ? structure(input) : Structure{};
    bool ok=true;
    for (int r=-1; r<repetitions; ++r)
    {
        double update_time=0;
        for (isize i=0; i<static_cast<isize>(m.quadratic_constraints.size()); ++i)
        { solver.update_quadratic(i,m.quadratic_constraints[usize(i)].upper); update_time+=solver.result().update_time; }
        for (isize i=0; i<static_cast<isize>(m.cone_constraints.size()); ++i)
        { solver.update_cone(i,m.cone_constraints[usize(i)].f); update_time+=solver.result().update_time; }
        solver.solve();
        if (r<0) continue;
        const auto& result=solver.result();
        double feasibility=violation(m,result.x);
        double objective=.5*result.x.dot(m.P*result.x)+m.c.dot(result.x);
        auto checked=metrics(m,result);
        feasibility=std::max(feasibility,checked.slack_error);
        double normalized_complementarity=checked.complementarity/(1+std::abs(objective));
        double agreement=reference.size() ? (reference-result.x).template lpNorm<Eigen::Infinity>() : 0;
        double analytic_error=0;
        if (name.find("ellipsoid_")==0)
            analytic_error=(result.x-analytic_ellipsoid(int(m.c.size()))).template lpNorm<Eigen::Infinity>();
        if (name.find("cone_dimension_")==0)
            analytic_error=(result.x-Vector::Constant(2,1/std::sqrt(2.0))).template lpNorm<Eigen::Infinity>();
        bool valid=result.status==PIQP_SOLVED && checked.finite && std::isfinite(objective) && feasibility<=1e-7 &&
                   checked.stationarity<=1e-7 && checked.dual_cone<=1e-7 && normalized_complementarity<=1e-7 &&
                   checked.complementarity>=-checked.roundoff && agreement<1e-4 && analytic_error<1e-4;
        if (!reference.size() && valid) reference=result.x;
        ok=ok&&valid;
        std::cout << name << ',' << backend << ',' << r << ',' << m.c.size() << ',' << int(result.status)
                  << ',' << result.iterations << ',' << result.refinement_steps << ',' << result.factor_retries
                  << ',' << counts.jacobian << ',' << counts.kkt << ',' << counts.factor << ',' << result.setup_time << ',' << update_time << ',' << result.solve_time << ',' << result.factor_time
                  << ',' << feasibility << ',' << checked.stationarity << ',' << checked.dual_cone << ',' << checked.slack_error
                  << ',' << checked.complementarity << ',' << normalized_complementarity << ',' << objective << ',' << agreement << ',' << analytic_error << ',' << valid << '\n';
    }
    return ok;
}

int main(int argc, char** argv)
{
    try
    {
        int repetitions=argc>1 ? std::stoi(argv[1]) : 10;
        std::string directory=argc>2 ? argv[2] : "";
        std::string filter=argc>3 ? argv[3] : "";
        std::cout << std::setprecision(17) << "instance,backend,repetition,n,status,iterations,refinement_steps,factor_retries,jacobian_nnz,kkt_nnz,factor_nnz,setup_s,update_s,solve_s,factor_s,primal_violation,dual_residual,dual_cone_violation,slack_error,complementarity,normalized_complementarity,objective,backend_error,analytic_error,valid\n";
        bool ok=true;
        auto measure=[&](const std::string& name, const Model& model)
        {
            if (!filter.empty() && name.find(filter)==std::string::npos) return;
            if (!directory.empty()) export_model(directory+"/"+name+".dat",model);
            if (repetitions==0) return;
            Vector reference;
            ok=run<ConstrainedDenseSolver<double>>(name,"dense",model,model,repetitions,reference)&&ok;
            auto sparse=sparse_model(model);
            ok=run<ConstrainedSparseSolver<double>>(name,"sparse",model,sparse,repetitions,reference)&&ok;
#ifdef PIQP_HAS_BLASFEO
            ok=run<ConstrainedSparseSolver<double>>(name,"multistage",model,sparse,repetitions,reference)&&ok;
#endif
        };
        for (int n:{8,32,128}) { measure("ellipsoid_qcqp_"+std::to_string(n),ellipsoid(n,false)); measure("ellipsoid_socp_"+std::to_string(n),ellipsoid(n,true)); }
        for (int n:{4,16,64}) measure("local_mixed_"+std::to_string(n),local_mixed(n));
        for (int n:{10,20,40,80}) measure("control_"+std::to_string(n),control(n));
        for (int n:{8,32,128,512,2048}) measure("cone_dimension_"+std::to_string(n),cone_dimension(n));
        if (repetitions>0) for (int i=4; i<argc; ++i)
        {
            std::string path=argv[i], name=path.substr(path.find_last_of('/')+1);
            name=name.substr(0,name.find_last_of('.'));
            auto model=read_public(path); Vector reference;
            ok=run<ConstrainedSparseSolver<double>>(name,"sparse",model,model,repetitions,reference)&&ok;
        }
        return ok ? 0 : 1;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
