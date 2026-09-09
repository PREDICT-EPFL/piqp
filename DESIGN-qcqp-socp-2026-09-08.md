# Convex quadratic and second-order cone constraints in PIQP

Design exploration, 8 September 2026, expanded 9 September 2026. Based on PIQP at `5d84959` and the sources linked below. This note consolidates the algorithm, numerical design, interfaces, initialization, and evaluation discussion for a future implementation plan. All proposed API names are illustrative. No solver implementation is part of this note.

The hypothesis is sound for convex quadratically constrained quadratic programs, abbreviated QCQPs, and second-order cone programs, abbreviated SOCPs. Both admit Newton systems that fit PIQP's multistage factorization when each constraint has suitable local support. Native QCQP support is the smaller extension. SOCP support needs a cone-aware interior-point method and block scaling. The practical PIQP algorithm itself has no established general convergence theorem in its paper. The extensions need their own derivation and numerical evidence.

The intended scope keeps the positive semidefinite quadratic objective and affine equalities. Quadratic inequalities have positive semidefinite Hessians and upper bounds. General indefinite quadratic constraints and quadratic equalities would be a different solver project.

## Existing implementation and scope

The relevant existing code is:

- [The public solver interface](include/piqp/solver.hpp) and [Python bindings](interfaces/python/src/piqp_python.cpp), which expose `setup`, `update`, `solve`, and `result`.
- [The iteration](include/piqp/solver.tpp), particularly `calculate_mu`, `calculate_step`, and the residual and predictor-corrector calculations.
- [Slack elimination](include/piqp/kkt_system.tpp) and [the backend interface](include/piqp/kkt_solver_base.hpp), which currently pass inequality scaling as a vector.
- [The multistage backend](include/piqp/sparse/multistage_kkt.tpp), particularly `extract_arrow_structure`, `update_scalings_and_factor`, and `construct_kkt_fac`.
- [Dense equilibration](include/piqp/dense/preconditioner.tpp), [sparse equilibration](include/piqp/sparse/preconditioner.tpp), and their `IdentityPreconditioner` alternatives in the corresponding headers.
- [Dense model](include/piqp/dense/model.hpp), [sparse model](include/piqp/sparse/model.hpp), [results](include/piqp/results.hpp), and [the multistage formulation](docs/_pages/multistage.md).
- [C declarations](interfaces/c/include/piqp.h), [C implementation](interfaces/c/src/piqp.cpp), [MATLAB class](interfaces/matlab/piqp.m), and [Octave class](interfaces/octave/piqp.m).

The following derivations are an analysis of how new constraints would fit those components. They establish a plausible linear-algebra design, not a finished nonlinear algorithm. The objective remains native, \(\tfrac12x^TPx+c^Tx\), with \(P\succeq0\). Existing affine constraints and variable bounds remain available alongside the new constraints.

## Native quadratic constraints

For a quadratic constraint, write

$$
f_i(x)=\tfrac12 x^TQ_ix+q_i^Tx-u_i\leq0,\qquad Q_i\succeq0.
$$

Introduce scalar slack \(s_i>0\) and multiplier \(z_i>0\). The constraint Jacobian has row \((Q_ix+q_i)^T\), and the Lagrangian Hessian becomes

$$
H=P+\sum_i z_iQ_i.
$$

Stack the quadratic Jacobian rows in \(J\). For the natural extension of PIQP's dual regularization, the linearized quadratic feasibility equation is

$$
J\Delta x+\Delta s-\delta\Delta z=r_f,
\qquad
Z\Delta s+S\Delta z=r_c.
$$

Eliminating \(\Delta s\) gives a negative dual block \(-D\), where

$$
D=Z^{-1}S+\delta I\succ0.
$$

The reduced primal matrix therefore gains

$$
\sum_i z_iQ_i+J^TD^{-1}J.
$$

Both terms are positive semidefinite. Positive primal regularization preserves the positive definiteness needed for a Cholesky factorization after elimination. Before equality elimination, the corresponding saddle matrix has a positive definite primal block and a negative definite regularized dual block. This is compatible with PIQP's regularized linear algebra.

HPIPM provides a direct precedent. Its QCQP method updates the Lagrangian Hessian and constraint Jacobian at each iteration and reuses its quadratic program, or QP, factorization routines. Sections III-A and III-B describe the Newton system and implementation, including a conditional predictor-corrector scheme. These are useful engineering precedents, not a proof for an extended PIQP algorithm. [HPIPM QCQP paper](https://publications.syscop.de/Frison2022.pdf).

The work beyond matrix assembly matters. Quadratic feasibility has the exact Taylor remainder \(\tfrac12\Delta x^TQ_i\Delta x\). Stationarity also has mixed terms involving \(\Delta z_iQ_i\Delta x\). The predictor-corrector derivation and its safeguards need to account for nonlinear residuals. Evaluating the true quadratic constraints at each iterate is essential. Positivity of the independent slacks does not imply feasibility of the current primal point in an infeasible-start method.

Keep the original objective matrix distinct from the iteration's Lagrangian Hessian. Replacing `data.P` everywhere with \(H\) would corrupt objective evaluation and other calculations. Likewise, the current affine `G` operations cannot substitute for evaluating quadratic constraints. The backend can consume assembled Newton data while the solver retains the original model.

## Affine second-order cones

For an individual second-order cone, abbreviated SOC, use an affine map

$$
F_jx+f_j\in\mathcal Q^{d_j},\qquad
\mathcal Q^d=\{(t,v):t\geq\lVert v\rVert_2\}.
$$

A symmetric primal-dual cone scaling replaces the scalar slack-to-dual ratios with a positive definite block for each cone. Calling the resulting regularized dual block \(D_j\), elimination contributes

$$
F_j^TD_j^{-1}F_j.
$$

The sign convention used to write the slack equality does not change this contribution. Deriving \(D_j\) in the selected scaled coordinates must precede implementation. The same positive-definiteness argument then applies to the reduced primal system.

The cone layer needs interior initialization, Jordan products and division, Nesterov-Todd scaling, abbreviated NT, cone boundary step lengths, and a matching corrector. With the ordinary Lorentz-cone convention, the identity is \(e=(1,0)\) and

$$
(a,u)\circ(b,v)=(ab+u^Tv,\;av+bu).
$$

An NT convention used by the references is \(\lambda=Wz=W^{-T}s\). Its scaled complementarity equation yields a symmetric dual block involving \(W^TW\). The implementation must derive where PIQP's proximal regularization enters before defining \(D_j\). Replacing a diagonal vector with a dense matrix without transforming the right-hand side and corrector would be insufficient.

Complementarity normalization must follow the selected barrier and Jordan-algebra conventions. Cone dimension, Jordan rank, and barrier degree are different quantities. The usual unnormalized logarithmic barrier for a Lorentz cone has degree two, while implementations may use a rescaled convention. Clarabel's SOC implementation reports degree one, and QOCO computes its algorithmic measure as \(s^Tz/m\), with \(m\) the number of cone coordinates. PIQP must choose its measure, centrality target, and predictor-corrector formulas together rather than combine these conventions. [Clarabel cone implementation](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/socone.rs), [QOCO complementarity calculation](https://github.com/qoco-org/qoco/blob/main/src/kkt.c).

The maximum interior step follows the cone boundary equation and the sign of the leading coordinate. Its implementation needs stable roots, nearly tangent directions, and a fraction-to-boundary rule. Componentwise positivity and the existing scalar step calculation are insufficient. Scalar boundary-repair code in the main iteration also needs a cone equivalent. Vandenberghe's cone-programming report gives the mathematical conventions in Sections 2–5 and step and scaling calculations in Sections 8–10. [The CVXOPT linear and quadratic cone program solvers](https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf).

ECOS demonstrates practical Nesterov-Todd scaling, regularization, iterative refinement, and sparse expansions of structured cone blocks. Such expansions matter for large cones, where storing dense scaling blocks would be wasteful. [ECOS algorithm paper](https://web.stanford.edu/~boyd/papers/pdf/ecos_ecc.pdf).

QOCO is a particularly close conic reference because it keeps the quadratic objective and separate affine equalities. Its paper covers NT scaling in Section 2.2, predictor-corrector directions in Section 2.3, initialization in Section 2.4, and numerical implementation in Section 3. Its regularized numerical solves are not a substitute for deriving PIQP's proximal iteration. [QOCO paper](https://arxiv.org/abs/2503.12658).

## Convergence and solver statuses

Remark 2 of the PIQP paper explicitly leaves theoretical convergence of practical Algorithm 1 outside the paper's scope. The underlying interior point-proximal method of multipliers, abbreviated IP-PMM, has a convergence analysis under its assumptions. PIQP's practical predictor-corrector choices and heuristic regularization updates do not automatically satisfy that analysis. [PIQP paper, Section III](https://arxiv.org/pdf/2304.00290), [IP-PMM paper](https://arxiv.org/abs/1904.10369).

Predictor-corrector methods are not inherently beyond theory. Safeguarded variants have convergence and complexity results, but those results depend on the actual algorithm and assumptions. A theorem for such a variant does not establish one for practical PIQP. [Salahi, Peng, and Terlaky on Mehrotra-type predictor-corrector algorithms](https://epubs.siam.org/doi/10.1137/050628787).

The proximal approach is not confined to the nonnegative orthant. Pougkakiotis and Gondzio extended IP-PMM to semidefinite programming. This supports investigating a cone extension, but does not prove convergence for this proposed combination of quadratic objectives, second-order cones, and implementation heuristics. [Proximal interior-point method for semidefinite programming](https://arxiv.org/abs/2010.14285).

Positive definiteness establishes factorability in exact arithmetic under the stated positivity assumptions. It establishes neither progress of the nonlinear iteration nor finite-precision reliability. Convergence under appropriate regularity, behavior without strict feasibility, termination residuals, and infeasibility certificates need separate attention. In particular, conic problems can exhibit weak infeasibility and failures of strong duality. Existing QP status heuristics should not be presented as conic certificates without derivation. Adopting the data conventions of ECOS or Clarabel does not require adopting their homogeneous embedding. That would be a separate algorithmic decision if certificate requirements justify it. [Clarabel formulation and analysis](https://arxiv.org/abs/2405.12762).

Successful termination must use the original model's Karush-Kuhn-Tucker conditions, abbreviated KKT: stationarity, equality and inequality feasibility, dual-cone membership, and complementarity. Quadratic constraints require their true values and gradients. Any reported dual objective or gap needs a derivation for the chosen formulation, including singular Hessians. A complementarity sum alone is not an unconditional primal-dual objective gap at an infeasible iterate. A general convergence theorem can remain a separate research objective while implementation proceeds with explicitly limited status claims.

## Native QCQP or conic reformulation

There are two ways to support convex QCQPs once a cone solver exists. With \(Q_i=L_i^TL_i\) and \(t_i=u_i-q_i^Tx\), the quadratic inequality is equivalent to

$$
(t_i,1,L_ix)\in\mathcal Q_r,
\quad
\mathcal Q_r=\{(a,b,v):a,b\geq0,\;2ab\geq\lVert v\rVert_2^2\}.
$$

This is an affine rotated-cone constraint. It needs no additional primal decision variable when the cone interface accepts affine maps. It does increase slack and dual dimensions, and factoring a sparse or singular \(Q_i\) can incur setup cost and fill. If a model already supplies \(L_i\), the conic representation can be attractive.

The reverse shortcut does not preserve the convex-QCQP formulation. Squaring \(\lVert Fx+f\rVert\leq a^Tx+b\) produces a quadratic with Hessian proportional to \(F^TF-aa^T\), which can be indefinite, and still requires \(a^Tx+b\geq0\). A native convex-QCQP solver therefore does not provide general SOCP support merely by squaring norms.

I favor native quadratic constraints for control problems, plus a cone implementation for affine norm constraints. A cone-only implementation would reduce the amount of distinct constraint logic and is a credible alternative if general SOCP coverage is the first priority. Performance comparisons should decide how much specialized QCQP code earns its place.

## Multistage and sparse linear algebra

PIQP's current multistage backend factors a reduced matrix of the form

$$
M_{\mathrm{QP}}=P+\operatorname{diag}(x_{\mathrm{reg}})
+\delta^{-1}A^TA+G^T\operatorname{diag}(z_{\mathrm{reg}}^{-1})G.
$$

Its stored blocks represent a block tridiagonal matrix plus a dense border for global variables. This is broader than a standard state-control Riccati recursion. The new terms fit that representation if each quadratic constraint or entire cone block touches only \((x_k,x_{k+1},g)\), or a subset of those variables. Constraints confined to one stage are an especially favorable case.

With bounded stage sizes, bounded global-variable dimension, and bounded constraint work per stage, assembly and factorization retain linear complexity in horizon length. This is a per-iteration statement, not an iteration-count or wall-clock performance guarantee. Large local cones and many local quadratic constraints increase the constants.

The support condition concerns an entire constraint:

- A diagonal quadratic Hessian spanning the horizon still produces a generally dense \(\nabla f_i\nabla f_i^T\). A single total-energy bound is an example.
- Different rows of one cone can touch different stages. Its dense scaling couples the union of those row supports, even if \(F^TF\) looks sparse.
- The affine term of a quadratic constraint contributes to its support just as its Hessian does.
- Declaring many variables global can preserve an arrow shape while making the dense border too large to be useful.

Some aggregate constraints can be reformulated to recover locality. For a sum of convex stage quadratics, introduce local epigraph variables \(e_k\geq f_k(x_k)\), linear accumulator equations, and an upper bound on the terminal accumulator. This is exact for an upper bound on the sum. It adds variables and should be a deliberate modeling choice.

`extract_arrow_structure` currently analyzes a matrix assembled from \(P\), \(A^TA\), and \(G^TG\). The extension must analyze all possible Newton nonzeros, including quadratic Hessians, gradient outer products, and whole-cone couplings. Detecting structure from the Jacobian at an initial point such as zero would miss later nonzeros. Use declared sparsity and support, independently of numerical cancellations. Constraint permutations must retain cone boundaries and map results back to input order.

For small local cones, dense block updates fit the existing Basic Linear Algebra Subroutines for Embedded Optimization, or BLASFEO, operations. [BLASFEO implementation](https://github.com/giaf/blasfeo). General sparse problems may benefit from retaining cone variables or using sparse expansions. I would share the factorization machinery while allowing backend-specific elimination, rather than forcing every backend to form the same condensed matrix. Forming normal equations can worsen conditioning, and large \(D_j^{-1}\) blocks can create fill. Regularization, factorization retries, and iterative refinement need evaluation on the new Newton systems.

## Global equilibration

The preconditioner needs explicit work for both extensions. Global data equilibration changes model coordinates at setup or update. NT scaling changes complementarity coordinates at every conic iteration. They solve different numerical problems and both are needed in a robust implementation.

Let \(x=E\hat x\), where \(E\) is positive diagonal, and multiply the objective by \(\gamma>0\). Using \(E\) here avoids confusion with the Newton dual blocks \(D_j\). The scaled objective is

$$
\hat P=\gamma E^TPE,\qquad \hat c=\gamma E^Tc.
$$

Multiplying quadratic constraint \(i\) by \(\alpha_i>0\) gives

$$
\hat Q_i=\alpha_i E^TQ_iE,\qquad
\hat q_i=\alpha_i E^Tq_i,\qquad
\hat u_i=\alpha_i u_i.
$$

For a local descriptor, use the diagonal entries selected by its indices. Scaling only \(q_i\) and \(u_i\) would change the problem. Choosing scales only from the current Jacobian would also miss curvature when \(Q_ix+q_i=0\). The scale-selection rule needs information from \(Q_i\), including constraints with zero linear terms.

For a cone block, a common positive scale \(\beta_j\) preserves membership:

$$
\hat F_j=\beta_j F_jE,\qquad \hat f_j=\beta_j f_j.
$$

Arbitrary independent row scales within a Lorentz cone change its geometry. More general cone automorphisms are possible, but a common block scale is the minimal baseline. Clarabel's `equilibrate` and `rectify_equilibration` show how a global equilibration routine can enforce this restriction. ECOS provides another implementation reference in `equil.c`. [Clarabel problem equilibration](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/implementations/default/problemdata.rs), [ECOS equilibration](https://github.com/embotech/ecos/blob/develop/src/equil.c).

For quadratic slacks and multipliers, the original-coordinate mappings are

$$
s_i=\hat s_i/\alpha_i,\qquad z_i=(\alpha_i/\gamma)\hat z_i.
$$

Cone slacks and duals obey the analogous mappings with \(\beta_j\). These factors follow from preserving the Lagrangian after objective scaling. They must also appear in residual unscaling, objective and complementarity reporting, stopping tolerances, and any future user-provided initialization. Cone conversion introduces another coordinate transform, described below.

Extend both dense and sparse Ruiz equilibration, their stored scale vectors, and the identity-preconditioner interfaces. Positive diagonal and block scaling preserve structural support, so equilibration should not invalidate multistage detection. The initial design can equilibrate at setup and either reuse or recompute scales on numerical updates, following PIQP's current contract. It need not rerun global equilibration during every Newton iteration. An identity-preconditioner prototype can help isolate algebra errors, but does not establish robustness on poorly scaled problems.

## Initialization from scratch

Here, initialization means PIQP's existing auxiliary solve for new primal and dual values. It does not mean an external starting point or reuse of the previous solution. In the current `solve_impl`, every solve resets regularization and slack-dual values, solves an auxiliary KKT system, and shifts and centers the result. `m_first_run` concerns timing accounting. Reusing a workspace and symbolic factorization is distinct from retaining iterates. [Current initialization](include/piqp/solver.tpp).

To expose the mathematics, stack the finite affine inequalities and bounds into one-sided rows \(\bar Gx+s=\bar h\). At initial \(\rho,\delta>0\), the current initialization corresponds to the unconstrained quadratic problem

$$
\min_{x,s}\;
\tfrac12x^TPx+c^Tx+\tfrac\rho2\|x\|^2
+\tfrac1{2\delta}\|Ax-b\|^2
+\tfrac1{2\delta}\|\bar Gx+s-\bar h\|^2
+\tfrac12\|s\|^2.
$$

The slacks in this auxiliary problem are unrestricted. Defining the residual multipliers and using \(s=-z\) gives

$$
\begin{bmatrix}
P+\rho I&A^T&\bar G^T\\
A&-\delta I&0\\
\bar G&0&-(1+\delta)I
\end{bmatrix}
\begin{bmatrix}x\\y\\z\end{bmatrix}
=\begin{bmatrix}-c\\b\\\bar h\end{bmatrix},
\qquad s=-z.
$$

This derivation describes the initial unit slack-to-dual scaling, before positivity repair. The implementation uses its own split storage for lower and upper bounds. The subsequent shifts change the auxiliary solution so that the interior-point method can start with positive pairs. Initial primal feasibility is unnecessary.

For affine cones, the same auxiliary quadratic problem remains available. Write each cone as \(-F_jx+s_j=f_j\), append its rows, and temporarily drop cone membership. After the linear solve, repair both \(s_j\) and \(z_j\) into the cone interior and choose the initial centrality scale. For an ordinary cone, a positive shift along \(e\) can enforce \(s_{j,0}>\|s_{j,1:}\|\). Merely making every coordinate positive does not suffice. The repair and centering policy must handle both very small and very large data scales.

QOCO's `initialize_ipm` gives a close source example of an identity-scaling KKT solve followed by `s = -z` and cone repair. ECOS's `init` illustrates initialization in a homogeneous formulation. Their systems differ from PIQP's proximal system, so they are references for the procedure rather than drop-in equations. [QOCO initialization source](https://github.com/qoco-org/qoco/blob/main/src/kkt.c), [ECOS initialization source](https://github.com/embotech/ecos/blob/develop/src/ecos.c).

For native quadratic constraints, substituting \(f_i(x)+s_i\) into that auxiliary residual penalty changes the problem. Eliminating the unrestricted slack produces

$$
\min_{s_i}\;\tfrac1{2\delta}(f_i(x)+s_i)^2+\tfrac12s_i^2
=\frac{f_i(x)^2}{2(1+\delta)}.
$$

This is quartic and can be nonconvex even when \(f_i\) is convex. For example, \((x^2-1)^2\) is nonconvex near zero. There is no direct quadratic auxiliary solve that incorporates these exact residuals.

The proposed first baseline is to run the existing objective-and-affine initializer, evaluate the true quadratic constraints at its \(x\), then choose positive scalar slack-dual pairs with controlled centrality for the new rows. Their initial residuals may be nonzero. A curvature-aware quadratic surrogate or an additional initialization step is a later option if benchmark evidence justifies it. Linearization at zero alone is a poor general replacement because a centered ellipsoid can have zero gradient there.

The initial unit cone scaling may be diagonal even though later scaling is dense. Symbolic allocation must therefore use the full structural pattern from setup. Initialization must not determine what nonzeros the workspace can subsequently represent.

A future warm-start interface accepting previous or external iterates is a separate feature. It would need dimension checks, original-coordinate mappings, all new slacks and multipliers, interior repair, and a centering policy. QOCO's optional primal starting point and the conic warm-start literature are useful later references, but this feature is not required by the present initialization discussion. [QOCO C API](https://qoco-org.github.io/qoco/qoco/api/C.html), [Skajaa, Andersen, and Ye on warm starting homogeneous interior-point methods](https://orbit.dtu.dk/en/publications/warmstarting-the-homogeneous-and-self-dual-interior-point-method-/).

## Interface design

The API precedents suggest the following choices:

| Solver | Existing interface idea | What PIQP should borrow |
| --- | --- | --- |
| HPIPM | Explicit optimal-control dimensions and quadratic data attached to stages, including `Qq` and `uq` | Local quadratic data and predictable workspace allocation |
| ECOS | Separate affine equalities and `Gx + s = h`, with cone dimensions in `dims` | Explicit cone boundaries and separate equality data |
| Clarabel | Native quadratic objective and `Ax + s = b`, with an ordered list of typed cones | Typed cone descriptors and a quadratic objective that does not need an epigraph |
| QOCO | Native quadratic objective, separate `A` and `G`, and scalar and second-order cone dimensions | A compact C representation and a direct conic-QP comparison |

These descriptions follow the [HPIPM Python QCQP example](https://github.com/giaf/hpipm/blob/master/examples/python/example_qcqp_getting_started.py), [ECOS Python interface](https://github.com/embotech/ecos-python), [Clarabel Python interface](https://clarabel.org/stable/python/getting_started_py/), and [QOCO C API](https://qoco-org.github.io/qoco/qoco/api/C.html).

### Python and shared descriptor semantics

My initial proposal extends PIQP's matrix interface with lists of constraint blocks. The following is proposed Python syntax, not executable PIQP code:

```python
solver = piqp.SparseSolver()
solver.settings.kkt_solver = piqp.KKTSolver.sparse_multistage

solver.setup(
	P=P, c=c,
	A=A, b=b,
	G=G, h_l=h_l, h_u=h_u,
	x_l=x_l, x_u=x_u,
	quadratic_constraints=[
		piqp.QuadraticConstraint(
			Q=Q_terminal, q=q_terminal, upper=terminal_upper,
			indices=terminal_indices,
		),
	],
	cone_constraints=[
		piqp.ConeConstraint(
			F=F_contact, f=f_contact,
			cone=piqp.SecondOrderCone(3),
			indices=contact_force_indices,
		),
	],
)
status = solver.solve()
```

For either descriptor, `indices` selects \(v=x[\text{indices}]\) in the specified order. Omitting it means the full decision vector. The quadratic descriptor means \(\tfrac12v^TQv+q^Tv\leq\text{upper}\). The cone descriptor means \(Fv+f\in K\). Dimensions and valid unique indices are setup checks. The convexity contract requires \(P\succeq0\) and \(Q_i\succeq0\). A mandatory sparse positive-semidefiniteness test is a separate setup-cost decision.

Local indexing avoids constructing a full-horizon sparse matrix for every small constraint. It is mathematical support information, so it also works for unstructured problems. Automatic multistage detection should remain the default. An optional explicit stage partition could later make the intended partition predictable, but a complete state-control modeling interface is unnecessary for initial support.

For a friction cone with `v = [fx, fy, fz]`, use the affine map

$$
F=\begin{bmatrix}0&0&\mu\\1&0&0\\0&1&0\end{bmatrix},\qquad f=0.
$$

This enforces \(\sqrt{f_x^2+f_y^2}\leq\mu f_z\), including the sign of the normal force when \(\mu>0\). A convenience constructor for \(\lVert Bv+d\rVert_2\leq a^Tv+b\) can pack \(F=[a^T;B]\) and \(f=[b;d]\). It would reduce ordering mistakes without adding a new solver representation.

I would initially support ordinary and rotated second-order cone descriptors. Rotated cones can map to ordinary cones through \(T(a,b,v)=(a+b,a-b,\sqrt2v)\). If internal slack and dual values are \(s_o,z_o\), return \(s_r=T^{-1}s_o\) and \(z_r=T^Tz_o\) before applying the other unscaling factors. The dual transform follows from preserving \(z_o^TTs_r\), so it is not the slack transform. Clarabel's documented cone list has an ordinary second-order cone but no distinct rotated-cone constructor. PIQP's rotated descriptor would be a proposed convenience. [Clarabel supported cones](https://clarabel.org/stable/api_cone_types/).

### C++

Extend the existing `sparse::Model<T, I>` and `dense::Model<T>` with descriptor arrays and add a `setup(model)` overload. Preserve the existing matrix overload. The model classes already own their matrices, so a second unrelated problem-container hierarchy is unnecessary. Proposed usage is:

```cpp
piqp::sparse::Model<double, int> model(P, c, A, b);
model.quadratic_constraints.emplace_back(Q, q, upper, local_indices);
model.cone_constraints.emplace_back(
	F, f, piqp::SecondOrderCone(3), force_indices);

piqp::SparseSolver<double, int> solver;
solver.setup(model);
auto status = solver.solve();
```

Use concrete descriptors for the supported mathematical forms. Virtual callbacks for arbitrary nonlinear constraints would introduce a different solver contract. Dense and sparse models should expose the same mathematics, with matrix storage following each model's existing convention. Accepting a small dense local block in a sparse model can be a convenience conversion if needed.

The proposed ownership rule is that setup copies model data into solver storage. The caller may then destroy or change the model without affecting the solver. Subsequent changes go through explicit update methods. Constructor details and matrix-view types remain implementation choices, but the ownership rule should be settled before binding work.

### C

A new setup entry point can accept descriptor arrays without changing existing public structure layouts. The following sketches the sparse descriptors. It is a proposal, not a declaration already in `piqp_typedef.h`:

```c
typedef struct {
	const piqp_csc* Q;
	const piqp_float* q;
	piqp_float upper;
	const piqp_int* indices;
} piqp_quadratic_constraint_sparse;

typedef struct {
	const piqp_csc* F;
	const piqp_float* f;
	piqp_cone_type type;
	const piqp_int* indices;
} piqp_cone_constraint_sparse;

typedef struct {
	piqp_int n_quadratic;
	const piqp_quadratic_constraint_sparse* quadratic;
	piqp_int n_cones;
	const piqp_cone_constraint_sparse* cones;
} piqp_constraints_sparse;

piqp_setup_sparse_extended(&workspace, &data, &constraints, &settings);
```

`piqp_cone_type` would initially distinguish ordinary and rotated second-order cones. Matrix dimensions determine local variable count and cone dimension. `indices == NULL` means all variables. Non-null index arrays contain one entry per local variable. Matrix and vector buffers need to remain valid only for the setup or update call because the solver copies them.

Dense descriptors need explicit dimensions and dense buffers. The current dense C wrapper maps row-major data, so the new buffers should follow that convention. Preserve the existing C application binary interface, abbreviated ABI, through new entry points and a getter for extended results rather than append fields to old public structs. Exact error reporting and read-only matrix-view types need review against the existing C API during implementation.

### MATLAB and Octave

The current classes interpret trailing positional setup arguments as settings. Adding constraint name-value pairs there would conflict with that parser. A new `setup(problem)` overload with a problem struct offers a clearer extension while retaining the old form:

```matlab
problem = struct('P', P, 'c', c, 'A', A, 'b', b);
problem.quadratic_constraints = {
	struct('Q', Q, 'q', q, 'upper', upper, 'indices', local_indices)
};
problem.cone_constraints = {
	struct('F', F, 'f', f, 'type', 'soc', 'indices', force_indices)
};
solver = piqp('sparse');
solver.setup(problem);
result = solver.solve();
```

Use cell arrays for differently sized blocks and ordinary structs for descriptors. A package namespace containing constructor helpers is unnecessary. The problem struct also carries existing affine bounds when present. Both MATLAB's compiled interface and Octave's interface need conversion and validation for these blocks.

### R

The separately maintained R package already has a model-oriented call and a one-shot helper. Extend those conventions with lists:

```r
model <- piqp(
	P = P, c = c, A = A, b = b,
	quadratic_constraints = list(
		list(Q = Q, q = q, upper = upper, indices = local_indices)
	),
	cone_constraints = list(
		list(F = F, f = f, type = "soc", indices = force_indices)
	),
	backend = "sparse"
)
result <- solve(model)
```

The same new arguments can extend `solve_piqp`. R support needs a coordinated change in its own repository after the native contract stabilizes. [Current R interface](https://predict-epfl.github.io/piqp-r/articles/piqp.html).

Python, C++, and C indices are zero-based. MATLAB, Octave, and R indices are one-based and convert at their binding boundary. All bindings should return constraints in the user's input order.

### Updates and results

Numerical updates should preserve dimensions, constraint order, selected variable indices, cone types and sizes, and declared sparse patterns. Require setup again for structural changes. This extends PIQP's existing allocation-free numerical-update goal. Clarabel likewise distinguishes numerical data updates from changing its cone collection. [Clarabel data updates](https://clarabel.org/stable/user_guide_data_updating/).

For example, proposed `update_quadratic(0, upper=new_upper)` and `update_cone(0, f=new_offset)` calls address descriptors by input position. The update contract must include zero-valued entries that may become nonzero later. Python can allocate wrapper objects, but the native solver workspace should not need allocation for numerical updates and solves.

Results should retain current primal and affine-dual fields and add scalar quadratic slacks and multipliers, plus flat cone slack and dual arrays with block offsets. Higher-level bindings can expose block views. For \(Fv+f\in K\), define cone duals \(z\in K^*\) with stationarity contribution \(-F^Tz\). For quadratic upper bounds, define \(z_i\geq0\) with contribution \(z_i(Q_iv+q_i)\). Local contributions scatter back into the selected full-vector indices. Report original-coordinate feasibility and complementarity. Keep physical constraint margins distinct from independent slacks at an infeasible iterate.

## Benchmarks and validation

There is no single collection that covers convex QCQPs, general SOCPs, multistage structure, and numerical conditioning as one complete counterpart to Maros–Mészáros. A useful suite combines public instances, application generators, and controlled numerical families.

| Source | Useful coverage | Selection needed |
| --- | --- | --- |
| [Conic Benchmark Library, CBLIB](https://cblib.zib.de/) | Public conic instances and a common interchange format | Select continuous linear, ordinary SOC, and rotated-SOC problems. Exclude unsupported cones and integer models. |
| [DIMACS conic challenge collection](https://github.com/vsdp/DIMACS) | Classical SOCP families, including structural mechanics, antenna, and scheduling instances, with scaled variants | Select the SOCP subset. The collection also includes semidefinite problems. |
| [Quadratic Programming Library, QPLIB](https://qplib.zib.de/statistics.html) | Quadratic models in native form, including some convex continuous QCQPs | Filter convex continuous instances with supported inequality orientation. Ordinary QPs, nonconvex models, and discrete models occur in the archive. |
| [ClarabelBenchmarks](https://github.com/oxfordcontrol/ClarabelBenchmarks) | Reproducible conic problem generators and comparison infrastructure | Retain supported cones and record generated problem data and versions. |
| [HPIPM QCQP applications](https://publications.syscop.de/Frison2022.pdf) | Control problems and a native QCQP comparison | Reproduce local-stage constraints and vary horizon and stage size. |

The archive sizes need careful interpretation. CBLIB's original 121-instance collection included 80 mixed-integer problems, so that headline count was not a continuous-SOCP count. The library has grown since. QPLIB's published statistics list 134 continuous instances and 32 convex instances, with the convex category also including ordinary QPs. Neither number describes a large ready-made convex-QCQP suite. Any continuous relaxation of an integer model should be labeled as a derived instance. [CBLIB description](https://cblib.zib.de/), [QPLIB statistics](https://qplib.zib.de/statistics.html).

The proposed generated suite includes terminal ellipsoids, stagewise norm and friction constraints, adjacent-stage quadratics, local cones with global variables, and aggregate energy constraints in both original and accumulator forms. Sweep horizon length separately from local dimensions and cone size. This distinguishes structural scaling from the cost of larger blocks.

Numerical families should vary variable and constraint scales independently, objective and constraint spectra, nearly dependent equalities, redundant constraints, singular positive-semidefinite matrices, small feasible margins, and cone-apex solutions. Rescaling a well-conditioned model tests equilibration. Nearly dependent constraints and narrow feasible sets test intrinsic difficulty that diagonal scaling cannot remove. Include feasible and infeasible instances with independently known outcomes, and label weakly infeasible conic examples separately.

Validation should cover the following distinct questions:

- Do original-model residuals, objectives, multipliers, and complementarity agree with a trusted reference at comparable tolerances?
- Do dense, general sparse, and multistage backends produce matching Newton directions for the same assembled system? Do symbolic patterns remain valid when zero initial gradients become nonzero?
- Do NT identities, Jordan operations, interior repair, and boundary steps hold across cone sizes and extreme scales? Does rotated-cone conversion preserve membership and primal-dual pairings?
- Do numerical updates match a fresh setup of the updated model, including stored zeros, scale reuse, and returned original-coordinate values?
- Does the proposed initializer avoid excessive iterations or failures compared with simple interior starts and any later surrogate initializer?
- Do QP-only cases retain existing results and acceptable setup and solve overhead?

Compare native quadratic constraints with their conic reformulations and include matrix-factorization cost, added cone dimensions, and sparse fill. Clarabel and QOCO can retain the quadratic objective. ECOS requires an objective epigraph for a nonzero quadratic objective, so its transformed setup and solve costs must be recorded. A general SOCP is not a valid native convex-QCQP comparator merely after squaring its norms.

Record setup, updates, solve time, factorization time, memory, iteration count, failures, and achieved original-coordinate residuals separately. Pin solver versions, hardware, precision, thread counts, and tolerances. Report distributions and failure rates rather than timing only successful easy cases. Parameter sequences can assess repeated numerical updates now and true warm starts if that feature is later implemented. No performance claim for this extension is established until these measurements exist.

## Implementation reference map

These references identify where to begin each derivation or code investigation. Repository links refer to the branches inspected for this note, not immutable revisions. The implementation plan should pin revisions. Source files are engineering references; the equations still need reconciliation with PIQP's signs, regularization, and scaling conventions.

| Component | PIQP starting point | External reference and purpose |
| --- | --- | --- |
| Native QCQP residuals and Newton assembly | `solver.tpp`, `kkt_system.tpp`, model and data types | [HPIPM QCQP paper, Sections III-A and III-B](https://publications.syscop.de/Frison2022.pdf), plus its [Python example](https://github.com/giaf/hpipm/blob/master/examples/python/example_qcqp_getting_started.py), for Jacobian and Lagrangian-Hessian updates and stage data. |
| Auxiliary initialization | `solve_impl` before the main iteration | [PIQP paper, Section IV-A](https://arxiv.org/pdf/2304.00290), [QOCO paper, Section 2.4](https://arxiv.org/abs/2503.12658), QOCO [`initialize_ipm`](https://github.com/qoco-org/qoco/blob/main/src/kkt.c), and ECOS [`init`](https://github.com/embotech/ecos/blob/develop/src/ecos.c). |
| Cone interior repair | Scalar slack and multiplier shifts in `solver.tpp` | [`bring2cone` in ECOS](https://github.com/embotech/ecos/blob/develop/src/cone.c) and [QOCO](https://github.com/qoco-org/qoco/blob/main/src/cone.c) for cone-aware shifts. |
| Jordan algebra, NT scaling, and corrector | New cone operations and existing predictor-corrector calculations | [CVXOPT report, Sections 2–5](https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf); QOCO [`soc_product`, `soc_division`, `compute_nt_scaling`, `nt_multiply`, and `nt_multiply_inv`](https://github.com/qoco-org/qoco/blob/main/src/cone.c); Clarabel [`update_scaling`, `mul_W`, `mul_Winv`, and `combined_ds_shift`](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/socone.rs). |
| Cone step lengths | `calculate_step` | [CVXOPT report, Section 8](https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf) and Clarabel [`step_length` and `_step_length_soc_component`](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/socone.rs). |
| Global data equilibration | Dense and sparse `RuizEquilibration::scale_data` and unscaling helpers | [PIQP paper, Section IV-D](https://arxiv.org/pdf/2304.00290), Clarabel [`equilibrate`](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/implementations/default/problemdata.rs) and [`rectify_equilibration`](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/src/solver/core/cones/socone.rs), and [ECOS `equil.c`](https://github.com/embotech/ecos/blob/develop/src/equil.c). |
| Cone blocks and sparse storage | `kkt_solver_base.hpp`, `extract_arrow_structure`, `construct_kkt_fac` | [ECOS paper, Sections II and III](https://web.stanford.edu/~boyd/papers/pdf/ecos_ecc.pdf) for structured scaling and sparse expansions; QOCO [`construct_kkt`](https://github.com/qoco-org/qoco/blob/main/src/kkt.c) for allocating cone entries even when the initial values are zero. |
| Regularization and iterative refinement | Factorization retry logic and backend solves | [QOCO paper, Section 3](https://arxiv.org/abs/2503.12658), [ECOS paper](https://web.stanford.edu/~boyd/papers/pdf/ecos_ecc.pdf), and [Clarabel paper, implementation section](https://arxiv.org/abs/2405.12762) for numerical approaches with different algorithmic contexts. |
| Public cone data and updates | Existing setup, update, and result bindings | [QOCO C API](https://qoco-org.github.io/qoco/qoco/api/C.html), [Clarabel Python setup](https://clarabel.org/stable/python/getting_started_py/), and [Clarabel numerical updates](https://clarabel.org/stable/user_guide_data_updating/). |

QOCO's paper and source are especially useful for connecting the cone formulas to a quadratic-objective implementation. ECOS explains sparse treatment of cone scaling and embedded numerical choices. Clarabel provides a broader cone abstraction and a quadratic-objective homogeneous formulation. HPIPM is the main reference for native QCQP and control structure. None implements precisely the mixed proximal method proposed here.

## Decisions for the future implementation plan

The preferred direction is native convex quadratic constraints first, followed by affine ordinary and rotated cones, with shared Newton factorization machinery. Keep the existing QP interface and model semantics, represent new constraints as local-indexed blocks, and preserve automatic multistage detection. Initialization remains from scratch. Global equilibration is part of the extension, and full convergence or infeasibility-certificate claims are not assumed.

The detailed plan still needs to settle the nonlinear QCQP corrector and safeguards, the conic proximal residuals and NT convention, complementarity normalization, and original-coordinate termination rules. It also needs a curvature-aware scale-selection rule, initial slack-dual centering rules, backend-specific cone elimination choices, and exact update and ownership types. A mandatory convexity check, an explicit stage partition, a homogeneous embedding, and true warm starts remain separate decisions rather than initial requirements.

The smallest useful implementation sequence would be:

1. Derive the regularized native-QCQP residuals, Newton system, and predictor-corrector rule. Implement a dense solver path and a terminal-ellipsoid example, including initialization and scaling of quadratic data. Compare primal solutions, original-model residuals, and multipliers with a conic reformulation.
2. Add sparse symbolic support and multistage assembly for native QCQPs. Validate zero initial gradients, adjacent-stage constraints, and global variables. Compare Newton directions with a full reference system and measure horizon scaling on an actual control problem.
3. Implement ordinary cone operations and a general solver path with a quadratic objective, cone-aware initialization, and equilibration. Validate interior steps, scaling identities, cone-apex cases, and mixed affine and cone constraints against Clarabel, QOCO, or ECOS on matching formulations.
4. Add cone-aware multistage assembly, rotated-cone conversion, and mixed quadratic-conic problems. Test whole-cone support detection and confirm the complete solver's residuals and numerical-update behavior.
5. Before release, settle convergence claims and infeasibility-status semantics, broaden degenerate-case coverage, and complete language bindings. Benchmark setup, repeated numerical updates, solve time, memory, and iteration count separately. Include QP-only regressions to quantify the cost of the extension.

Each phase should produce a usable solver and example before the next expansion. Expose C++ first, then Python for experiments, followed by C and the MATLAB and Octave bindings. Coordinate R separately. Bindings can progress alongside numerical work once their native data and result contracts are stable.

After the mathematical and data contracts are settled, cone-operation development can proceed independently of sparse QCQP assembly. Mixed solver integration depends on both. I would begin with native QCQP support and a multistage terminal-set problem, while designing the constraint descriptors to admit cones from the outset.
