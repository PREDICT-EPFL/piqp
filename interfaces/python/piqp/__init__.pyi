# Generated using pybind11-stubgen 2.5.1
from __future__ import annotations
import numpy
import piqp
import scipy.sparse
import typing
__all__ = ['DenseSolver', 'Info', 'PIQP_DUAL_INFEASIBLE', 'PIQP_INVALID_SETTINGS', 'PIQP_MAX_ITER_REACHED', 'PIQP_NUMERICS', 'PIQP_PRIMAL_INFEASIBLE', 'PIQP_SOLVED', 'PIQP_UNSOLVED', 'Result', 'Settings', 'SparseSolver', 'Status']
class DenseSolver:
    def __init__(self: piqp.DenseSolver) -> None:
        ...
    def setup(self: piqp.DenseSolver, P: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous], c: numpy.ndarray[numpy.float64[m, 1]], A: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous] | None = None, b: numpy.ndarray[numpy.float64[m, 1]] | None = None, G: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous] | None = None, h: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_lb: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_ub: numpy.ndarray[numpy.float64[m, 1]] | None = None) -> None:
        ...
    def solve(self: piqp.DenseSolver) -> piqp.Status:
        ...
    def update(self: piqp.DenseSolver, P: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous] | None = None, c: numpy.ndarray[numpy.float64[m, 1]] | None = None, A: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous] | None = None, b: numpy.ndarray[numpy.float64[m, 1]] | None = None, G: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.f_contiguous] | None = None, h: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_lb: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_ub: numpy.ndarray[numpy.float64[m, 1]] | None = None, reuse_preconditioner: bool = True) -> None:
        ...
    @property
    def result(self) -> piqp.Result:
        ...
    @property
    def settings(self) -> piqp.Settings:
        ...
    @settings.setter
    def settings(self) -> piqp.Settings:
        ...
class Info:
    delta: float
    dual_inf: float
    dual_obj: float
    dual_rel_inf: float
    dual_step: float
    duality_gap: float
    duality_gap_rel: float
    factor_retires: int
    iter: int
    mu: float
    no_dual_update: int
    no_primal_update: int
    primal_inf: float
    primal_obj: float
    primal_rel_inf: float
    primal_step: float
    reg_limit: float
    rho: float
    run_time: float
    setup_time: float
    sigma: float
    solve_time: float
    status: piqp.Status
    update_time: float
    def __init__(self: piqp.Info) -> None:
        ...
class Result:
    info: piqp.Info
    lambda: numpy.ndarray[numpy.float64[m, 1]]
    nu: numpy.ndarray[numpy.float64[m, 1]]
    nu_lb: numpy.ndarray[numpy.float64[m, 1]]
    nu_ub: numpy.ndarray[numpy.float64[m, 1]]
    s: numpy.ndarray[numpy.float64[m, 1]]
    s_lb: numpy.ndarray[numpy.float64[m, 1]]
    s_ub: numpy.ndarray[numpy.float64[m, 1]]
    x: numpy.ndarray[numpy.float64[m, 1]]
    y: numpy.ndarray[numpy.float64[m, 1]]
    z: numpy.ndarray[numpy.float64[m, 1]]
    z_lb: numpy.ndarray[numpy.float64[m, 1]]
    z_ub: numpy.ndarray[numpy.float64[m, 1]]
    zeta: numpy.ndarray[numpy.float64[m, 1]]
class Settings:
    check_duality_gap: bool
    compute_timings: bool
    delta_init: float
    eps_abs: float
    eps_duality_gap_abs: float
    eps_duality_gap_rel: float
    eps_rel: float
    iterative_refinement_always_enabled: bool
    iterative_refinement_eps_abs: float
    iterative_refinement_eps_rel: float
    iterative_refinement_max_iter: int
    iterative_refinement_min_improvement_rate: float
    iterative_refinement_static_regularization_eps: float
    iterative_refinement_static_regularization_rel: float
    max_factor_retires: int
    max_iter: int
    preconditioner_iter: int
    preconditioner_scale_cost: bool
    reg_finetune_dual_update_threshold: int
    reg_finetune_lower_limit: float
    reg_finetune_primal_update_threshold: int
    reg_lower_limit: float
    rho_init: float
    tau: float
    verbose: bool
class SparseSolver:
    def __init__(self: piqp.SparseSolver) -> None:
        ...
    def setup(self: piqp.SparseSolver, P: scipy.sparse.csc_matrix, c: numpy.ndarray[numpy.float64[m, 1]], A: scipy.sparse.csc_matrix | None, b: numpy.ndarray[numpy.float64[m, 1]] | None, G: scipy.sparse.csc_matrix | None, h: numpy.ndarray[numpy.float64[m, 1]] | None, x_lb: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_ub: numpy.ndarray[numpy.float64[m, 1]] | None = None) -> None:
        ...
    def solve(self: piqp.SparseSolver) -> piqp.Status:
        ...
    def update(self: piqp.SparseSolver, P: scipy.sparse.csc_matrix | None = None, c: numpy.ndarray[numpy.float64[m, 1]] | None = None, A: scipy.sparse.csc_matrix | None = None, b: numpy.ndarray[numpy.float64[m, 1]] | None = None, G: scipy.sparse.csc_matrix | None = None, h: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_lb: numpy.ndarray[numpy.float64[m, 1]] | None = None, x_ub: numpy.ndarray[numpy.float64[m, 1]] | None = None, reuse_preconditioner: bool = True) -> None:
        ...
    @property
    def result(self) -> piqp.Result:
        ...
    @property
    def settings(self) -> piqp.Settings:
        ...
    @settings.setter
    def settings(self) -> piqp.Settings:
        ...
class Status:
    """
    Members:
    
      PIQP_SOLVED
    
      PIQP_MAX_ITER_REACHED
    
      PIQP_PRIMAL_INFEASIBLE
    
      PIQP_DUAL_INFEASIBLE
    
      PIQP_NUMERICS
    
      PIQP_UNSOLVED
    
      PIQP_INVALID_SETTINGS
    """
    PIQP_DUAL_INFEASIBLE: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_DUAL_INFEASIBLE: -3>
    PIQP_INVALID_SETTINGS: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_INVALID_SETTINGS: -10>
    PIQP_MAX_ITER_REACHED: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_MAX_ITER_REACHED: -1>
    PIQP_NUMERICS: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_NUMERICS: -8>
    PIQP_PRIMAL_INFEASIBLE: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_PRIMAL_INFEASIBLE: -2>
    PIQP_SOLVED: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_SOLVED: 1>
    PIQP_UNSOLVED: typing.ClassVar[piqp.Status]  # value = <Status.PIQP_UNSOLVED: -9>
    __members__: typing.ClassVar[dict[str, piqp.Status]]  # value = {'PIQP_SOLVED': <Status.PIQP_SOLVED: 1>, 'PIQP_MAX_ITER_REACHED': <Status.PIQP_MAX_ITER_REACHED: -1>, 'PIQP_PRIMAL_INFEASIBLE': <Status.PIQP_PRIMAL_INFEASIBLE: -2>, 'PIQP_DUAL_INFEASIBLE': <Status.PIQP_DUAL_INFEASIBLE: -3>, 'PIQP_NUMERICS': <Status.PIQP_NUMERICS: -8>, 'PIQP_UNSOLVED': <Status.PIQP_UNSOLVED: -9>, 'PIQP_INVALID_SETTINGS': <Status.PIQP_INVALID_SETTINGS: -10>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self: piqp.Status) -> int:
        ...
    def __init__(self: piqp.Status, value: int) -> None:
        ...
    def __int__(self: piqp.Status) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self: piqp.Status, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
PIQP_DUAL_INFEASIBLE: piqp.Status  # value = <Status.PIQP_DUAL_INFEASIBLE: -3>
PIQP_INVALID_SETTINGS: piqp.Status  # value = <Status.PIQP_INVALID_SETTINGS: -10>
PIQP_MAX_ITER_REACHED: piqp.Status  # value = <Status.PIQP_MAX_ITER_REACHED: -1>
PIQP_NUMERICS: piqp.Status  # value = <Status.PIQP_NUMERICS: -8>
PIQP_PRIMAL_INFEASIBLE: piqp.Status  # value = <Status.PIQP_PRIMAL_INFEASIBLE: -2>
PIQP_SOLVED: piqp.Status  # value = <Status.PIQP_SOLVED: 1>
PIQP_UNSOLVED: piqp.Status  # value = <Status.PIQP_UNSOLVED: -9>
__version__: str = '0.4.1'
