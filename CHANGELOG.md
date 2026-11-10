# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New method `backward()` on the dense and sparse solvers which computes the gradients of a loss with respect to the problem data (P, c, A, b, G, h_l, h_u, x_l, x_u) by implicit differentiation of the KKT conditions at the solution, given the gradients with respect to the solution (x, y, z, s). By passing a stored `forward_result`, the gradients can be computed at the result of an earlier solve, e.g. for rollouts. Available in the C++, C (`piqp_backward()`), Python, Matlab and Octave interfaces. See the new [Differentiation](https://predict-epfl.github.io/piqp/differentiation) page in the documentation
- `backward()` returns the new `BackwardStatus` (`PIQP_BWD_SOLVED`, `PIQP_BWD_INVALID_SETTINGS`, `PIQP_BWD_INVALID_FORWARD_RESULT`, `PIQP_BWD_NUMERICS`, `PIQP_BWD_MAX_ITER_REACHED`), the forward `Status` is only used for `solve()`
- The gradients are available via `gradients()` and must only be used if `gradients_valid()` is true. A failed `backward()`, `setup()` or `update()` invalidates them
- New setting `enable_backward`, which has to be set before `setup()` to use `backward()`. It allocates the gradient storage in `setup()`, such that `backward()` does not allocate
- New settings `gradient_smoothing`, `gradient_smoothing_mu`, `gradient_smoothing_tol` and `gradient_smoothing_max_iter`: with `gradient_smoothing` enabled, `backward()` differentiates at a relaxed point with `s * z = mu`, which makes the gradient smooth across active-set changes

### Changed
- The C settings struct `piqp_settings` and the C workspace `piqp_workspace` (new field `gradients`) have new fields, so code compiled against the C interface must be recompiled

## [0.7.0] - 2026-11-05

### Added
- New KKT solver backend `sparse_multistage_parallel`, which factorizes multistage KKT systems in parallel using OpenMP and Blasfeo [#46](https://github.com/PREDICT-EPFL/piqp/pull/46)
- New setting `num_threads` to override the default number of OpenMP threads [#46](https://github.com/PREDICT-EPFL/piqp/pull/46)
- New settings `eps_primal_inf` and `eps_dual_inf` for the tolerances of the certificate-based infeasibility detection [#50](https://github.com/PREDICT-EPFL/piqp/pull/50)

### Changed
- On primal or dual infeasibility, the result now contains a normalized certificate of infeasibility in the dual or primal variables respectively [#50](https://github.com/PREDICT-EPFL/piqp/pull/50)
- Replace the heuristic infeasibility detection with certificate-based detection on the proximal steps. `infeasibility_threshold` now only guards the regularization updates [#50](https://github.com/PREDICT-EPFL/piqp/pull/50)
- The C settings struct `piqp_settings` has new fields, so code compiled against the C interface must be recompiled
- Building with `BUILD_WITH_OPENMP=ON` now fails if OpenMP cannot be found [#46](https://github.com/PREDICT-EPFL/piqp/pull/46)
- The Blasfeo backends now recover from failed factorizations [#46](https://github.com/PREDICT-EPFL/piqp/pull/46)
- All memory in the dense KKT solver is now allocated during setup [#45](https://github.com/PREDICT-EPFL/piqp/pull/45)

### Fixed
- Fix false infeasibility detection in some feasible problems [#37](https://github.com/PREDICT-EPFL/piqp/issues/37) [#49](https://github.com/PREDICT-EPFL/piqp/issues/49)
- Fix stale KKT matrices after partial matrix updates that recompute the scaling, which could lead to wrong solutions or false infeasibility [#48](https://github.com/PREDICT-EPFL/piqp/pull/48)
- Average the cost scaling over the populated columns of P and fix aliasing between cost scaling and box scaling scratch storage [#43](https://github.com/PREDICT-EPFL/piqp/pull/43)
- Fix UBSan invalid enum load in dense KKT constructor and flush output in the R interface [#45](https://github.com/PREDICT-EPFL/piqp/pull/45)
- Install `piqp.dll` and `piqpc.dll` on Windows [#44](https://github.com/PREDICT-EPFL/piqp/pull/44)

## [0.6.4] - 2026-08-28

### Fixed
- Fix dual recovery out-of-bounds memory bug [#42](https://github.com/PREDICT-EPFL/piqp/issues/42)

## [0.6.3] - 2026-05-11

### Changed
- Improve C++ compile times by moving large implementation details into `.tpp` files [#38](https://github.com/PREDICT-EPFL/piqp/pull/38)

### Fixed
- Fix compatibility with Eigen 5 [#41](https://github.com/PREDICT-EPFL/piqp/pull/41)
- Publish Octave release tarballs with versioned filenames [#36](https://github.com/PREDICT-EPFL/piqp/issues/36)

## [0.6.2] - 2025-09-16

### Changed
- Record KKT timings
- Disable stack execution for matlab and python libraries on linux [#35](https://github.com/PREDICT-EPFL/piqp/pull/35)

## [0.6.1] - 2025-08-22

### Changed
- Add tracy support [#30](https://github.com/PREDICT-EPFL/piqp/pull/30)
- Enforce Eigen ABI with Template Instantiation [#32](https://github.com/PREDICT-EPFL/piqp/pull/32)
- Improved and more robust infeasibility detection [#34](https://github.com/PREDICT-EPFL/piqp/pull/34)

## [0.6.0] - 2025-06-30

### Changed
- Fix compatibility with Octave 10 or newer [#20](https://github.com/PREDICT-EPFL/piqp/pull/20)
- Change preconditioner to scale lower and upper bounds using the same scaling and refactor KKT system logic [#21](https://github.com/PREDICT-EPFL/piqp/pull/21)
- Add support for double sided general inequality constraints [#22](https://github.com/PREDICT-EPFL/piqp/pull/22)
- General convergence improvements [#23](https://github.com/PREDICT-EPFL/piqp/pull/23)
- Switch build system to scikit-build-core [#25](https://github.com/PREDICT-EPFL/piqp/pull/25)
- Make building with EIGEN_MAX_ALIGN_BYTES definition opt-in [#28](https://github.com/PREDICT-EPFL/piqp/pull/28)

## [0.5.0] - 2025-03-17

### Changed
- In the sparse interface for Python A, b, G, and h are now None by default
- Use Cholesky decomposition instead of LDLt factorization in dense interface
- New KKT solver backend `sparse_multistage` for multistage optimization problems [#14](https://github.com/PREDICT-EPFL/piqp/pull/14)
- Fix with run_time calculation, i.e., now it's run_time = setup_time/update_time + solve_time and not accumulated over multiple runs.

## [0.4.2] - 2024-08-02

### Changed

- Handle inf constraints in general inequalities by setting corresponding row in G to zero and emitting a warning to the user.

## [0.4.1] - 2024-06-22

### Changed

- Fix installation paths for CMake install.

## [0.4.0] - 2024-06-21

### Changed

- Instead of building both shared and static libraries, it can now be selected by setting `BUILD_SHARED_LIBS`.
- Better support for CMake subdirectory installs.
- Added Octave interface (thanks to @redstone99)
- Added boundary control, i.e., dual inequality variables can't become zero anymore avoiding potential NaNs in the solution.

## [0.3.1] - 2024-05-25

### Changed

- Added python stub files for better python IDE support.

## [0.3.0] - 2024-05-24

### Changed

- Make equality and inequality constrains optional on setup.
- Pre-compile common template instantiations in C++ interface to speed up compilation times.
- Add utilities to save and load problem data.
- Various bug fixes.
- Various doc improvements.

## [0.2.4] - 2023-12-25

### Changed

- Fixed issue where regularization parameters were not correctly reset after an update.

## [0.2.3] - 2023-09-14

### Changed

- Allow compilation with custom scalar types like mpreal.
- Disable floating point contractions in sparse ldlt.

## [0.2.2] - 2023-08-04

### Changed

- Fixed compilation issues for conda-forge on macOS

## [0.2.1] - 2023-08-03

### Changed

- Renamed C libraries from `piqp` to `piqpc` and `piqpcstatic` to avoid naming conflicts (especially on Windows).
- Pull external packages through FetchContent instead of git submodules.

## [0.2.0] - 2023-07-28

### Changed

- Added Matlab interface.
- The backend is now printed in verbose mode.

## [0.1.3] - 2023-07-20

### Changed

- Changed print function for potential Matlab and R interfaces.

## [0.1.2] - 2023-07-14

### Changed

- Fixed PyPi upload.
- Fixed incorrect type in settings.
- Added runtime dimension checks in release mode.
- Convert infinite inequality bounds to a finite value internally (-inf -> -1e30, inf -> 1e30).

## [0.1.1] - 2023-06-30

### Changed

- Fixed compilation issues on Windows.
- Removed French accents in verbose mode prints which caused issues in certain consoles, i.e. Windows.
- Exposed fine tune thresholds in settings.
- Fine-tuned default settings for better numerical robustness.

## [0.1.0] - 2023-06-28

Initial release of PIQP!

[unreleased]: https://github.com/PREDICT-EPFL/piqp/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.4...v0.7.0
[0.6.4]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.3...v0.6.4
[0.6.3]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PREDICT-EPFL/piqp/releases/tag/v0.1.0
