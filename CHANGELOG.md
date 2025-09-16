# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.6.0...v0.6.2
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