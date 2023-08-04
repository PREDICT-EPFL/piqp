# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.2...HEAD
[0.2.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/PREDICT-EPFL/piqp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/PREDICT-EPFL/piqp/releases/tag/v0.1.0