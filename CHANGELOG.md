# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [1.2.1] - 2025-03-19

### Fixed

- Do not import `pytest_regressions`package when pytest discovers plugin. Only do it at run time #76

## [1.2.0] - 2025-03-19

### Fixed

- Fix a bug in splitting were already assigned split values were discarded #74

### Added

- Add taking into account already assigned values when doing simple split #74

## [1.1.0] - 2025-03-13

### Changed

- Use networkx for chunk division for dataset splitting #71
- Make chunk division compatible with NaN values #71

## [1.0.1] - 2025-03-12

### Added

- Added python 3.13 compatibility #69

### Changed

- Removed unnecessary verion pins #35
- Changed code to be compatible with numpy 2.0 #35

### Fixed

- Add requests dependency to the non-extra package because it is imported by
`schema_utils_functions.py` but is only installed with the extra fiftyone #22
- Relaxed `pandas` and `pyarrow` version dependecies. #67

## [1.0.0] - 2024-09-24

### Added

- Github CI, with pytest, codecov, sphinx and pyright #9
- Github CI for publishing release and pypi packages from a tag #10

### Changed

- Move codebase to github
- Move doc to readthedocs.io #6
- Upgrade fiftyone from 0.23 to 0.25 #7

### Fixed

- Fix the fiftyone web app crashing when filtering on attributes columns #7

## Archived changelog

For previous changelog entries, before github migration, got [changelog_old.md](docs/changelog_old.md)
