# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Use networkx for chunk division for dataset splitting #71
- Make chunk division compatible with NaN values # 71

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

For previous changelog entries, before github migration, click [here](docs/changelog_old.md)
