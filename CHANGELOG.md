# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed

- Removed unnecessary verion pins #35
- Changed code to be compatible with numpy 2.0 #35

### Fixed

- Add requests dependency to the non-extra package because it is imported by
`schema_utils_functions.py` but is only installed with the extra fiftyone #22

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
