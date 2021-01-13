# Changelog

Prominent changes for individual releases will be listed here.

## 0.1.4 - 2020-01-13

### Added

- [proptest](https://crates.io/crates/proptest) support through new macros `prop_assert_matrix_eq!`
  and `prop_assert_scalar_eq!`. The `proptest-support` feature must be enabled to use this new
  functionality.

### Changed

- Very minor changes to formatting for output of some assertions.

## 0.1.3

- Avoid pulling in default features for num-traits dependency (thanks to Sébastien Crozet!).

## 0.1.0 - 0.1.2

Initial releases.
