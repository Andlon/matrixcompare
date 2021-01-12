/*! Tools for comparing matrices for debugging purposes.

When testing and debugging linear algebra code, it is often necessary to compare two matrices
for equality. Moreover, more often than not, we can not expect exact pair-wise equality of the
numbers contained in the matrices. Most linear algebra libraries provide facilities to compare
matrices for approximate equality, but usually this only gives a binary answer: whether
the matrices are equal or not. This can be frustrating, as it gives no indication of which elements
in the matrix are not approximately equal, and how far away the numbers are from each other.
`matrixcompare` remedies this problem by giving formatted output explaining exactly which entries
in the matrices failed to compare (approximately) equal.

Consider the following contrived example.

```rust,should_panic
# use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
let a = mock_matrix![1.00, 2.00;
                     3.00, 4.00];
let b = mock_matrix![1.01, 2.00;
                     3.40, 4.00];
assert_matrix_eq!(a, b, comp = abs, tol = 1e-8);
```

The above example panics with the following error message

```text
Matrices X and Y have 2 mismatched element pairs.
The mismatched elements are listed below, in the format
(row, col): x = X[[row, col]], y = Y[[row, col]].

 (0, 0): x = 1, y = 1.01. Absolute error: 0.010000000000000009.
 (1, 0): x = 3, y = 3.4. Absolute error: 0.3999999999999999.

Comparison criterion: absolute difference, |x - y| <= 0.00000001.
```

See the documentation for the [assert_matrix_eq!](macro.assert_matrix_eq.html) macro for more
information.

## Design and integration with linear algebra libraries

`matrixcompare` is designed to be easy to integrate with any linear algebra library. In particular:

- The core traits are defined in `matrixcompare-core`. This crate has no dependencies other than
the standard library, and only contains a very small amount of code that defines the interface
through which the rest of `matrixcompare` is able to access the data contained in matrices.
- The `core` split allows the actual comparison logic and output format to evolve separately
from the `core` crate. This way we can minimize breaking changes in `matrixcompare-core` and
hopefully relatively soon stabilize it, without having to stabilize the entire `matrixcompare` crate.
- Linear algebra library authors should only depend on and implement the traits in
`matrixcompare-core`, while end users can use any functionality provided in `matrixcompare`.
- Since access to the underlying structures are abstracted, `matrixcompare` can be used to
compare matrices originating from different linear algebra libraries, provided that the libraries
in question implement the traits found in `matrixcompare-core`.

The design of `matrixcompare` heavily favors ease of use/integration, correctness and
flexibility over performance. It is intended to be used for automated tests, and as such does
not belong in performance sensitive code. There are no particular guarantees about performance,
other than that the asymptotic complexity is roughly the same as a more optimized implementation.

## `proptest` integration

`proptest` ships its own macros for use with its tests. Although it's possible to directly
use `assert_matrix_eq!` from `matrixcompare` in proptests, every failing test will result in a
panic message being written to the error output, which causes unnecessary noise when debugging
a failing test. To overcome this situation, we provide the macro `prop_assert_matrix_eq!`, which
works exactly as `assert_matrix_eq!`, except that instead of panicing, it returns errors compatible
with `proptest`.

To use this feature, the `proptest-support` feature must be enabled.

*/

#![allow(clippy::float_cmp)]

#[macro_use]
mod matrix_comparison;

#[macro_use]
mod scalar_comparison;

mod comparison_failure;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub mod comparators;
mod macros;
pub mod ulp;

pub use self::matrix_comparison::compare_matrices;
pub use self::scalar_comparison::{compare_scalars, ScalarComparisonFailure};

pub use self::comparison_failure::{
    Coordinate, DimensionMismatch, ElementsMismatch, Entry, MatrixComparisonFailure,
    MatrixElementComparisonFailure,
};

pub use matrixcompare_core::*;

#[cfg(feature = "proptest-support")]
mod proptest;