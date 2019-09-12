//! Tools for comparing matrices for debugging purposes.
//!
//! TODO: Docs

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
pub use self::scalar_comparison::{
    compare_scalars, ScalarComparisonFailure, ScalarComparisonResult,
};

pub use self::comparison_failure::{
    DimensionMismatch, DuplicateEntries, ElementsMismatch, MatrixComparisonFailure,
    MatrixElementComparisonFailure, OutOfBoundsIndices,
};

pub use matrixcompare_core::*;