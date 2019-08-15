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

pub use self::matrix_comparison::{
    compare_matrices
};
pub use self::scalar_comparison::{
    compare_scalars, ScalarComparisonFailure, ScalarComparisonResult,
};

pub use self::comparison_failure::{DimensionMismatch, ElementsMismatch, MatrixComparisonFailure,
                                   MatrixElementComparisonFailure, OutOfBoundsIndices, DuplicateEntries};

pub enum Accessor<'a, T> {
    Dense(&'a dyn DenseAccessor<T>),
    Sparse(&'a dyn SparseAccessor<T>),
}

pub trait Matrix<T> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn access(&self) -> Accessor<T>;
}

pub trait DenseAccessor<T>: Matrix<T> {
    fn fetch_single(&self, row: usize, col: usize) -> T;
}

pub trait SparseAccessor<T>: Matrix<T> {
    fn nnz(&self) -> usize;
    fn fetch_triplets(&self) -> Vec<(usize, usize, T)>;
}

impl<T, X> Matrix<T> for &X
where
    X: Matrix<T>,
{
    fn rows(&self) -> usize {
        X::rows(*self)
    }

    fn cols(&self) -> usize {
        X::cols(*self)
    }

    fn access(&self) -> Accessor<T> {
        X::access(*self)
    }
}

impl<T, X> DenseAccessor<T> for &X
where
    X: DenseAccessor<T>,
{
    fn fetch_single(&self, row: usize, col: usize) -> T {
        X::fetch_single(*self, row, col)
    }
}

impl<T, X> SparseAccessor<T> for &X
where
    X: SparseAccessor<T>,
{
    fn nnz(&self) -> usize {
        X::nnz(*self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        X::fetch_triplets(&self)
    }
}
