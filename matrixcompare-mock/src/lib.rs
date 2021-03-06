//! Utility data structures and functionality for testing the
//! `matrixcompare` crate. Not intended for usage outside of
//! `matrixcompare` tests.

use matrixcompare_core::{Access, DenseAccess, Matrix, SparseAccess};
use proptest::prelude::*;
use std::fmt::Debug;

use num::Zero;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct MockDenseMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
pub struct MockSparseMatrix<T> {
    shape: (usize, usize),
    triplets: Vec<(usize, usize, T)>,
}

impl<T> MockSparseMatrix<T> {
    pub fn from_triplets(rows: usize, cols: usize, triplets: Vec<(usize, usize, T)>) -> Self {
        Self {
            shape: (rows, cols),
            triplets,
        }
    }

    pub fn take_triplets(self) -> Vec<(usize, usize, T)> {
        self.triplets
    }
}

impl<T> MockSparseMatrix<T>
where
    T: Zero + Clone,
{
    pub fn to_dense(&self) -> Result<MockDenseMatrix<T>, ()> {
        let (r, c) = (self.rows(), self.cols());
        let mut result =
            MockDenseMatrix::from_row_major(self.rows(), self.cols(), vec![T::zero(); r * c]);
        for (i, j, v) in &self.triplets {
            *result.get_mut(*i, *j).ok_or(())? = v.clone();
        }

        Ok(result)
    }
}

impl<T> MockDenseMatrix<T> {
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(
            rows * cols,
            data.len(),
            "Data must have rows*cols number of elements."
        );
        Self { data, rows, cols }
    }

    fn get_linear_index(&self, i: usize, j: usize) -> Option<usize> {
        if i < self.rows && j < self.cols {
            Some(i * self.cols + j)
        } else {
            None
        }
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&T> {
        self.get_linear_index(i, j).map(|idx| &self.data[idx])
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut T> {
        self.get_linear_index(i, j)
            .map(move |idx| &mut self.data[idx])
    }
}

impl<T: Clone> Matrix<T> for MockDenseMatrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn access(&self) -> Access<T> {
        Access::Dense(self)
    }
}

impl<T: Clone> DenseAccess<T> for MockDenseMatrix<T> {
    fn fetch_single(&self, row: usize, col: usize) -> T {
        let idx = row * self.cols + col;
        self.data[idx].clone()
    }
}

impl<T: Clone> Matrix<T> for MockSparseMatrix<T> {
    fn rows(&self) -> usize {
        self.shape.0
    }

    fn cols(&self) -> usize {
        self.shape.1
    }

    fn access(&self) -> Access<T> {
        Access::Sparse(self)
    }
}

impl<T: Clone> SparseAccess<T> for MockSparseMatrix<T> {
    fn nnz(&self) -> usize {
        self.triplets.len()
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplets.clone()
    }
}

/// Macro that helps with the construction of small dense (mock) matrices for testing.
///
/// Originally lifted from the `rulinalg` crate (author being the same as for this crate).
#[macro_export]
macro_rules! mock_matrix {
    () => {
        {
            // Handle the case when called with no arguments, i.e. matrix![]
            use $crate::MockDenseMatrix;
            MockDenseMatrix::from_row_major(0, 0, vec![])
        }
    };
    ($( $( $x: expr ),*);*) => {
        {
            use $crate::MockDenseMatrix;
            let data_as_nested_array = [ $( [ $($x),* ] ),* ];
            let rows = data_as_nested_array.len();
            let cols = data_as_nested_array[0].len();
            let data_as_flat_array: Vec<_> = data_as_nested_array.iter()
                .flat_map(|row| row.into_iter())
                .cloned()
                .collect();
            MockDenseMatrix::from_row_major(rows, cols, data_as_flat_array)
        }
    }
}

pub fn i64_range() -> Range<i64> {
    -100i64 .. 100
}

pub fn dense_matrix_strategy<T, S>(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
    strategy: S,
) -> impl Strategy<Value = MockDenseMatrix<T>>
where
    T: Debug,
    S: Clone + Strategy<Value = T>,
{
    (rows, cols).prop_flat_map(move |(r, c)| {
        proptest::collection::vec(strategy.clone(), r * c)
            .prop_map(move |data| MockDenseMatrix::from_row_major(r, c, data))
    })
}

pub fn dense_matrix_strategy_i64(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
) -> impl Strategy<Value = MockDenseMatrix<i64>> {
    dense_matrix_strategy(rows, cols, i64_range())
}

/// A strategy for "normal" f64 numbers (excluding infinities, NaN).
pub fn dense_matrix_strategy_normal_f64(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
) -> impl Strategy<Value = MockDenseMatrix<f64>> {
    dense_matrix_strategy(rows, cols, proptest::num::f64::NORMAL)
}

pub fn sparse_matrix_strategy<T, S>(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
    strategy: S,
) -> impl Strategy<Value = MockSparseMatrix<T>>
where
    T: Debug,
    S: Clone + Strategy<Value = T>,
{
    // Generate sparse matrices by generating maps whose keys (ij entries) are in bounds
    // and values are picked from the supplied strategy
    (rows, cols).prop_flat_map(move |(r, c)| {
        let max_nnz = r * c;
        let ij_strategy = (0..r, 0..c);
        let values_strategy = strategy.clone();
        // Use BTreeMap to avoid potential randomness in hash map iteration order
        proptest::collection::btree_map(ij_strategy, values_strategy, 0..=max_nnz)
            .prop_map(|map_matrix| map_matrix
                .into_iter()
                .map(|((i, j), v)| (i, j, v))
                .collect())
            .prop_map(move |triplets| MockSparseMatrix::from_triplets(r, c, triplets))
    })
}

pub fn sparse_matrix_strategy_i64(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
) -> impl Strategy<Value = MockSparseMatrix<i64>> {
    sparse_matrix_strategy(rows, cols, i64_range())
}

pub fn sparse_matrix_strategy_normal_f64(
    rows: impl Strategy<Value = usize>,
    cols: impl Strategy<Value = usize>,
) -> impl Strategy<Value = MockSparseMatrix<f64>> {
    sparse_matrix_strategy(rows, cols, proptest::num::f64::NORMAL)
}