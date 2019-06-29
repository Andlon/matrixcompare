use crate::{Accessor, DenseAccessor, Matrix, SparseAccessor};

#[derive(Clone, Debug)]
pub struct MockDenseMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
pub struct MockSparseMatrix<T> {
    shape: (usize, usize),
    triplets: Vec<(usize, usize, T)>
}

impl<T> MockSparseMatrix<T> {
    pub fn from_triplets(rows: usize, cols: usize, triplets: Vec<(usize, usize, T)>) -> Self {
        Self {
            shape: (rows, cols),
            triplets
        }
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
}

impl<T: Clone> Matrix<T> for MockDenseMatrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn access(&self) -> Accessor<T> {
        Accessor::Dense(self)
    }
}

impl<T: Clone> DenseAccessor<T> for MockDenseMatrix<T> {
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

    fn access(&self) -> Accessor<T> {
        Accessor::Sparse(self)
    }
}

impl<T: Clone> SparseAccessor<T> for MockSparseMatrix<T> {
    fn nnz(&self) -> usize {
        self.triplets.len()
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        self.triplets.clone()
    }
}
