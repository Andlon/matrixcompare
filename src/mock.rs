use crate::{Accessor, DenseMatrix, Matrix};

#[derive(Clone, Debug)]
pub struct MockDenseMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
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

impl<T: Clone> DenseMatrix<T> for MockDenseMatrix<T> {
    fn get(&self, row: usize, col: usize) -> T {
        let idx = row * self.cols + col;
        self.data[idx].clone()
    }
}
