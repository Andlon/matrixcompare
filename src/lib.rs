#[macro_use]
pub mod assert_matrix_eq;
//
//#[macro_use]
//mod assert_vector_eq;
//
#[macro_use]
mod assert_scalar_eq;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

pub mod comparison;
pub mod ulp;

#[cfg(test)]
pub mod mock;

pub use self::comparison::{
    AbsoluteElementwiseComparator,
    ExactElementwiseComparator,
    UlpElementwiseComparator,
    FloatElementwiseComparator,

    // The following is just imported because we want to
    // expose trait bounds in the documentation
    ElementwiseComparator
};

pub use self::assert_matrix_eq::elementwise_matrix_comparison;
//pub use self::assert_vector_eq::elementwise_vector_comparison;
pub use self::assert_scalar_eq::scalar_comparison;

pub trait Matrix<T> {
    fn get(&self, row: usize, col: usize) -> T;

    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}

pub trait SparseMatrix<T> {
    fn nnz(&self) -> usize;

    fn get_triplet(&self, index: usize) -> (usize, usize, T);
}

impl<T, X> Matrix<T> for &X
    where X: Matrix<T>
{
    fn get(&self, row: usize, col: usize) -> T {
        X::get(*self, row, col)
    }

    fn rows(&self) -> usize {
        X::rows(*self)
    }

    fn cols(&self) -> usize {
        X::cols(*self)
    }
}
