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
    // The following is just imported because we want to
    // expose trait bounds in the documentation
    ElementwiseComparator,
    ExactElementwiseComparator,
    FloatElementwiseComparator,

    UlpElementwiseComparator,
};

pub use self::assert_matrix_eq::elementwise_matrix_comparison;
//pub use self::assert_vector_eq::elementwise_vector_comparison;
pub use self::assert_scalar_eq::scalar_comparison;

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

pub trait SparseAccessor<T> {
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
