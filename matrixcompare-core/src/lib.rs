/// Defines how the elements of a matrix may be accessed.
pub enum Access<'a, T> {
    Dense(&'a dyn DenseAccess<T>),
    Sparse(&'a dyn SparseAccess<T>),
}

pub trait Matrix<T> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;

    fn access(&self) -> Access<T>;
}

pub trait DenseAccess<T>: Matrix<T> {
    fn fetch_single(&self, row: usize, col: usize) -> T;
}

pub trait SparseAccess<T>: Matrix<T> {
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

    fn access(&self) -> Access<T> {
        X::access(*self)
    }
}

impl<T, X> DenseAccess<T> for &X
    where
        X: DenseAccess<T>,
{
    fn fetch_single(&self, row: usize, col: usize) -> T {
        X::fetch_single(*self, row, col)
    }
}

impl<T, X> SparseAccess<T> for &X
    where
        X: SparseAccess<T>,
{
    fn nnz(&self) -> usize {
        X::nnz(*self)
    }

    fn fetch_triplets(&self) -> Vec<(usize, usize, T)> {
        X::fetch_triplets(&self)
    }
}