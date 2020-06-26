use crate::comparators::ElementwiseComparator;
use crate::{Access, DenseAccess, DimensionMismatch, ElementsMismatch, Matrix, MatrixComparisonFailure, MatrixElementComparisonFailure, SparseAccess, Coordinate};
use num::Zero;
use std::collections::{HashMap, HashSet};

use crate::Entry;

enum HashMapBuildError {
    OutOfBoundsCoord(Coordinate),
    DuplicateCoord(Coordinate)
}

fn try_build_sparse_hash_map<T>(
    rows: usize,
    cols: usize,
    triplets: &[(usize, usize, T)],
) -> Result<HashMap<(usize, usize), T>, HashMapBuildError>
    where
        T: Clone,
{
    let mut matrix = HashMap::new();

    for (i, j, v) in triplets.iter().cloned() {
        if i >= rows || j >= cols {
            return Err(HashMapBuildError::OutOfBoundsCoord((i, j)));
        } else if let Some(_) = matrix.insert((i, j), v) {
            return Err(HashMapBuildError::DuplicateCoord((i, j)));
        }
    }

    Ok(matrix)
}

fn compare_sparse_sparse<T, C>(
    x: &dyn SparseAccess<T>,
    y: &dyn SparseAccess<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(x.rows() == y.rows() && x.cols() == y.cols());

    let x_triplets = x.fetch_triplets();
    let y_triplets = y.fetch_triplets();

    let x_hash = try_build_sparse_hash_map(x.rows(), x.cols(), &x_triplets)
        .map_err(|build_error| match build_error {
            HashMapBuildError::OutOfBoundsCoord(coord)
                => MatrixComparisonFailure::SparseEntryOutOfBounds(Entry::Left(coord)),
            HashMapBuildError::DuplicateCoord(coord)
                => MatrixComparisonFailure::DuplicateSparseEntry(Entry::Left(coord))
        })?;

    let y_hash = try_build_sparse_hash_map(y.rows(), y.cols(), &y_triplets)
        .map_err(|build_error| match build_error {
            HashMapBuildError::OutOfBoundsCoord(coord)
            => MatrixComparisonFailure::SparseEntryOutOfBounds(Entry::Right(coord)),
            HashMapBuildError::DuplicateCoord(coord)
            => MatrixComparisonFailure::DuplicateSparseEntry(Entry::Right(coord))
        })?;

    let mut mismatches = Vec::new();
    let x_keys: HashSet<_> = x_hash.keys().collect();
    let y_keys: HashSet<_> = y_hash.keys().collect();
    let zero = T::zero();

    for coord in x_keys.union(&y_keys) {
        let a = x_hash.get(coord).unwrap_or(&zero);
        let b = y_hash.get(coord).unwrap_or(&zero);
        if let Err(error) = comparator.compare(&a, &b) {
            mismatches.push(MatrixElementComparisonFailure {
                x: a.clone(),
                y: b.clone(),
                error,
                row: coord.0,
                col: coord.1,
            });
        }
    }

    // Sorting the mismatches by (i, j) gives us predictable output, independent of e.g.
    // the order we compare the two matrices.
    mismatches.sort_by_key(|mismatch| (mismatch.row, mismatch.col));

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(MatrixComparisonFailure::MismatchedElements(
            ElementsMismatch {
                comparator_description: comparator.description(),
                mismatches,
            },
        ))
    }
}

fn find_dense_sparse_mismatches<T, C>(
    dense: &dyn DenseAccess<T>,
    sparse: &HashMap<(usize, usize), T>,
    comparator: &C,
    swap_order: bool,
) -> Option<ElementsMismatch<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function

    let mut mismatches = Vec::new();
    let zero = T::zero();

    for i in 0..dense.rows() {
        for j in 0..dense.cols() {
            let a = &dense.fetch_single(i, j);
            let b = sparse.get(&(i, j)).unwrap_or(&zero);
            let (a, b) = if swap_order { (b, a) } else { (a, b) };
            if let Err(error) = comparator.compare(a, b) {
                mismatches.push(MatrixElementComparisonFailure {
                    x: a.clone(),
                    y: b.clone(),
                    error,
                    row: i,
                    col: j,
                });
            }
        }
    }

    if mismatches.is_empty() {
        None
    } else {
        Some(ElementsMismatch {
                comparator_description: comparator.description(),
                mismatches,
        })
    }
}

fn compare_dense_sparse<T, C>(
    x: &dyn DenseAccess<T>,
    y: &dyn SparseAccess<T>,
    comparator: &C,
    swap_order: bool,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(x.rows() == y.rows() && x.cols() == y.cols());

    let y_triplets = y.fetch_triplets();

    let y_hash = try_build_sparse_hash_map(y.rows(), y.cols(), &y_triplets);

    match y_hash {
        Ok(y_hash) => {
            let mismatches = find_dense_sparse_mismatches(x, &y_hash, comparator, swap_order);
            if let Some(mismatches) = mismatches {
                Err(MatrixComparisonFailure::MismatchedElements(mismatches))
            } else {
                Ok(())
            }
        },
        Err(build_error) => {
            let make_entry = |coord| if swap_order {
                Entry::Left(coord)
            } else {
                Entry::Right(coord)
            };

            use MatrixComparisonFailure::*;
            match build_error {
                HashMapBuildError::OutOfBoundsCoord(coord)
                    => Err(SparseEntryOutOfBounds(make_entry(coord))),
                HashMapBuildError::DuplicateCoord(coord)
                    => Err(DuplicateSparseEntry(make_entry(coord)))
            }
        }
    }
}

fn compare_dense_dense<T, C>(
    x: &dyn DenseAccess<T>,
    y: &dyn DenseAccess<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(x.rows() == y.rows() && x.cols() == y.cols());

    let mut mismatches = Vec::new();
    for i in 0..x.rows() {
        for j in 0..x.cols() {
            let a = x.fetch_single(i, j);
            let b = y.fetch_single(i, j);
            if let Err(error) = comparator.compare(&a, &b) {
                mismatches.push(MatrixElementComparisonFailure {
                    x: a.clone(),
                    y: b.clone(),
                    error,
                    row: i,
                    col: j,
                });
            }
        }
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(MatrixComparisonFailure::MismatchedElements(
            ElementsMismatch {
                comparator_description: comparator.description(),
                mismatches,
            },
        ))
    }
}

pub fn compare_matrices<T, C>(
    x: impl Matrix<T>,
    y: impl Matrix<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    let shapes_match = x.rows() == y.rows() && x.cols() == y.cols();
    if shapes_match {
        use Access::{Dense, Sparse};
        let result = match (x.access(), y.access()) {
            (Dense(x_access), Dense(y_access)) => {
                compare_dense_dense(x_access, y_access, comparator)
            }
            (Dense(x_access), Sparse(y_access)) => {
                let swap = false;
                compare_dense_sparse(x_access, y_access, comparator, swap)
            }
            (Sparse(x_access), Dense(y_access)) => {
                let swap = true;
                compare_dense_sparse(y_access, x_access, comparator, swap)
            }
            (Sparse(x_access), Sparse(y_access)) => {
                compare_sparse_sparse(x_access, y_access, comparator)
            }
        };
        result
    } else {
        Err(MatrixComparisonFailure::MismatchedDimensions(
            DimensionMismatch {
                dim_x: (x.rows(), x.cols()),
                dim_y: (y.rows(), y.cols()),
            },
        ))
    }
}
