use crate::comparators::ElementwiseComparator;
use crate::{
    Access, Coordinate, DenseAccess, DimensionMismatch, ElementsMismatch, Matrix,
    MatrixComparisonFailure, MatrixElementComparisonFailure, SparseAccess,
};
use num::Zero;
use std::collections::{HashMap, HashSet};

use crate::Entry;

enum HashMapBuildError {
    OutOfBoundsCoord(Coordinate),
    DuplicateCoord(Coordinate),
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
        } else if matrix.insert((i, j), v).is_some() {
            return Err(HashMapBuildError::DuplicateCoord((i, j)));
        }
    }

    Ok(matrix)
}

fn compare_sparse_sparse<T, C>(
    left: &dyn SparseAccess<T>,
    right: &dyn SparseAccess<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(left.rows() == right.rows() && left.cols() == right.cols());

    let left_hash = try_build_sparse_hash_map(left.rows(), left.cols(), &left.fetch_triplets())
        .map_err(|build_error| match build_error {
            HashMapBuildError::OutOfBoundsCoord(coord) => {
                MatrixComparisonFailure::SparseEntryOutOfBounds(Entry::Left(coord))
            }
            HashMapBuildError::DuplicateCoord(coord) => {
                MatrixComparisonFailure::DuplicateSparseEntry(Entry::Left(coord))
            }
        })?;

    let right_hash = try_build_sparse_hash_map(right.rows(), right.cols(), &right.fetch_triplets())
        .map_err(|build_error| match build_error {
            HashMapBuildError::OutOfBoundsCoord(coord) => {
                MatrixComparisonFailure::SparseEntryOutOfBounds(Entry::Right(coord))
            }
            HashMapBuildError::DuplicateCoord(coord) => {
                MatrixComparisonFailure::DuplicateSparseEntry(Entry::Right(coord))
            }
        })?;

    let mut mismatches = Vec::new();
    let left_keys: HashSet<_> = left_hash.keys().collect();
    let right_keys: HashSet<_> = right_hash.keys().collect();
    let zero = T::zero();

    for coord in left_keys.union(&right_keys) {
        let a = left_hash.get(coord).unwrap_or(&zero);
        let b = right_hash.get(coord).unwrap_or(&zero);
        if let Err(error) = comparator.compare(&a, &b) {
            mismatches.push(MatrixElementComparisonFailure {
                left: a.clone(),
                right: b.clone(),
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
                    left: a.clone(),
                    right: b.clone(),
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
    dense: &dyn DenseAccess<T>,
    sparse: &dyn SparseAccess<T>,
    comparator: &C,
    swap_order: bool,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(dense.rows() == sparse.rows() && dense.cols() == sparse.cols());

    let triplets = sparse.fetch_triplets();

    let sparse_hash = try_build_sparse_hash_map(sparse.rows(), sparse.cols(), &triplets);

    match sparse_hash {
        Ok(y_hash) => {
            let mismatches = find_dense_sparse_mismatches(dense, &y_hash, comparator, swap_order);
            if let Some(mismatches) = mismatches {
                Err(MatrixComparisonFailure::MismatchedElements(mismatches))
            } else {
                Ok(())
            }
        }
        Err(build_error) => {
            let make_entry = |coord| {
                if swap_order {
                    Entry::Left(coord)
                } else {
                    Entry::Right(coord)
                }
            };

            use MatrixComparisonFailure::*;
            match build_error {
                HashMapBuildError::OutOfBoundsCoord(coord) => {
                    Err(SparseEntryOutOfBounds(make_entry(coord)))
                }
                HashMapBuildError::DuplicateCoord(coord) => {
                    Err(DuplicateSparseEntry(make_entry(coord)))
                }
            }
        }
    }
}

fn compare_dense_dense<T, C>(
    left: &dyn DenseAccess<T>,
    right: &dyn DenseAccess<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    // We assume the compatibility of dimensions have been checked by the outer calling function
    assert!(left.rows() == right.rows() && left.cols() == right.cols());

    let mut mismatches = Vec::new();
    for i in 0..left.rows() {
        for j in 0..left.cols() {
            let a = left.fetch_single(i, j);
            let b = right.fetch_single(i, j);
            if let Err(error) = comparator.compare(&a, &b) {
                mismatches.push(MatrixElementComparisonFailure {
                    left: a.clone(),
                    right: b.clone(),
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

/// Comparison of two matrices.
///
/// Most users will only need to use the comparison macro. This function is mainly of use to
/// users who want to build their own macros.
pub fn compare_matrices<T, C>(
    left: impl Matrix<T>,
    right: impl Matrix<T>,
    comparator: &C,
) -> Result<(), MatrixComparisonFailure<T, C::Error>>
where
    T: Zero + Clone,
    C: ElementwiseComparator<T>,
{
    let shapes_match = left.rows() == right.rows() && left.cols() == right.cols();
    if shapes_match {
        use Access::{Dense, Sparse};
        match (left.access(), right.access()) {
            (Dense(left_access), Dense(right_access)) => {
                compare_dense_dense(left_access, right_access, comparator)
            }
            (Dense(left_access), Sparse(right_access)) => {
                let swap = false;
                compare_dense_sparse(left_access, right_access, comparator, swap)
            }
            (Sparse(left_access), Dense(right_access)) => {
                let swap = true;
                compare_dense_sparse(right_access, left_access, comparator, swap)
            }
            (Sparse(left_access), Sparse(right_access)) => {
                compare_sparse_sparse(left_access, right_access, comparator)
            }
        }
    } else {
        Err(MatrixComparisonFailure::MismatchedDimensions(
            DimensionMismatch {
                dim_left: (left.rows(), left.cols()),
                dim_right: (right.rows(), right.cols()),
            },
        ))
    }
}
