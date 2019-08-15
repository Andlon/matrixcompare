use crate::comparators::ElementwiseComparator;
use crate::{
    Accessor, DenseAccessor, DimensionMismatch, DuplicateEntries, ElementsMismatch, Matrix,
    MatrixComparisonFailure, MatrixElementComparisonFailure, OutOfBoundsIndices, SparseAccessor,
};
use num::Zero;
use std::collections::{HashMap, HashSet};

type DuplicateEntriesMap<T> = HashMap<(usize, usize), Vec<T>>;

fn try_build_hash_map_from_triplets<T>(
    triplets: &[(usize, usize, T)],
) -> Result<HashMap<(usize, usize), T>, DuplicateEntriesMap<T>>
where
    T: Clone,
{
    let mut duplicates = DuplicateEntriesMap::new();
    let mut matrix = HashMap::new();

    for (i, j, v) in triplets.iter().cloned() {
        if let Some(old_entry) = matrix.insert((i, j), v) {
            duplicates
                .entry((i, j))
                .or_insert_with(|| Vec::new())
                .push(old_entry);
        }
    }

    if duplicates.is_empty() {
        Ok(matrix)
    } else {
        // If there are duplicates, we must also be sure to update the duplicates map with
        // the duplicate entries that are still in the matrix hash map
        for (key, ref mut values) in &mut duplicates {
            values.push(matrix.get(key).cloned().expect(
                "Entry (i, j) must be in the map,\
                 otherwise it wouldn't be in duplicates",
            ));
        }

        Err(duplicates)
    }
}

fn compare_sparse_sparse<T, C>(
    x: &SparseAccessor<T>,
    y: &SparseAccessor<T>,
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

    let x_out_of_bounds = find_out_of_bounds_indices(x.rows(), x.cols(), &x_triplets);
    let y_out_of_bounds = find_out_of_bounds_indices(y.rows(), y.cols(), &y_triplets);

    if !x_out_of_bounds.is_empty() || !y_out_of_bounds.is_empty() {
        Err(MatrixComparisonFailure::SparseIndicesOutOfBounds(
            OutOfBoundsIndices {
                indices_x: x_out_of_bounds,
                indices_y: y_out_of_bounds,
            },
        ))
    } else {
        let x_hash = try_build_hash_map_from_triplets(&x_triplets);
        let y_hash = try_build_hash_map_from_triplets(&y_triplets);

        if x_hash.is_err() || y_hash.is_err() {
            let x_duplicates = x_hash.err().unwrap_or(HashMap::new());
            let y_duplicates = y_hash.err().unwrap_or(HashMap::new());
            Err(MatrixComparisonFailure::DuplicateSparseEntries(
                DuplicateEntries {
                    x_duplicates,
                    y_duplicates,
                },
            ))
        } else {
            let mut mismatches = Vec::new();
            let x_hash = x_hash.ok().unwrap();
            let y_hash = y_hash.ok().unwrap();
            let x_keys: HashSet<_> = x_hash.keys().collect();
            let y_keys: HashSet<_> = y_hash.keys().collect();
            let zero = T::zero();

            for key in x_keys.union(&y_keys) {
                let a = x_hash.get(key).unwrap_or(&zero);
                let b = y_hash.get(key).unwrap_or(&zero);
                if let Err(error) = comparator.compare(&a, &b) {
                    mismatches.push(MatrixElementComparisonFailure {
                        x: a.clone(),
                        y: b.clone(),
                        error,
                        row: key.0,
                        col: key.1,
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
    }
}

fn find_out_of_bounds_indices<T>(
    rows: usize,
    cols: usize,
    triplets: &[(usize, usize, T)],
) -> Vec<(usize, usize)> {
    triplets
        .iter()
        .filter(|&(i, j, _)| i > &rows || j > &cols)
        .map(|&(i, j, _)| (i, j))
        .collect()
}

fn compare_dense_sparse<T, C>(
    x: &DenseAccessor<T>,
    y: &SparseAccessor<T>,
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

    let y_out_of_bounds = find_out_of_bounds_indices(y.rows(), y.cols(), &y_triplets);
    if !y_out_of_bounds.is_empty() {
        Err(MatrixComparisonFailure::SparseIndicesOutOfBounds(
            OutOfBoundsIndices {
                indices_x: Vec::new(),
                indices_y: y_out_of_bounds,
            },
        ))
    } else {
        let y_hash = try_build_hash_map_from_triplets(&y_triplets);

        if let Err(y_duplicates) = y_hash {
            Err(MatrixComparisonFailure::DuplicateSparseEntries(
                DuplicateEntries {
                    x_duplicates: HashMap::new(),
                    y_duplicates,
                },
            ))
        } else {
            let mut mismatches = Vec::new();
            let y_hash = y_hash.ok().unwrap();
            let zero = T::zero();

            for i in 0..x.rows() {
                for j in 0..x.cols() {
                    let a = &x.fetch_single(i, j);
                    let b = y_hash.get(&(i, j)).unwrap_or(&zero);
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
    }
}

fn compare_dense_dense<T, C>(
    x: &DenseAccessor<T>,
    y: &DenseAccessor<T>,
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
        use Accessor::{Dense, Sparse};
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
