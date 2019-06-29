use std::fmt;

use crate::comparators::{ComparisonFailure, ElementwiseComparator};
use crate::{Accessor, DenseAccessor, Matrix, SparseAccessor};

const MAX_MISMATCH_REPORTS: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatrixElementComparisonFailure<T, E> {
    pub x: T,
    pub y: T,
    pub error: E,
    pub row: usize,
    pub col: usize,
}

impl<T, E> fmt::Display for MatrixElementComparisonFailure<T, E>
where
    T: fmt::Display,
    E: ComparisonFailure,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({i}, {j}): x = {x}, y = {y}.{reason}",
            i = self.row,
            j = self.col,
            x = self.x,
            y = self.y,
            reason = self
                .error
                .failure_reason()
                // Add a space before the reason
                .map(|s| format!(" {}", s))
                .unwrap_or(String::new())
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DimensionMismatch {
    pub dim_x: (usize, usize),
    pub dim_y: (usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfBoundsIndices {
    pub indices_x: Vec<(usize, usize)>,
    pub indices_y: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementsMismatch<T, Error> {
    pub comparator_description: String,
    pub mismatches: Vec<MatrixElementComparisonFailure<T, Error>>,
}

#[derive(Debug, PartialEq)]
pub enum MatrixComparisonResult<T, Error> {
    Match,
    MismatchedDimensions(DimensionMismatch),
    MismatchedElements(ElementsMismatch<T, Error>),
}

impl<T, Error> MatrixComparisonResult<T, Error>
where
    T: fmt::Display,
    Error: ComparisonFailure,
{
    pub fn panic_message(&self) -> Option<String> {
        match self {
            &MatrixComparisonResult::MismatchedElements(ElementsMismatch {
                ref comparator_description,
                ref mismatches,
            }) => {
                // TODO: Aligned output
                let mut formatted_mismatches = String::new();

                let mismatches_overflow = mismatches.len() > MAX_MISMATCH_REPORTS;
                let overflow_msg = if mismatches_overflow {
                    let num_hidden_entries = mismatches.len() - MAX_MISMATCH_REPORTS;
                    format!(
                        " ... ({} mismatching elements not shown)\n",
                        num_hidden_entries
                    )
                } else {
                    String::new()
                };

                for mismatch in mismatches.iter().take(MAX_MISMATCH_REPORTS) {
                    formatted_mismatches.push_str(" ");
                    formatted_mismatches.push_str(&mismatch.to_string());
                    formatted_mismatches.push_str("\n");
                }

                // Strip off the last newline from the above
                formatted_mismatches = formatted_mismatches.trim_end().to_string();

                Some(format!(
                    "\n
Matrices X and Y have {num} mismatched element pairs.
The mismatched elements are listed below, in the format
(row, col): x = X[[row, col]], y = Y[[row, col]].

{mismatches}
{overflow_msg}
Comparison criterion: {description}
\n",
                    num = mismatches.len(),
                    description = comparator_description,
                    mismatches = formatted_mismatches,
                    overflow_msg = overflow_msg
                ))
            }
            &MatrixComparisonResult::MismatchedDimensions(DimensionMismatch { dim_x, dim_y }) => {
                Some(format!(
                    "\n
Dimensions of matrices X and Y do not match.
 dim(X) = {x_rows} x {x_cols}
 dim(Y) = {y_rows} x {y_cols}
\n",
                    x_rows = dim_x.0,
                    x_cols = dim_x.1,
                    y_rows = dim_y.0,
                    y_cols = dim_y.1
                ))
            }
            _ => None,
        }
    }
}

//type DuplicateEntriesMap<T> = HashMap<(usize, usize), Vec<T>>;

//fn try_build_hash_map_from_triplets<T>(triplets: &[(usize, usize, T)])
//    -> Result<HashMap<(usize, usize), T>, DuplicateEntriesMap<T>>
//where
//    T: Clone
//{
//    let mut duplicates = DuplicateEntriesMap::new();
//    let mut matrix = HashMap::new();
//
//    for (i, j, v) in triplets.iter().cloned() {
//        if let Some(old_entry) = matrix.insert((i, j), v) {
//            duplicates.entry((i, j)).or_insert_with(|| Vec::new()).push(old_entry);
//        }
//    }
//
//    if duplicates.is_empty() {
//        Ok(matrix)
//    } else {
//        // If there are duplicates, we must also be sure to update the duplicates map with
//        // the duplicate entries that are still in the matrix hash map
//        for (key, ref mut values) in &mut duplicates {
//            values.push(matrix.get(key)
//                              .cloned()
//                              .expect("Entry (i, j) must be in the map,\
//                                        otherwise it wouldn't be in duplicates"));
//        }
//
//        Err(duplicates)
//    }
//}

fn compare_dense_sparse<T, C>(
    x: &DenseAccessor<T>,
    y: &SparseAccessor<T>,
    _comparator: &C,
) -> MatrixComparisonResult<T, C::Error>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    // TODO: Delegate to function
    // We assume shapes are already handled by the outer calling function
    assert!(x.rows() == y.rows() && x.cols() == y.cols());

    //    let triplets = y.fetch_triplets();

    //    let hash_matrix = try_build_hash_map_from_triplets(&triplets);

    unimplemented!()
}

fn compare_dense_dense<T, C>(
    x: &DenseAccessor<T>,
    y: &DenseAccessor<T>,
    comparator: C,
) -> MatrixComparisonResult<T, C::Error>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    // TODO: Rewrite this so that we instead delegate to a function
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
        MatrixComparisonResult::Match
    } else {
        MatrixComparisonResult::MismatchedElements(ElementsMismatch {
            comparator_description: comparator.description(),
            mismatches,
        })
    }
}

//struct ReverseComparatorAdapter<C> {
//    comparator: C
//}
//
//impl<C, T> ElementwiseComparator<T> for ReverseComparatorAdapter<C>
//    where C: ElementwiseComparator<T>
//{
//    type Error = C::Error;
//
//    fn compare(&self, x: &T, y: &T) -> Result<(), C::Error> {
//        self.comparator.compare(y, x)
//    }
//
//    fn description(&self) -> String {
//        self.comparator.description()
//    }
//}

pub fn compare_matrices<T, C>(
    x: impl Matrix<T>,
    y: impl Matrix<T>,
    comparator: C,
) -> MatrixComparisonResult<T, C::Error>
where
    T: Clone,
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
                compare_dense_sparse(x_access, y_access, &comparator)
            }
            _ => unimplemented!(),
        };
        result
    } else {
        MatrixComparisonResult::MismatchedDimensions(DimensionMismatch {
            dim_x: (x.rows(), x.cols()),
            dim_y: (y.rows(), y.cols()),
        })
    }
}
