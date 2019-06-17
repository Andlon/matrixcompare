use std::fmt;

use crate::comparators::{ComparisonFailure, ElementwiseComparator};
use crate::{Accessor, DenseAccessor, Matrix};

const MAX_MISMATCH_REPORTS: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatrixElementComparisonFailure<T, E>
where
    E: ComparisonFailure,
{
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

#[derive(Debug, PartialEq)]
pub enum MatrixComparisonResult<T, C>
where
    C: ElementwiseComparator<T>,
{
    Match,
    MismatchedDimensions {
        dim_x: (usize, usize),
        dim_y: (usize, usize),
    },
    MismatchedElements {
        comparator: C,
        mismatches: Vec<MatrixElementComparisonFailure<T, C::Error>>,
    },
}

impl<T, C> MatrixComparisonResult<T, C>
where
    T: fmt::Display,
    C: ElementwiseComparator<T>,
{
    pub fn panic_message(&self) -> Option<String> {
        match self {
            &MatrixComparisonResult::MismatchedElements {
                ref comparator,
                ref mismatches,
            } => {
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
                    description = comparator.description(),
                    mismatches = formatted_mismatches,
                    overflow_msg = overflow_msg
                ))
            }
            &MatrixComparisonResult::MismatchedDimensions { dim_x, dim_y } => Some(format!(
                "\n
Dimensions of matrices X and Y do not match.
 dim(X) = {x_rows} x {x_cols}
 dim(Y) = {y_rows} x {y_cols}
\n",
                x_rows = dim_x.0,
                x_cols = dim_x.1,
                y_rows = dim_y.0,
                y_cols = dim_y.1
            )),
            _ => None,
        }
    }
}

fn fetch_dense_dense_mismatches<T, C>(
    x: &DenseAccessor<T>,
    y: &DenseAccessor<T>,
    comparator: &C,
) -> Vec<MatrixElementComparisonFailure<T, C::Error>>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
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
    mismatches
}

pub fn compare_matrices<T, C>(
    x: impl Matrix<T>,
    y: impl Matrix<T>,
    comparator: C,
) -> MatrixComparisonResult<T, C>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    let shapes_match = x.rows() == y.rows() && x.cols() == y.cols();
    if shapes_match {
        use Accessor::Dense;
        let mismatches = match (x.access(), y.access()) {
            (Dense(x_access), Dense(y_access)) => {
                fetch_dense_dense_mismatches(x_access, y_access, &comparator)
            }
            _ => unimplemented!(),
        };

        if mismatches.is_empty() {
            MatrixComparisonResult::Match
        } else {
            MatrixComparisonResult::MismatchedElements {
                comparator,
                mismatches,
            }
        }
    } else {
        MatrixComparisonResult::MismatchedDimensions {
            dim_x: (x.rows(), x.cols()),
            dim_y: (y.rows(), y.cols()),
        }
    }
}
