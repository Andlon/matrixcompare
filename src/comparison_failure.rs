use core::fmt;
use crate::comparators::ComparisonFailure;
use std::collections::HashMap;
use std::fmt::Formatter;

const MAX_MISMATCH_REPORTS: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatrixElementComparisonFailure<T, E> {
    pub x: T,
    pub y: T,
    pub error: E,
    pub row: usize,
    pub col: usize,
}

impl<T, E> MatrixElementComparisonFailure<T, E> {
    pub fn reverse(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
            error: self.error,
            row: self.row,
            col: self.col
        }
    }
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

impl DimensionMismatch {
    pub fn reverse(self) -> Self {
        Self {
            dim_x: self.dim_y,
            dim_y: self.dim_x
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutOfBoundsIndices {
    pub indices_x: Vec<(usize, usize)>,
    pub indices_y: Vec<(usize, usize)>,
}

impl OutOfBoundsIndices {
    pub fn reverse(self) -> Self {
        Self {
            indices_x: self.indices_y,
            indices_y: self.indices_x
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementsMismatch<T, Error> {
    pub comparator_description: String,
    pub mismatches: Vec<MatrixElementComparisonFailure<T, Error>>,
}

impl<T, Error> ElementsMismatch<T, Error>
{
    pub fn reverse(self) -> Self {
        Self {
            comparator_description: self.comparator_description,
            mismatches: self.mismatches.into_iter().map(MatrixElementComparisonFailure::reverse).collect()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateEntries<T> {
    pub x_duplicates: HashMap<(usize, usize), Vec<T>>,
    pub y_duplicates: HashMap<(usize, usize), Vec<T>>,
}

impl<T> DuplicateEntries<T> {
    pub fn reverse(self) -> Self {
        Self {
            x_duplicates: self.y_duplicates,
            y_duplicates: self.x_duplicates
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatrixComparisonFailure<T, Error> {
    MismatchedDimensions(DimensionMismatch),
    MismatchedElements(ElementsMismatch<T, Error>),
    SparseIndicesOutOfBounds(OutOfBoundsIndices),
    DuplicateSparseEntries(DuplicateEntries<T>),
}

impl<T, Error> MatrixComparisonFailure<T, Error>
{
    /// "Reverses" the result, in the sense that the roles of x and y are interchanged.
    pub fn reverse(self) -> Self {
        use MatrixComparisonFailure::*;
        match self {
            MismatchedDimensions(dim) => MismatchedDimensions(dim.reverse()),
            MismatchedElements(elements) => MismatchedElements(elements.reverse()),
            SparseIndicesOutOfBounds(indices) => SparseIndicesOutOfBounds(indices.reverse()),
            DuplicateSparseEntries(duplicates) => DuplicateSparseEntries(duplicates.reverse())
        }
    }
}

impl<T, Error> fmt::Display for MatrixComparisonFailure<T, Error>
where
    T: fmt::Display,
    Error: ComparisonFailure,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            &MatrixComparisonFailure::MismatchedElements(ElementsMismatch {
                                                             ref comparator_description,
                                                             ref mismatches,
                                                         }) => {
                // TODO: Aligned output
                let mut formatted_mismatches = String::new();

                let mismatches_overflow = mismatches.len() > MAX_MISMATCH_REPORTS;
                // TODO: Write directly to formatter
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

                write!(f,
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
                )
            }
            &MatrixComparisonFailure::MismatchedDimensions(DimensionMismatch { dim_x, dim_y }) => {
                write!(f,
                    "\n
Dimensions of matrices X and Y do not match.
 dim(X) = {x_rows} x {x_cols}
 dim(Y) = {y_rows} x {y_cols}
\n",
                    x_rows = dim_x.0,
                    x_cols = dim_x.1,
                    y_rows = dim_y.0,
                    y_cols = dim_y.1
                )
            }
            // TODO
            &MatrixComparisonFailure::SparseIndicesOutOfBounds(ref _out_of_bounds) => {
                write!(f, "TODO: Error for out of bounds")
            }
            &MatrixComparisonFailure::DuplicateSparseEntries(ref _duplicate) => {
                write!(f, "TODO: Error for duplicate entries")
            }
        }
    }
}