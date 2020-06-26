use core::fmt;
use std::fmt::{Display, Formatter};

const MAX_MISMATCH_REPORTS: usize = 12;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MatrixElementComparisonFailure<T, E> {
    pub left: T,
    pub right: T,
    pub error: E,
    pub row: usize,
    pub col: usize,
}

impl<T, E> Display for MatrixElementComparisonFailure<T, E>
where
    T: Display,
    E: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({i}, {j}): x = {x}, y = {y}.",
            i = self.row,
            j = self.col,
            x = self.left,
            y = self.right
        )?;
        write!(f, "{}", self.error)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DimensionMismatch {
    pub dim_left: (usize, usize),
    pub dim_right: (usize, usize),
}

impl Display for DimensionMismatch {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            r"Dimensions of matrices X (left) and Y (right) do not match.
 dim(X) = {x_rows} x {x_cols}
 dim(Y) = {y_rows} x {y_cols}",
            x_rows = self.dim_left.0,
            x_cols = self.dim_left.1,
            y_rows = self.dim_right.0,
            y_cols = self.dim_right.1
        )
    }
}

/// A pair of (row, column) coordinates in a matrix.
pub type Coordinate = (usize, usize);

/// A coordinate in the left or right matrix being compared.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    Left(Coordinate),
    Right(Coordinate),
}

impl Display for Entry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Left((i, j)) => write!(f, "Left({}, {})", i, j),
            Self::Right((i, j)) => write!(f, "Right({}, {})", i, j),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElementsMismatch<T, Error> {
    pub comparator_description: String,
    pub mismatches: Vec<MatrixElementComparisonFailure<T, Error>>,
}

impl<T, Error> Display for ElementsMismatch<T, Error>
where
    T: Display,
    Error: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // TODO: Aligned output
        let mut formatted_mismatches = String::new();

        let mismatches_overflow = self.mismatches.len() > MAX_MISMATCH_REPORTS;
        // TODO: Write directly to formatter
        let overflow_msg = if mismatches_overflow {
            let num_hidden_entries = self.mismatches.len() - MAX_MISMATCH_REPORTS;
            format!(
                " ... ({} mismatching elements not shown)\n",
                num_hidden_entries
            )
        } else {
            String::new()
        };

        for mismatch in self.mismatches.iter().take(MAX_MISMATCH_REPORTS) {
            formatted_mismatches.push_str(" ");
            formatted_mismatches.push_str(&mismatch.to_string());
            formatted_mismatches.push_str("\n");
        }

        // Strip off the last newline from the above
        formatted_mismatches = formatted_mismatches.trim_end().to_string();

        write!(
            f,
            "Matrices X (left) and Y (right) have {num} mismatched element pairs.
The mismatched elements are listed below, in the format
(row, col): x = X[[row, col]], y = Y[[row, col]].

{mismatches}
{overflow_msg}
Comparison criterion: {description}",
            num = self.mismatches.len(),
            description = self.comparator_description,
            mismatches = formatted_mismatches,
            overflow_msg = overflow_msg
        )
    }
}

/// The error type associated with matrix comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixComparisonFailure<T, Error> {
    MismatchedDimensions(DimensionMismatch),
    MismatchedElements(ElementsMismatch<T, Error>),
    SparseEntryOutOfBounds(Entry),
    DuplicateSparseEntry(Entry),
}

impl<T, E> std::error::Error for MatrixComparisonFailure<T, E>
where
    T: fmt::Debug + Display,
    E: fmt::Debug + Display,
{
}

impl<T, Error> Display for MatrixComparisonFailure<T, Error>
where
    T: Display,
    Error: Display,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            MatrixComparisonFailure::MismatchedElements(ref mismatch) => mismatch.fmt(f),
            MatrixComparisonFailure::MismatchedDimensions(ref mismatch) => mismatch.fmt(f),
            MatrixComparisonFailure::SparseEntryOutOfBounds(entry) => write!(
                f,
                r"At least one sparse entry is out of bounds. Example: {}.",
                entry
            ),
            MatrixComparisonFailure::DuplicateSparseEntry(entry) => write!(
                f,
                r"At least one duplicate sparse entry detected. Example: {}.",
                entry
            ),
        }
    }
}
