use matrixcompare::{
    DimensionMismatch, ElementsMismatch, Entry, MatrixComparisonFailure,
    MatrixElementComparisonFailure,
};
use std::ops::Range;

pub const MATRIX_DIM_RANGE: Range<usize> = 0..5;

/// Reverses a comparison result.
///
/// See docs for `reverse_failure`.
pub fn reverse_result<T, E>(
    result: Result<(), MatrixComparisonFailure<T, E>>,
) -> Result<(), MatrixComparisonFailure<T, E>> {
    result.map_err(|err| reverse_failure(err))
}

/// Reverses the role of left and right in the given failure object.
///
/// Only used for testing that comparison is symmetric.
/// It is implicitly assumed that the error metric is symmetric.
pub fn reverse_failure<T, E>(
    failure: MatrixComparisonFailure<T, E>,
) -> MatrixComparisonFailure<T, E> {
    use MatrixComparisonFailure::*;
    match failure {
        MismatchedDimensions(dim) => MismatchedDimensions(reverse_dimension_mismatch(dim)),
        MismatchedElements(elements) => MismatchedElements(reverse_elements_mismatch(elements)),
        SparseEntryOutOfBounds(entry) => SparseEntryOutOfBounds(reverse_entry(entry)),
        DuplicateSparseEntry(entry) => SparseEntryOutOfBounds(reverse_entry(entry)),
    }
}

fn reverse_dimension_mismatch(dims: DimensionMismatch) -> DimensionMismatch {
    DimensionMismatch {
        dim_left: dims.dim_right,
        dim_right: dims.dim_left,
    }
}

fn reverse_elements_mismatch<T, E>(mismatch: ElementsMismatch<T, E>) -> ElementsMismatch<T, E> {
    ElementsMismatch {
        comparator_description: mismatch.comparator_description,
        mismatches: mismatch
            .mismatches
            .into_iter()
            .map(reverse_matrix_element_comparison_failure)
            .collect(),
    }
}

fn reverse_matrix_element_comparison_failure<T, E>(
    failure: MatrixElementComparisonFailure<T, E>,
) -> MatrixElementComparisonFailure<T, E> {
    MatrixElementComparisonFailure {
        left: failure.right,
        right: failure.left,
        error: failure.error,
        row: failure.row,
        col: failure.col,
    }
}

fn reverse_entry(entry: Entry) -> Entry {
    match entry {
        Entry::Left(coord) => Entry::Right(coord),
        Entry::Right(coord) => Entry::Left(coord),
    }
}
