use std::fmt;

use crate::comparators::ComparisonFailure;
use crate::{Accessor, DenseAccessor, ElementwiseComparator, Matrix};

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
    C: ElementwiseComparator<T>
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
    C: ElementwiseComparator<T>
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
    C: ElementwiseComparator<T>
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
    C: ElementwiseComparator<T>
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

#[cfg(test)]
mod tests {
    use super::{compare_matrices, MatrixComparisonResult};
    use crate::comparators::{ExactElementwiseComparator, ExactError};
    use crate::mock::MockDenseMatrix;
    use crate::assert_matrix_eq;
    use quickcheck::TestResult;

    quickcheck! {
        fn property_elementwise_comparison_incompatible_matrices_yield_dimension_mismatch(
            m: usize,
            n: usize,
            p: usize,
            q: usize) -> TestResult {
            if m == p && n == q {
                return TestResult::discard()
            }

            // It does not actually matter which comparator we use here, but we need to pick one
            let comp = ExactElementwiseComparator;
            let ref x = MockDenseMatrix::from_row_major(m, n, vec![0; m * n]);
            let ref y = MockDenseMatrix::from_row_major(p, q, vec![0; p * q]);

            let expected = MatrixComparisonResult::MismatchedDimensions { dim_x: (m, n), dim_y: (p, q) };

            TestResult::from_bool(compare_matrices(x, y, comp) == expected)
        }
    }

    quickcheck! {
        fn property_elementwise_comparison_matrix_matches_self(m: usize, n: usize) -> bool {
            let comp = ExactElementwiseComparator;
            let ref x = MockDenseMatrix::from_row_major(m, n, vec![0; m * n]);

            compare_matrices(x, x, comp) == MatrixComparisonResult::Match
        }
    }

    #[test]
    fn compare_matrices_reports_correct_mismatches() {
        use super::MatrixComparisonResult::MismatchedElements;
        use super::MatrixElementComparisonFailure;

        let comp = ExactElementwiseComparator;

        {
            // Single element matrices
            let ref x = MockDenseMatrix::from_row_major(1, 1, vec![1]);
            let ref y = MockDenseMatrix::from_row_major(1, 1, vec![2]);

            let expected = MismatchedElements {
                comparator: comp,
                mismatches: vec![MatrixElementComparisonFailure {
                    x: 1,
                    y: 2,
                    error: ExactError,
                    row: 0,
                    col: 0,
                }],
            };

            assert_eq!(compare_matrices(x, y, comp), expected);
        }

        {
            // Mismatch in top-left and bottom-corner elements for a short matrix
            let ref x = MockDenseMatrix::from_row_major(2, 3, vec![0, 1, 2, 3, 4, 5]);
            let ref y = MockDenseMatrix::from_row_major(2, 3, vec![1, 1, 2, 3, 4, 6]);
            let mismatches = vec![
                MatrixElementComparisonFailure {
                    x: 0,
                    y: 1,
                    error: ExactError,
                    row: 0,
                    col: 0,
                },
                MatrixElementComparisonFailure {
                    x: 5,
                    y: 6,
                    error: ExactError,
                    row: 1,
                    col: 2,
                },
            ];

            let expected = MismatchedElements {
                comparator: comp,
                mismatches: mismatches,
            };

            assert_eq!(compare_matrices(x, y, comp), expected);
        }

        {
            // Mismatch in top-left and bottom-corner elements for a tall matrix
            let ref x = MockDenseMatrix::from_row_major(3, 2, vec![0, 1, 2, 3, 4, 5]);
            let ref y = MockDenseMatrix::from_row_major(3, 2, vec![1, 1, 2, 3, 4, 6]);
            let mismatches = vec![
                MatrixElementComparisonFailure {
                    x: 0,
                    y: 1,
                    error: ExactError,
                    row: 0,
                    col: 0,
                },
                MatrixElementComparisonFailure {
                    x: 5,
                    y: 6,
                    error: ExactError,
                    row: 2,
                    col: 1,
                },
            ];

            let expected = MismatchedElements {
                comparator: comp,
                mismatches: mismatches,
            };

            assert_eq!(compare_matrices(x, y, comp), expected);
        }

        {
            // Check some arbitrary elements
            let ref x = MockDenseMatrix::from_row_major(2, 4, vec![0, 1, 2, 3, 4, 5, 6, 7]);
            let ref y = MockDenseMatrix::from_row_major(2, 4, vec![0, 1, 3, 3, 4, 6, 6, 7]);

            let mismatches = vec![
                MatrixElementComparisonFailure {
                    x: 2,
                    y: 3,
                    error: ExactError,
                    row: 0,
                    col: 2,
                },
                MatrixElementComparisonFailure {
                    x: 5,
                    y: 6,
                    error: ExactError,
                    row: 1,
                    col: 1,
                },
            ];

            let expected = MismatchedElements {
                comparator: comp,
                mismatches: mismatches,
            };

            assert_eq!(compare_matrices(x, y, comp), expected);
        }
    }

    #[test]
    pub fn matrix_eq_absolute_compare_self_for_integer() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_matrix_eq!(x, x, comp = abs, tol = 0);
    }

    #[test]
    pub fn matrix_eq_absolute_compare_self_for_floating_point() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = abs, tol = 1e-10);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_absolute_mismatched_dimensions() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1, 2, 3, 4, 5, 6]);
        let y = MockDenseMatrix::from_row_major(2, 3, vec![1, 2, 3, 4]);
        assert_matrix_eq!(x, y, comp = abs, tol = 0);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_absolute_mismatched_floating_point_elements() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.00, 2.00, 3.00, 4.00, 5.00, 6.00]);
        let y = MockDenseMatrix::from_row_major(2, 3, vec![1.00, 2.01, 3.00, 3.99, 5.00, 6.00]);
        assert_matrix_eq!(x, y, comp = abs, tol = 1e-10);
    }

    #[test]
    pub fn matrix_eq_exact_compare_self_for_integer() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_matrix_eq!(x, x, comp = exact);
    }

    #[test]
    pub fn matrix_eq_exact_compare_self_for_floating_point() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = exact);
    }

    #[test]
    pub fn matrix_eq_ulp_compare_self() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = ulp, tol = 0);
    }

    #[test]
    pub fn matrix_eq_default_compare_self_for_floating_point() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x);
    }

    #[test]
    pub fn matrix_eq_default_compare_self_for_integer() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1, 2, 3, 4, 5, 6]);
        assert_matrix_eq!(x, x);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_ulp_different_signs() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, -3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, y, comp = ulp, tol = 0);
    }

    #[test]
    #[should_panic]
    pub fn matrix_eq_ulp_nan() {
        use std::f64;
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, y, comp = ulp, tol = 0);
    }

    #[test]
    pub fn matrix_eq_float_compare_self() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = float);
    }

    #[test]
    pub fn matrix_eq_float_compare_self_with_eps() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = float, eps = 1e-6);
    }

    #[test]
    pub fn matrix_eq_float_compare_self_with_ulp() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = float, ulp = 12);
    }

    #[test]
    pub fn matrix_eq_float_compare_self_with_eps_and_ulp() {
        let x = MockDenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_matrix_eq!(x, x, comp = float, eps = 1e-6, ulp = 12);
        assert_matrix_eq!(x, x, comp = float, ulp = 12, eps = 1e-6);
    }

    #[test]
    pub fn matrix_eq_pass_by_ref() {
        let x = MockDenseMatrix::from_row_major(1, 1, vec![0.0f64]);

        // Exercise all the macro definitions and make sure that we are able to call it
        // when the arguments are references.
        assert_matrix_eq!(&x, &x);
        assert_matrix_eq!(&x, &x, comp = exact);
        assert_matrix_eq!(&x, &x, comp = abs, tol = 0.0);
        assert_matrix_eq!(&x, &x, comp = ulp, tol = 0);
        assert_matrix_eq!(&x, &x, comp = float);
        assert_matrix_eq!(&x, &x, comp = float, eps = 0.0, ulp = 0);
    }
}
