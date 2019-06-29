use matcomp::comparators::{ElementwiseComparator, ExactElementwiseComparator, ExactError};
use matcomp::mock::MockDenseMatrix;
use matcomp::{assert_matrix_eq, ElementsMismatch};
use matcomp::{compare_matrices, DimensionMismatch, MatrixComparisonResult};
use quickcheck::{quickcheck, TestResult};

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

        let expected = MatrixComparisonResult::MismatchedDimensions(DimensionMismatch { dim_x: (m, n), dim_y: (p, q) });

        TestResult::from_bool(compare_matrices(x, y, &comp) == expected)
    }
}

quickcheck! {
    fn property_elementwise_comparison_matrix_matches_self(m: usize, n: usize) -> bool {
        let comp = ExactElementwiseComparator;
        let ref x = MockDenseMatrix::from_row_major(m, n, vec![0; m * n]);

        compare_matrices(x, x, &comp) == MatrixComparisonResult::Match
    }
}

#[test]
fn compare_matrices_reports_correct_mismatches() {
    use matcomp::MatrixComparisonResult::MismatchedElements;
    use matcomp::MatrixElementComparisonFailure;

    let comp = ExactElementwiseComparator;
    let description =
        <ExactElementwiseComparator as ElementwiseComparator<usize>>::description(&comp);

    {
        // Single element matrices
        let ref x = MockDenseMatrix::from_row_major(1, 1, vec![1]);
        let ref y = MockDenseMatrix::from_row_major(1, 1, vec![2]);

        let expected = MismatchedElements(ElementsMismatch {
            comparator_description: description.clone(),
            mismatches: vec![MatrixElementComparisonFailure {
                x: 1,
                y: 2,
                error: ExactError,
                row: 0,
                col: 0,
            }],
        });

        assert_eq!(compare_matrices(x, y, &comp), expected);
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

        let expected = MismatchedElements(ElementsMismatch {
            comparator_description: description.clone(),
            mismatches: mismatches,
        });

        assert_eq!(compare_matrices(x, y, &comp), expected);
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

        let expected = MismatchedElements(ElementsMismatch {
            comparator_description: description.clone(),
            mismatches: mismatches,
        });

        assert_eq!(compare_matrices(x, y, &comp), expected);
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

        let expected = MismatchedElements(ElementsMismatch {
            comparator_description: description.clone(),
            mismatches: mismatches,
        });

        assert_eq!(compare_matrices(x, y, &comp), expected);
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
