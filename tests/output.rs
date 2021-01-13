use matrixcompare::comparators::{AbsoluteElementwiseComparator, ExactElementwiseComparator};
use matrixcompare::compare_matrices;
use matrixcompare_mock::{mock_matrix, MockSparseMatrix};

use pretty_assertions::assert_eq;

#[test]
fn mismatched_elements() {
    let a = mock_matrix![1, 2, 3; 4, 5, 6];
    let b = mock_matrix![1, 2, 9; 5, 4, 6];

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"Matrices X (left) and Y (right) have 3 mismatched element pairs.
The mismatched elements are listed below, in the format
(row, col): x = X[[row, col]], y = Y[[row, col]].

 (0, 2): x = 3, y = 9.
 (1, 0): x = 4, y = 5.
 (1, 1): x = 5, y = 4.

Comparison criterion: exact equality x == y."
    );
}

#[test]
fn mismatched_elements_abs_f64() {
    let a = mock_matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
    let b = mock_matrix![1.0, 2.0, 9.0; 5.0, 4.0, 6.0];

    let err = compare_matrices(&a, &b, &AbsoluteElementwiseComparator { tol: 1e-12 }).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"Matrices X (left) and Y (right) have 3 mismatched element pairs.
The mismatched elements are listed below, in the format
(row, col): x = X[[row, col]], y = Y[[row, col]].

 (0, 2): x = 3, y = 9. Absolute error: 6.
 (1, 0): x = 4, y = 5. Absolute error: 1.
 (1, 1): x = 5, y = 4. Absolute error: 1.

Comparison criterion: absolute difference, |x - y| <= 0.000000000001."
    );
}

#[test]
fn mismatched_dimensions() {
    let a = mock_matrix![1, 2; 4, 5];
    let b = mock_matrix![1, 2, 9; 5, 4, 6];

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"Dimensions of matrices X (left) and Y (right) do not match.
 dim(X) = 2 x 2
 dim(Y) = 2 x 3"
    );
}

#[test]
fn duplicate_entry_left() {
    let a = MockSparseMatrix::from_triplets(3, 3, vec![(1, 0, 2), (1, 0, 2)]);
    let b = MockSparseMatrix::from_triplets(3, 3, vec![]);

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"At least one duplicate sparse entry detected. Example: Left(1, 0)."
    );
}

#[test]
fn duplicate_entry_right() {
    let a = MockSparseMatrix::from_triplets(3, 3, vec![]);
    let b = MockSparseMatrix::from_triplets(3, 3, vec![(1, 0, 2), (1, 0, 2)]);

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"At least one duplicate sparse entry detected. Example: Right(1, 0)."
    );
}

#[test]
fn out_of_bounds_left() {
    let a = MockSparseMatrix::from_triplets(3, 3, vec![(5, 0, 2), (1, 0, 2)]);
    let b = MockSparseMatrix::from_triplets(3, 3, vec![]);

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"At least one sparse entry is out of bounds. Example: Left(5, 0)."
    );
}

#[test]
fn out_of_bounds_right() {
    let a = MockSparseMatrix::from_triplets(3, 3, vec![]);
    let b = MockSparseMatrix::from_triplets(3, 3, vec![(5, 0, 2), (1, 0, 2)]);

    let err = compare_matrices(&a, &b, &ExactElementwiseComparator).unwrap_err();
    let err_string = err.to_string();

    println!("{}", err);
    assert_eq!(
        err_string,
        r"At least one sparse entry is out of bounds. Example: Right(5, 0)."
    );
}
