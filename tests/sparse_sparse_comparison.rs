use matrixcompare::comparators::ExactElementwiseComparator;
use matrixcompare::{compare_matrices, MatrixComparisonFailure, Entry};
use matrixcompare_mock::{sparse_matrix_strategy_i64, sparse_matrix_strategy_normal_f64,
                         MockSparseMatrix};
use proptest::prelude::*;

mod common;
use common::MATRIX_DIM_RANGE;
use common::reverse_result;

#[test]
fn sparse_sparse_out_of_bounds() {
    use MatrixComparisonFailure::SparseEntryOutOfBounds;

    macro_rules! assert_out_of_bounds_detected {
        // oob1 and oob2 contain the out of bounds coordinates for sparse1 and sparse 2
        ($sparse1:expr, $sparse2:expr, $oob1:expr, $oob2:expr) => {
            // sparse1-sparse2
            {
                let result = compare_matrices(&$sparse1, &$sparse2, &ExactElementwiseComparator);
                let err = result.unwrap_err();
                match err {
                    SparseEntryOutOfBounds(Entry::Left(coord)) => assert!($oob1.contains(&coord)),
                    SparseEntryOutOfBounds(Entry::Right(coord)) => assert!($oob2.contains(&coord)),
                    _ => panic!("Unexpected variant")
                }
            }

            // sparse2-sparse1
            {
                let result = compare_matrices(&$sparse2, &$sparse1, &ExactElementwiseComparator);
                let err = result.unwrap_err();
                match err {
                    // Left-right get flipped since we're swapping the comparison order
                    SparseEntryOutOfBounds(Entry::Right(coord)) => assert!($oob1.contains(&coord)),
                    SparseEntryOutOfBounds(Entry::Left(coord)) => assert!($oob2.contains(&coord)),
                    _ => panic!("Unexpected variant")
                }
            }
        };
    }

    // Row out of bounds in sparse2
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 0, 2)]);
        let sparse2 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3), (1, 2, 6), (2, 0, 1)]);
        let oob1 = vec![];
        let oob2 = vec![(2, 0)];
        assert_out_of_bounds_detected!(sparse1, sparse2, oob1, oob2);
    }

    // Col out of bounds in sparse2
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 0, 2)]);
        let sparse2 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3), (1, 3, 1), (1, 2, 6)]);
        let oob1 = vec![];
        let oob2 = vec![(1, 3)];
        assert_out_of_bounds_detected!(sparse1, sparse2, oob1, oob2);
    }

    // Row and col out of bounds in sparse2
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 0, 2)]);
        let sparse2 = MockSparseMatrix::from_triplets(
            2,
            3,
            vec![(2, 3, 1), (0, 1, -3), (2, 0, 1), (1, 2, 6)],
        );
        let oob1 = vec![];
        let oob2 = vec![(2, 0), (2, 3)];
        assert_out_of_bounds_detected!(sparse1, sparse2, oob1, oob2);
    }

    // Row and col out of bounds in both sparse1 and sparse2
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 0, 2), (4, 6, 3)]);
        let sparse2 = MockSparseMatrix::from_triplets(
            2,
            3,
            vec![(2, 3, 1), (0, 1, -3), (2, 0, 1), (1, 2, 6)],
        );
        let oob1 = vec![(4, 6)];
        let oob2 = vec![(2, 0), (2, 3), (4, 6)];
        assert_out_of_bounds_detected!(sparse1, sparse2, oob1, oob2);
    }
}

#[test]
fn sparse_sparse_duplicate_entries() {
    use MatrixComparisonFailure::DuplicateSparseEntry;

    // Sparse2 has duplicate entries
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, 3), (1, 0, 3), (0, 2, 1)]);
        let sparse2 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3),
                                                                 (1, 0, 6),
                                                                 (1, 0, 3),
                                                                 (1, 2, 1)]);

        // sparse1-sparse1
        {
            let result = compare_matrices(&sparse1, &sparse2, &ExactElementwiseComparator);
            let err = result.unwrap_err();
            match err {
                DuplicateSparseEntry(Entry::Right(coord)) => assert_eq!(coord, (1, 0)),
                _ => panic!("Unexpected error")
            }
        }

        // sparse2-sparse1
        {
            let result = compare_matrices(&sparse2, &sparse1, &ExactElementwiseComparator);
            let err = result.unwrap_err();
            match err {
                DuplicateSparseEntry(Entry::Left(coord)) => assert_eq!(coord, (1, 0)),
                _ => panic!("Unexpected error")
            }
        }
    }

    // Both matrices have duplicate entries
    {
        let sparse1 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, 3), (0, 1, 3), (0, 2, 1)]);
        let sparse2 = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3),
                                                                 (1, 0, 6),
                                                                 (1, 0, 3),
                                                                 (1, 2, 1)]);

        // sparse1-sparse1
        {
            let result = compare_matrices(&sparse1, &sparse2, &ExactElementwiseComparator);
            let err = result.unwrap_err();
            match err {
                DuplicateSparseEntry(Entry::Left(coord)) => assert_eq!(coord, (0, 1)),
                DuplicateSparseEntry(Entry::Right(coord)) => assert_eq!(coord, (1, 0)),
                _ => panic!("Unexpected error")
            }
        }

        // sparse2-sparse1
        {
            let result = compare_matrices(&sparse2, &sparse1, &ExactElementwiseComparator);
            let err = result.unwrap_err();
            match err {
                // Left-right is flipped
                DuplicateSparseEntry(Entry::Right(coord)) => assert_eq!(coord, (0, 1)),
                DuplicateSparseEntry(Entry::Left(coord)) => assert_eq!(coord, (1, 0)),
                _ => panic!("Unexpected error")
            }
        }
    }
}

/// A strategy producing pairs of dense and sparse matrices with the same dimensions.
fn same_size_sparse_sparse_matrices(
) -> impl Strategy<Value = (MockSparseMatrix<i64>, MockSparseMatrix<i64>)> {
    let rows = MATRIX_DIM_RANGE;
    let cols = MATRIX_DIM_RANGE;

    (rows, cols).prop_flat_map(|(r, c)| {
        let dense = sparse_matrix_strategy_i64(Just(r), Just(c));
        let sparse = sparse_matrix_strategy_i64(Just(r), Just(c));
        (dense, sparse)
    })
}

proptest! {
    #[test]
    fn sparse_sparse_self_comparison_succeeds_i64(
        sparse in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        prop_assert!(compare_matrices(&sparse, &sparse, &ExactElementwiseComparator).is_ok())
    }

    #[test]
    fn sparse_sparse_matrices_should_compare_the_same_as_dense_dense_i64(
        sparse1 in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse2 in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&sparse1, &sparse2, &c),
                        compare_matrices(&sparse1.to_dense().unwrap(),
                                          sparse2.to_dense().unwrap(),
                                          &c));
    }

    #[test]
    fn same_size_sparse_sparse_matrices_should_compare_the_same_as_dense_dense_i64(
        (sparse1, sparse2) in same_size_sparse_sparse_matrices()
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&sparse1, &sparse2, &c),
                        compare_matrices(&sparse1.to_dense().unwrap(),
                                          sparse2.to_dense().unwrap(),
                                          &c));
    }

    #[test]
    fn sparse_sparse_matrices_should_compare_the_same_as_dense_dense_f64(
        sparse1 in sparse_matrix_strategy_normal_f64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse2 in sparse_matrix_strategy_normal_f64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&sparse1, &sparse2, &c),
                        compare_matrices(&sparse1.to_dense().unwrap(),
                                          sparse2.to_dense().unwrap(),
                                          &c));
    }

    #[test]
    fn sparse_sparse_comparison_is_symmetric_for_all_matrices_i64(
        sparse1 in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse2 in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        let result1 = compare_matrices(&sparse1, &sparse2, &c);
        let result2 = compare_matrices(&sparse2, &sparse1, &c);

        prop_assert_eq!(result1.clone(), reverse_result(result2.clone()));
        prop_assert_eq!(reverse_result(result1), result2);
    }

    #[test]
    fn same_size_sparse_sparse_comparison_is_symmetric_for_all_matrices_i64(
        (sparse1, sparse2) in same_size_sparse_sparse_matrices()
    ) {
        let c = ExactElementwiseComparator;
        let result1 = compare_matrices(&sparse1, &sparse2, &c);
        let result2 = compare_matrices(&sparse2, &sparse1, &c);

        prop_assert_eq!(result1.clone(), reverse_result(result2.clone()));
        prop_assert_eq!(reverse_result(result1), result2);
    }
}
