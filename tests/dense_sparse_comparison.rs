use matrixcompare::comparators::ExactElementwiseComparator;
use matrixcompare::{compare_matrices, MatrixComparisonFailure, Entry};
use matrixcompare_core::Matrix;
use matrixcompare_mock::{
    dense_matrix_strategy_i64, dense_matrix_strategy_normal_f64, i64_range, mock_matrix,
    sparse_matrix_strategy_i64, sparse_matrix_strategy_normal_f64, MockDenseMatrix,
    MockSparseMatrix,
};
use proptest::prelude::*;

mod common;
use common::MATRIX_DIM_RANGE;

#[test]
fn dense_sparse_index_out_of_bounds() {
    use MatrixComparisonFailure::SparseEntryOutOfBounds;

    macro_rules! assert_out_of_bounds_detected {
        ($dense:expr, $sparse:expr, $oob:expr) => {
            // Dense-sparse
            {
                let result = compare_matrices(&$dense, &$sparse, &ExactElementwiseComparator);
                let err = result.unwrap_err();
                match err {
                    SparseEntryOutOfBounds(Entry::Right(coord)) => assert!($oob.contains(&coord)),
                    _ => panic!("Unexpected variant")
                }
            }

            // Sparse-dense
            {
                let result = compare_matrices(&$sparse, &$dense, &ExactElementwiseComparator);
                let err = result.unwrap_err();
                match err {
                    SparseEntryOutOfBounds(Entry::Left(coord)) => assert!($oob.contains(&coord)),
                    _ => panic!("Unexpected variant")
                }
            }
        };
    }

    // Row out of bounds
    {
        let dense = mock_matrix![1, 2, 3;
                                 4, 5, 6];
        let sparse = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3), (1, 2, 6), (2, 0, 1)]);

        let oob = vec![(2, 0)];
        assert_out_of_bounds_detected!(dense, sparse, oob);
    }

    // Col out of bounds
    {
        let dense = mock_matrix![1, 2, 3;
                                 4, 5, 6];
        let sparse = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3), (1, 3, 1), (1, 2, 6)]);
        let oob = vec![(1, 3)];
        assert_out_of_bounds_detected!(dense, sparse, oob);
    }

    // Row and col out of bounds
    {
        let dense = mock_matrix![1, 2, 3;
                                 4, 5, 6];
        let sparse = MockSparseMatrix::from_triplets(
            2,
            3,
            vec![(2, 3, 1), (0, 1, -3), (2, 0, 1), (1, 2, 6)],
        );
        let oob = vec![(2, 0), (2, 3)];
        assert_out_of_bounds_detected!(dense, sparse, oob);
    }
}

#[test]
fn dense_sparse_duplicate_entries() {
    use MatrixComparisonFailure::DuplicateSparseEntry;

    let dense = mock_matrix![1, 2, 3;
                             4, 5, 6];
    let sparse = MockSparseMatrix::from_triplets(2, 3, vec![(0, 1, -3),
                                                            (1, 0, 6),
                                                            (1, 0, 3),
                                                            (1, 2, 1)]);

    // Dense-sparse
    {
        let result = compare_matrices(&dense, &sparse, &ExactElementwiseComparator);
        let err = result.unwrap_err();
        match err {
            DuplicateSparseEntry(Entry::Right(coord)) => assert_eq!(coord, (1, 0)),
            _ => panic!("Unexpected error")
        }
    }

    // Sparse-dense
    {
        let result = compare_matrices(&sparse, &dense, &ExactElementwiseComparator);
        let err = result.unwrap_err();
        match err {
            DuplicateSparseEntry(Entry::Left(coord)) => assert_eq!(coord, (1, 0)),
            _ => panic!("Unexpected error")
        }
    }
}

/// A strategy producing pairs of dense and sparse matrices with the same dimensions.
fn same_size_dense_sparse_matrices(
) -> impl Strategy<Value = (MockDenseMatrix<i64>, MockSparseMatrix<i64>)> {
    let rows = MATRIX_DIM_RANGE;
    let cols = MATRIX_DIM_RANGE;

    (rows, cols).prop_flat_map(|(r, c)| {
        let dense = dense_matrix_strategy_i64(Just(r), Just(c));
        let sparse = sparse_matrix_strategy_i64(Just(r), Just(c));
        (dense, sparse)
    })
}

/// Given the dimensions of a matrix, produces a strategy that generates values
/// that are all out of bounds.
///
/// More precisely, generates values (r, c) such that r >= rows || c >= cols.
fn out_of_bounds_strategy(rows: usize, cols: usize) -> impl Strategy<Value = (usize, usize)> {
    let max_row_idx = 3 * rows + 1;
    let max_col_idx = 3 * cols + 1;
    let r_lower = rows;
    let c_lower = cols;
    (0..=2)
        .prop_flat_map(move |choice| {
            match choice {
                // Row out of bounds (if cols == 0, then col must also be out of bounds)
                0 if cols > 0 => (r_lower..max_row_idx, 0..cols),
                // Column out of bounds (if rows == 0, then row must also be out of bounds)
                1 if rows > 0 => (0..rows, c_lower..max_col_idx),
                // Both row and column out of bounds
                _ => (r_lower..max_row_idx, c_lower..max_col_idx),
            }
        })
        .prop_map(move |(r, c)| {
            // Sanity check: Either row or col index must be out of bounds
            assert!(r >= rows || c >= cols);
            (r, c)
        })
}

/// A strategy producing sparse matrices with at least one index out of bounds.
///
/// Additionally returns the triplets that are out of bounds that were added to the matrix.
fn sparse_matrix_out_of_bounds_strategy(
) -> impl Strategy<Value = (MockSparseMatrix<i64>, Vec<(usize, usize, i64)>)> {
    (
        sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        1usize..5,
    )
        .prop_flat_map(|(matrix, num_out_of_bounds)| {
            let r = matrix.rows();
            let c = matrix.cols();
            let out_of_bounds_triplets =
                (out_of_bounds_strategy(r, c), i64_range()).prop_map(|((r, c), v)| (r, c, v));
            (
                Just(matrix),
                proptest::collection::vec(out_of_bounds_triplets, num_out_of_bounds),
            )
        })
        .prop_flat_map(|(matrix, additional_triplets)| {
            let (r, c) = (matrix.rows(), matrix.cols());
            let mut triplets = matrix.take_triplets();
            triplets.extend(&additional_triplets);

            Just(triplets).prop_shuffle().prop_map(move |triplets| {
                (
                    MockSparseMatrix::from_triplets(r, c, triplets),
                    additional_triplets.clone(),
                )
            })
        })
}

fn dense_sparse_out_of_bounds_pair_strategy() -> impl Strategy<
    Value = (
        MockDenseMatrix<i64>,
        MockSparseMatrix<i64>,
        Vec<(usize, usize, i64)>,
    ),
> {
    sparse_matrix_out_of_bounds_strategy()
        .prop_flat_map(|(sparse, triplets)| {
            let (r, c) = (sparse.rows(), sparse.cols());
            (
                Just(sparse),
                Just(triplets),
                dense_matrix_strategy_i64(Just(r), Just(c)),
            )
        })
        .prop_map(|(sparse, triplets, dense)| (dense, sparse, triplets))
}

proptest! {
    #[test]
    fn sparse_dense_self_comparison_succeeds_i64(
        sparse in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        prop_assert!(compare_matrices(&sparse, &sparse.to_dense().unwrap(), &c).is_ok());
    }

    #[test]
    fn sparse_and_dense_matrices_indices_out_of_bounds_are_detected(
        (dense, sparse, out_of_bounds_triplets) in dense_sparse_out_of_bounds_pair_strategy()
    ) {
        use MatrixComparisonFailure::SparseEntryOutOfBounds;
        let c = ExactElementwiseComparator;

        let mut out_of_bounds_indices: Vec<_> = out_of_bounds_triplets
            .into_iter()
            .map(|(i, j, _)| (i, j))
            .collect();
        out_of_bounds_indices.sort();

        // Compare dense-sparse
        {
            let result = compare_matrices(&dense, &sparse, &c);
            let err = result.unwrap_err();
            match err {
                SparseEntryOutOfBounds(Entry::Right(coord))
                    => prop_assert!(out_of_bounds_indices.contains(&coord)),
                _ => prop_assert!(false)
            }
        }

        // Compare sparse-dense
        {
            let result = compare_matrices(&sparse, &dense, &c);
            let err = result.unwrap_err();
            match err {
                SparseEntryOutOfBounds(Entry::Left(coord))
                    => prop_assert!(out_of_bounds_indices.contains(&coord)),
                _ => prop_assert!(false)
            }
        }
    }

    #[test]
    fn sparse_and_dense_matrices_should_compare_the_same_as_dense_dense_i64(
        dense in dense_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&dense, &sparse, &c),
                        compare_matrices(&dense, sparse.to_dense().unwrap(), &c));
    }

    #[test]
    fn same_size_sparse_and_dense_matrices_should_compare_the_same_as_dense_dense_i64(
        (dense, sparse) in same_size_dense_sparse_matrices()
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&dense, &sparse, &c),
                        compare_matrices(&dense, sparse.to_dense().unwrap(), &c));
    }

    #[test]
    fn sparse_and_dense_matrices_should_compare_the_same_as_dense_dense_f64(
        dense in dense_matrix_strategy_normal_f64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse in sparse_matrix_strategy_normal_f64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        prop_assert_eq!(compare_matrices(&dense, &sparse, &c),
                        compare_matrices(&dense, sparse.to_dense().unwrap(), &c));
    }

    #[test]
    fn sparse_dense_comparison_is_symmetric_for_all_matrices_i64(
        dense in dense_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE),
        sparse in sparse_matrix_strategy_i64(MATRIX_DIM_RANGE, MATRIX_DIM_RANGE)
    ) {
        let c = ExactElementwiseComparator;
        let result1 = compare_matrices(&dense, &sparse, &c);
        let result2 = compare_matrices(&sparse, &dense, &c);

        prop_assert_eq!(result1.clone(), result2.clone().map_err(|err| err.reverse()));
        prop_assert_eq!(result1.map_err(|err| err.reverse()), result2);
    }

    #[test]
    fn same_size_sparse_dense_comparison_is_symmetric_for_all_matrices_i64(
        (dense, sparse) in same_size_dense_sparse_matrices()
    ) {
        let c = ExactElementwiseComparator;
        let result1 = compare_matrices(&dense, &sparse, &c);
        let result2 = compare_matrices(&sparse, &dense, &c);

        prop_assert_eq!(result1.clone(), result2.clone().map_err(|err| err.reverse()));
        prop_assert_eq!(result1.map_err(|err| err.reverse()), result2);
    }
}
