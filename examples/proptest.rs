//! An example that shows how to use the `proptest`-enabled macros.

#[cfg(test)]
mod tests {
    use matrixcompare::{prop_assert_matrix_eq, prop_assert_scalar_eq};
    use matrixcompare_mock::{dense_matrix_strategy, MockDenseMatrix};
    use proptest::prelude::*;

    #[cfg(test)]
    fn matrix() -> impl Strategy<Value = MockDenseMatrix<f64>> {
        dense_matrix_strategy(2..=5usize, 3..=5usize, 0.5..1.5)
    }

    proptest! {
        #[test]
        fn false_matrix_assertion_default_args((a, b) in (matrix(), matrix())) {
            prop_assert_matrix_eq!(a, b);
        }

        #[test]
        fn false_matrix_assertion_comp_abs((a, b) in (matrix(), matrix())) {
            prop_assert_matrix_eq!(a, b, comp = abs, tol=1e-12);
        }

        #[test]
        fn false_scalar_asssertion_comp_abs((a, b) in (0.5 .. 1.5f64, 0.5 .. 1.5f64)) {
            prop_assert_scalar_eq!(a, b, comp = abs, tol=1e-12);
        }
    }
}

fn main() {
    // The main entry point does nothing. In order to run this example, you need to run it with
    // `cargo test`, i.e. `cargo test --example proptest`.
}
