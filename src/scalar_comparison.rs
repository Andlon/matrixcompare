use std::fmt;

use crate::comparators::ComparisonFailure;
use crate::ElementwiseComparator;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ScalarComparisonFailure<T, E>
where
    E: ComparisonFailure,
{
    pub x: T,
    pub y: T,
    pub error: E,
}

impl<T, E> fmt::Display for ScalarComparisonFailure<T, E>
where
    T: fmt::Display,
    E: ComparisonFailure,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "x = {x}, y = {y}.{reason}",
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
pub enum ScalarComparisonResult<T, C>
where
    C: ElementwiseComparator<T>
{
    Match,
    Mismatch {
        comparator: C,
        mismatch: ScalarComparisonFailure<T, C::Error>,
    },
}

impl<T, C> ScalarComparisonResult<T, C>
where
    T: fmt::Display,
    C: ElementwiseComparator<T>
{
    pub fn panic_message(&self) -> Option<String> {
        match self {
            &ScalarComparisonResult::Mismatch {
                ref comparator,
                ref mismatch,
            } => Some(format!(
                "\n
Scalars x and y do not compare equal.

{mismatch}

Comparison criterion: {description}
\n",
                description = comparator.description(),
                mismatch = mismatch.to_string()
            )),
            _ => None,
        }
    }
}

pub fn compare_scalars<T, C>(x: &T, y: &T, comparator: C) -> ScalarComparisonResult<T, C>
where
    T: Clone,
    C: ElementwiseComparator<T>
{
    match comparator.compare(x, y) {
        Err(error) => ScalarComparisonResult::Mismatch {
            comparator,
            mismatch: ScalarComparisonFailure {
                x: x.clone(),
                y: y.clone(),
                error,
            },
        },
        _ => ScalarComparisonResult::Match,
    }
}

#[cfg(test)]
mod tests {
    use crate::comparators::{ExactElementwiseComparator, ExactError};
    use crate::compare_scalars;
    use crate::assert_scalar_eq;

    #[test]
    fn scalar_comparison_reports_correct_mismatch() {
        use super::ScalarComparisonFailure;
        use super::ScalarComparisonResult::Mismatch;

        let comp = ExactElementwiseComparator;

        {
            let x = 0.2;
            let y = 0.3;

            let expected = Mismatch {
                comparator: comp,
                mismatch: ScalarComparisonFailure {
                    x: 0.2,
                    y: 0.3,
                    error: ExactError,
                },
            };

            assert_eq!(compare_scalars(&x, &y, comp), expected);
        }
    }

    #[test]
    pub fn scalar_eq_default_compare_self_for_integer() {
        let x = 2;
        assert_scalar_eq!(x, x);
    }

    #[test]
    pub fn scalar_eq_default_compare_self_for_floating_point() {
        let x = 2.0;
        assert_scalar_eq!(x, x);
    }

    #[test]
    #[should_panic]
    pub fn scalar_eq_default_mismatched_elements() {
        let x = 3;
        let y = 4;
        assert_scalar_eq!(x, y);
    }

    #[test]
    pub fn scalar_eq_exact_compare_self_for_integer() {
        let x = 2;
        assert_scalar_eq!(x, x, comp = exact);
    }

    #[test]
    pub fn scalar_eq_exact_compare_self_for_floating_point() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = exact);;
    }

    #[test]
    #[should_panic]
    pub fn scalar_eq_exact_mismatched_elements() {
        let x = 3;
        let y = 4;
        assert_scalar_eq!(x, y, comp = exact);
    }

    #[test]
    pub fn scalar_eq_abs_compare_self_for_integer() {
        let x = 2;
        assert_scalar_eq!(x, x, comp = abs, tol = 1);
    }

    #[test]
    pub fn scalar_eq_abs_compare_self_for_floating_point() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = abs, tol = 1e-8);
    }

    #[test]
    #[should_panic]
    pub fn scalar_eq_abs_mismatched_elements() {
        let x = 3.0;
        let y = 4.0;
        assert_scalar_eq!(x, y, comp = abs, tol = 1e-8);
    }

    #[test]
    pub fn scalar_eq_ulp_compare_self() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = ulp, tol = 1);
    }

    #[test]
    #[should_panic]
    pub fn scalar_eq_ulp_mismatched_elements() {
        let x = 3.0;
        let y = 4.0;
        assert_scalar_eq!(x, y, comp = ulp, tol = 4);
    }

    #[test]
    pub fn scalar_eq_float_compare_self() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = ulp, tol = 1);
    }

    #[test]
    #[should_panic]
    pub fn scalar_eq_float_mismatched_elements() {
        let x = 3.0;
        let y = 4.0;
        assert_scalar_eq!(x, y, comp = float);
    }

    #[test]
    pub fn scalar_eq_float_compare_self_with_eps() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = float, eps = 1e-6);
    }

    #[test]
    pub fn scalar_eq_float_compare_self_with_ulp() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = float, ulp = 12);
    }

    #[test]
    pub fn scalar_eq_float_compare_self_with_eps_and_ulp() {
        let x = 2.0;
        assert_scalar_eq!(x, x, comp = float, eps = 1e-6, ulp = 12);
        assert_scalar_eq!(x, x, comp = float, ulp = 12, eps = 1e-6);
    }

    #[test]
    pub fn scalar_eq_pass_by_ref() {
        let x = 0.0;

        // Exercise all the macro definitions and make sure that we are able to call it
        // when the arguments are references.
        assert_scalar_eq!(&x, &x);
        assert_scalar_eq!(&x, &x, comp = exact);
        assert_scalar_eq!(&x, &x, comp = abs, tol = 0.0);
        assert_scalar_eq!(&x, &x, comp = ulp, tol = 0);
        assert_scalar_eq!(&x, &x, comp = float);
        assert_scalar_eq!(&x, &x, comp = float, eps = 0.0, ulp = 0);
    }
}
