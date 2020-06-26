use std::fmt;

use crate::comparators::ElementwiseComparator;

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarComparisonFailure<T, E> {
    pub left: T,
    pub right: T,
    pub error: E,
    pub comparator_description: String,
}

impl<T, E> fmt::Display for ScalarComparisonFailure<T, E>
where
    T: fmt::Display,
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Scalars x and y do not compare equal.")?;
        writeln!(f)?;
        write!(f, "x = {x}, y = {y}. ", x = self.left, y = self.right)?;
        writeln!(f, "{}", self.error)?;
        writeln!(f)?;
        writeln!(f, "Comparison criterion: {}", self.comparator_description)
    }
}

/// Comparison of two scalars.
pub fn compare_scalars<T, C>(
    left: &T,
    right: &T,
    comparator: C,
) -> Result<(), ScalarComparisonFailure<T, C::Error>>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    comparator
        .compare(left, right)
        .map_err(|error| ScalarComparisonFailure {
            comparator_description: comparator.description(),
            left: left.clone(),
            right: right.clone(),
            error,
        })
}
