use std::fmt;

use crate::comparators::ElementwiseComparator;

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarComparisonFailure<T, E> {
    pub x: T,
    pub y: T,
    pub error: E,
    pub comparator_description: String
}

impl<T, E> fmt::Display for ScalarComparisonFailure<T, E>
where
    T: fmt::Display,
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Scalars x and y do not compare equal.")?;
        writeln!(f)?;
        write!(f, "x = {x}, y = {y}. ", x = self.x, y = self.y)?;
        writeln!(f, "{}", self.error)?;
        writeln!(f)?;
        writeln!(f, "Comparison criterion: {}", self.comparator_description)
    }
}

pub fn compare_scalars<T, C>(x: &T, y: &T, comparator: C) -> Result<(), ScalarComparisonFailure<T, C::Error>>
where
    T: Clone,
    C: ElementwiseComparator<T>,
{
    comparator.compare(x, y).map_err(|error|
        ScalarComparisonFailure {
            comparator_description: comparator.description(),
            x: x.clone(),
            y: y.clone(),
            error,
        }
    )
}
