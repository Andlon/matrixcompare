//#[macro_use]
//mod assert_matrix_eq;
//
//#[macro_use]
//mod assert_vector_eq;
//
#[macro_use]
mod assert_scalar_eq;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

mod comparison;
mod ulp;

pub use self::comparison::{
    AbsoluteElementwiseComparator,
    ExactElementwiseComparator,
    UlpElementwiseComparator,
    FloatElementwiseComparator,

    // The following is just imported because we want to
    // expose trait bounds in the documentation
    ElementwiseComparator
};

//pub use self::assert_matrix_eq::elementwise_matrix_comparison;
//pub use self::assert_vector_eq::elementwise_vector_comparison;
pub use self::assert_scalar_eq::scalar_comparison;
