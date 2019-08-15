/// Compare matrices for exact or approximate equality.
///
/// The `assert_matrix_eq!` simplifies the comparison of two matrices by
/// providing the following features:
///
/// - Verifies that the dimensions of the matrices match.
/// - Offers both exact and approximate comparison of individual elements.
/// - Multiple types of comparators available, depending on the needs of the user.
/// - Built-in error reporting makes it easy to determine which elements of the two matrices
///   that do not compare equal.
///
/// # Usage
/// Given two matrices `x` and `y`, the default invocation performs an exact elementwise
/// comparison of the two matrices.
///
/// ```
/// # use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
/// # let x = mock_matrix![1.0f64]; let y = mock_matrix![1.0f64];
/// // Performs elementwise exact comparison
/// assert_matrix_eq!(x, y);
/// ```
///
/// An exact comparison is often not desirable. In particular, with floating point types,
/// rounding errors or other sources of inaccuracies tend to complicate the matter.
/// For this purpose, `assert_matrix_eq!` provides several comparators.
///
/// ```
/// # use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
/// # let x = mock_matrix![1.0f64]; let y = mock_matrix![1.0f64];
/// // Available comparators:
/// assert_matrix_eq!(x, y, comp = exact);
/// assert_matrix_eq!(x, y, comp = float);
/// assert_matrix_eq!(x, y, comp = abs, tol = 1e-12);
/// assert_matrix_eq!(x, y, comp = ulp, tol = 8);
/// ```
/// **Note**: The `comp` argument *must* be specified after `x` and `y`, and cannot come
/// after comparator-specific options. This is a deliberate design decision,
/// with the rationale that assertions should look as uniform as possible for
/// the sake of readability.
///
///
/// ### The `exact` comparator
/// This comparator simply uses the default `==` operator to compare each pair of elements.
/// The default comparator delegates the comparison to the `exact` comparator.
///
/// ### The `float` comparator
/// The `float` comparator is designed to be a conservative default for comparing floating-point numbers.
/// It is inspired by the `AlmostEqualUlpsAndAbs` comparison function proposed in the excellent blog post
/// [Comparing Floating Point Numbers, 2012 Edition]
/// (https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/)
/// by Bruce Dawson.
///
/// If you expect the two matrices to be almost exactly the same, but you want to leave some
/// room for (very small) rounding errors, then this comparator should be your default choice.
///
/// The comparison criterion can be summarized as follows:
///
/// 1. If `assert_matrix_eq!(x, y, comp = abs, tol = max_eps)` holds for `max_eps` close to the
///    machine epsilon for the floating point type,
///    then the comparison is successful.
/// 2. Otherwise, returns the result of `assert_matrix_eq!(x, y, comp = ulp, tol = max_ulp)`,
///    where `max_ulp` is a small positive integer constant.
///
/// The `max_eps` and `max_ulp` parameters can be tweaked to your preference with the syntax:
///
/// ```
/// # use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
/// # let x = mock_matrix![1.0f64]; let y = mock_matrix![1.0f64];
/// # let max_eps = 1.0; let max_ulp = 0;
/// assert_matrix_eq!(x, y, comp = float, eps = max_eps, ulp = max_ulp);
/// ```
///
/// These additional parameters can be specified in any order after the choice of comparator,
/// and do not both need to be present.
///
/// ### The `abs` comparator
/// Compares the absolute difference between individual elements against the specified tolerance.
/// Specifically, for every pair of elements x and y picked from the same row and column in X and Y
/// respectively, the criterion is defined by
///
/// ```text
///     | x - y | <= tol.
/// ```
///
/// In addition to floating point numbers, the comparator can also be used for integral numbers,
/// both signed and unsigned. In order to avoid unsigned underflow, the difference is always
/// computed by subtracting the smaller number from the larger number.
/// Note that the type of `tol` is required to be the same as that of the scalar field.
///
///
/// ### The `ulp` comparator
/// Elementwise comparison of floating point numbers based on their
/// [ULP](https://en.wikipedia.org/wiki/Unit_in_the_last_place) difference.
/// Once again, this is inspired by the proposals
/// [in the aforementioned blog post by Bruce Dawon]
/// (https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/),
/// but it handles some cases explicitly as to provide better error reporting.
///
/// Note that the ULP difference of two floating point numbers is not defined in the following cases:
///
/// - The two numbers have different signs. The only exception here is +0 and -0,
///   which are considered an exact match.
/// - One of the numbers is NaN.
///
/// ULP-based comparison is typically used when two numbers are expected to be very,
/// very close to each other. However, it is typically not very useful very close to zero,
/// which is discussed in the linked blog post above.
/// The error in many mathematical functions can often be bounded by a certain number of ULP, and so
/// this comparator is particularly useful if this number is known.
///
/// Note that the scalar type of the matrix must implement the [Ulp trait](ulp/trait.Ulp.html) in order
/// to be used with this comparator. By default, `f32` and `f64` implementations are provided.
///
/// # Error reporting
///
/// One of the main motivations for the `assert_matrix_eq!` macro is the ability to give
/// useful error messages which help pinpoint the problems. For example, consider the example
///
/// ```rust,should_panic
/// # use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
///
/// fn main() {
///     let a = mock_matrix![1.00, 2.00;
///                          3.00, 4.00];
///     let b = mock_matrix![1.01, 2.00;
///                          3.40, 4.00];
///     assert_matrix_eq!(a, b, comp = abs, tol = 1e-8);
/// }
/// ```
///
/// which yields the output
///
/// ```text
/// Matrices X and Y have 2 mismatched element pairs.
/// The mismatched elements are listed below, in the format
/// (row, col): x = X[[row, col]], y = Y[[row, col]].
///
/// (0, 0): x = 1, y = 1.01. Absolute error: 0.010000000000000009.
/// (1, 0): x = 3, y = 3.4. Absolute error: 0.3999999999999999.
///
/// Comparison criterion: absolute difference, |x - y| <= 0.00000001.
/// ```
///
/// # Trait bounds on elements
/// Each comparator has specific requirements on which traits the elements
/// need to implement. To discover which traits are required for each comparator,
/// we refer the reader to implementors of
/// [ElementwiseComparator](macros/trait.ElementwiseComparator.html),
/// which provides the underlying comparison for the various macro invocations.
///
/// # Examples
///
/// ```
/// # use matrixcompare::assert_matrix_eq; use matrixcompare_mock::mock_matrix;
///
/// let ref a = mock_matrix![1, 2;
///                          3, 4i64];
/// let ref b = mock_matrix![1, 3;
///                          3, 4i64];
///
/// let ref x = mock_matrix![1.000, 2.000,
///                          3.000, 4.000f64];
/// let ref y = mock_matrix![0.999, 2.001,
///                          2.998, 4.000f64];
///
/// // comp = abs is also applicable to integers
/// assert_matrix_eq!(a, b, comp = abs, tol = 1);
/// assert_matrix_eq!(x, y, comp = abs, tol = 0.01);
/// ```
#[macro_export]
macro_rules! assert_matrix_eq {
    ($x:expr, $y:expr) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::ExactElementwiseComparator;

            let comp = ExactElementwiseComparator;
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                // Note: We need the panic to incur here inside of the macro in order
                // for the line number to be correct when using it for tests,
                // hence we build the panic message in code, but panic here.
                if let Some(msg) = failure.panic_message() {
                    panic!("{msg}
Please see the documentation for ways to compare matrices approximately.\n\n",
                    msg = msg.trim_end());
                }
            }
        }
    };
    ($x:expr, $y:expr, comp = exact) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::ExactElementwiseComparator;

            let comp = ExactElementwiseComparator;
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                if let Some(msg) = failure.panic_message() {
                    panic!(msg);
                }
            }
        }
    };
    ($x:expr, $y:expr, comp = abs, tol = $tol:expr) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::AbsoluteElementwiseComparator;

            let comp = AbsoluteElementwiseComparator { tol: $tol };
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                if let Some(msg) = failure.panic_message() {
                    panic!(msg);
                }
            }
        }
    };
    ($x:expr, $y:expr, comp = ulp, tol = $tol:expr) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::UlpElementwiseComparator;

            let comp = UlpElementwiseComparator { tol: $tol };
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                if let Some(msg) = failure.panic_message() {
                    panic!(msg);
                }
            }
        }
    };
    ($x:expr, $y:expr, comp = float) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::FloatElementwiseComparator;

            let comp = FloatElementwiseComparator::default();
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                if let Some(msg) = failure.panic_message() {
                    panic!(msg);
                }
            }
        }
    };
    // This following allows us to optionally tweak the epsilon and ulp tolerances
    // used in the default float comparator.
    ($x:expr, $y:expr, comp = float, $($key:ident = $val:expr),+) => {
        {
            use $crate::{compare_matrices};
            use $crate::comparators::FloatElementwiseComparator;

            let comp = FloatElementwiseComparator::default()$(.$key($val))+;
            let result = compare_matrices(&$x, &$y, &comp);
            if let Err(failure) = result {
                if let Some(msg) = failure.panic_message() {
                    panic!(msg);
                }
            }
        }
    };
}

/// Compare scalars for exact or approximate equality.
///
/// This macro works analogously to [assert_matrix_eq!](macro.assert_matrix_eq.html),
/// but is used for comparing scalars (e.g. integers, floating-point numbers)
/// rather than matrices. Please see the documentation for `assert_matrix_eq!`
/// for details about issues that arise when comparing floating-point numbers,
/// as well as an explanation for how these macros can be used to resolve
/// these issues.
///
/// # Examples
///
/// ```
/// # use matrixcompare::{assert_scalar_eq};
/// let x = 3.00;
/// let y = 3.05;
/// // Assert that |x - y| <= 0.1
/// assert_scalar_eq!(x, y, comp = abs, tol = 0.1);
/// ```
#[macro_export]
macro_rules! assert_scalar_eq {
    ($x:expr, $y:expr) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::ExactElementwiseComparator;
            let comp = ExactElementwiseComparator;
            let msg = compare_scalars(&$x, &$y, comp).panic_message();
            if let Some(msg) = msg {
                // Note: We need the panic to incur here inside of the macro in order
                // for the line number to be correct when using it for tests,
                // hence we build the panic message in code, but panic here.
                panic!("{msg}
Please see the documentation for ways to compare scalars approximately.\n\n",
                    msg = msg.trim_end());
            }
        }
    };
    ($x:expr, $y:expr, comp = exact) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::ExactElementwiseComparator;
            let comp = ExactElementwiseComparator;
            let msg = compare_scalars(&$x, &$y, comp).panic_message();
            if let Some(msg) = msg {
                panic!(msg);
            }
        }
    };
    ($x:expr, $y:expr, comp = abs, tol = $tol:expr) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::AbsoluteElementwiseComparator;
            let comp = AbsoluteElementwiseComparator { tol: $tol.clone() };
            let msg = compare_scalars(&$x.clone(), &$y.clone(), comp).panic_message();
            if let Some(msg) = msg {
                panic!(msg);
            }
        }
    };
    ($x:expr, $y:expr, comp = ulp, tol = $tol:expr) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::UlpElementwiseComparator;
            let comp = UlpElementwiseComparator { tol: $tol.clone() };
            let msg = compare_scalars(&$x.clone(), &$y.clone(), comp).panic_message();
            if let Some(msg) = msg {
                panic!(msg);
            }
        }
    };
    ($x:expr, $y:expr, comp = float) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::FloatElementwiseComparator;
            let comp = FloatElementwiseComparator::default();
            let msg = compare_scalars(&$x.clone(), &$y.clone(), comp).panic_message();
            if let Some(msg) = msg {
                panic!(msg);
            }
        }
    };
    // The following allows us to optionally tweak the epsilon and ulp tolerances
    // used in the default float comparator.
    ($x:expr, $y:expr, comp = float, $($key:ident = $val:expr),+) => {
        {
            use $crate::{compare_scalars};
            use $crate::comparators::FloatElementwiseComparator;
            let comp = FloatElementwiseComparator::default()$(.$key($val))+;
            let msg = compare_scalars(&$x.clone(), &$y.clone(), comp).panic_message();
            if let Some(msg) = msg {
                panic!(msg);
            }
        }
    };
}
