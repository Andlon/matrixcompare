//! Tools for integrating `matrixcompare` with `proptest`.
//!
//! In order to use this module, you need to enable the `proptest-support` feature.

/// Internal macro.
#[macro_export]
#[doc(hidden)]
macro_rules! build_proptest_message {
    ($failure:expr) => {
        format!(
            "Comparison failure at {}:{}. Error:\n {}",
            file!(),
            line!(),
            $failure
        );
    };
}

/// A version of `assert_matrix_eq` suitable for use in `proptest` property-based tests.
///
/// Works exactly as `assert_matrix_eq`, except that instead of causing a panic,
/// it returns an error compatible with property-based tests from the `proptest` crate.
///
/// Requires the `proptest-support` feature to be enabled.
#[macro_export]
macro_rules! prop_assert_matrix_eq {
    ($($args:tt)*) => {
        let failure_handler = |msg| {
            // Add filename and line numbers to message (since we don't panic, it's useful
            // to have this information in the output).
            let amended_message = $crate::build_proptest_message!(msg);
            return ::core::result::Result::Err(
                ::proptest::test_runner::TestCaseError::fail(amended_message));
        };
        $crate::base_matrix_eq!(failure_handler, $($args)*);
    }
}

/// A version of `assert_scalar_eq` suitable for use in `proptest` property-based tests.
///
/// Works exactly as `assert_scalar_eq`, except that instead of causing a panic,
/// it returns an error compatible with property-based tests from the `proptest` crate.
///
/// Requires the `proptest-support` feature to be enabled.
#[macro_export]
macro_rules! prop_assert_scalar_eq {
    ($($args:tt)*) => {
        let failure_handler = |msg| {
            // Add filename and line numbers to message (since we don't panic, it's useful
            // to have this information in the output).
            let amended_message = $crate::build_proptest_message!(msg);
            return ::core::result::Result::Err(
                ::proptest::test_runner::TestCaseError::fail(amended_message));
        };
        $crate::base_scalar_eq!(failure_handler, $($args)*);
    }
}
