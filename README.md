matrixcompare
=============

matrixcompare is a utility library for comparing matrices (dense or sparse)
for testing/debugging purposes. To that effect, it provides functions
and assertions for comparing matrices with exact or approximate equality.
The metric used for approximate equality is configurable and packaged into
a convenient API. matrixcompare does not provide any matrices of its own,
but is instead intended to be integrated into libraries that provide these
kind of data structures.

matrixcompare guarantees that for two matrices `A` and `B`, the comparison
takes place in `O(log(nnz(A)) nnz(A) + log(nnz(B)) nnz(B))`, where `nnz(_)`
is the number of structural non-zeros of the matrix. For a dense matrix `A`
with `m` rows and `n` columns, `nnz(A) = m * n`.
Beyond this basic complexity guarantee, there is no emphasis
on performance, and instead the focus is on ease of use and integration.
Note that the logarithm in the complexity bound is a result of sorting
the test results in the event that there are errors.

TODO: More docs/readme