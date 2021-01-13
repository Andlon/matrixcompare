matrixcompare
=============
!["CI badge"](https://github.com/Andlon/matrixcompare/workflows/Build%20and%20run%20tests/badge.svg)

matrixcompare is a utility library for comparing matrices (dense or sparse)
for testing/debugging purposes. To that effect, it provides functions
and assertions for comparing matrices with exact or approximate equality.
The metric used for approximate equality is configurable and packaged into
a convenient API. matrixcompare does not provide any matrices of its own,
but is instead intended to be integrated into libraries that provide these
kind of data structures.

Please see the [documentation](https://docs.rs/matrixcompare) for more information,
or the [changelog](CHANGELOG.md) for a list of changes in recent releases.

Contributing
============

Contributions are welcome! For any larger contribution, it is however advisable to write up
your ideas in an issue before you commence work. This is particularly true for contributions
which would significantly increase the scope of the project.

License
=======

`matrixcompare` is licensed under the MIT license.
See the `LICENSE` file in the repository for the exact license.