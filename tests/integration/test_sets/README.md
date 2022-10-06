
## Test set files

This directory contains text files defining lists of tests, which might be useful to run together. For example, one might define a certain set of tests which run within the computational resources of a particular compute node, for a given system size.  Users can define their own test sets, as described below.

### Test set file format

The file should contain a list of tests, one per line. Empty lines and lines beginning with a `#` are ignored.  Also, any text following the character `#` in a line is ignored.

The tests should be listed as a filename, optionally followed by a space and the name of a TestCase, again optionally followed by a period and the name of a specific test.

Consider the following examples:

```
# run all tests in the file test_multiply.py
test_multiply.py

# run all tests in the TestCase "Subspaces", from the file test_multiply.py
test_multiply.py Subspaces

# run only the test "Subspaces.test_parity_XX_even" from test_multiply.py
test_multiply.py Subspaces.test_parity_XX_even
```
