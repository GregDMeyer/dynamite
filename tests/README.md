
## dynamite tests

This directory contains both unit and integration tests for dynamite.

### Unit tests

The unit tests are designed to be fast and self-contained. To run them all at once navigate into the `unit` directory and run

```
python -m unittest discover .
```

### Integration tests

The integration tests are designed to make sure everything is operating correctly together, when run in realistic scenarios (under MPI, etc.). The easiest way to run them all is to navigate into the `integration` directory and run

```
python run_all_tests.py
```

This script has a number of command-line options; simply run it with the `-h` flag to see them.

The integration tests are designed to run under MPI; they require the `mpi4py` package if they are run with more than 1 rank.
