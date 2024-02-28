#!/usr/bin/env python


class InputError(Exception):
    """Raised when there is an error in the input."""

    pass


class MissingDataError(Exception):
    """Raised when too many values are found in the input. Setting an inputter might fix this."""

    pass


class ConvergenceError(Exception):
    """Raised when the piecewise linear regression does not converge."""

    pass
