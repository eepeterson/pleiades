import numpy as np

_RTOL = 0.
_ATOL = 1.E-12

# For setters implement check_type, check_value, flag_greens
# For getters implement flag_greens
# for functions implement check_type, check_value, flag_greens


def flag_greens_on_get(func):
    def wrapper(obj):
        if not obj._uptodate:
            obj._compute_greens()
        return func(obj)
    return wrapper


def flag_greens_on_set(func):
    """Decorator to signal Green's functions are now out of date."""
    def setter_wrapper(obj, value):
        retval = func(obj, value)
        obj._uptodate = False
        return retval
    return setter_wrapper


def flag_greens_on_transform(ref_val):
    """Determines if Green's functions need to be recomputed.

    Checks value of current against the value of future and decides whether
    or not to set self._uptodate to True or False. This function is intended
    to make the user experience smoother while preserving performance by
    caching the Green's functions. Variables that require recomputing
    Green's functions include any positional variables or weights.

    Parameters
    ----------
    current :
        The current value
    future :
        The future value being set
    """
    def actual_decorator(func):
        def transform_wrapper(obj, value, **kwargs):
            obj._uptodate = np.allclose(ref_val, value, rtol=0., atol=_ATOL)
            return func(obj, value, **kwargs)
        return transform_wrapper
    return actual_decorator
