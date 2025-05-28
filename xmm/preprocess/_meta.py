import sympy as sp


class Metadata:
    call_map = {
        "exp": sp.exp,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan2": sp.atan2,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "asinh": sp.asinh,
        "acosh": sp.acosh,
        "atanh": sp.atanh,
    }

    supported_calls = list(call_map.keys())

    CUDA_fn_map = {
        "exp": "expf",
        "log": "logf",
        "sin": "sinf",
        "cos": "cosf",
        "tan": "tanf",
        "asin": "asinf",
        "acos": "acosf",
        "atan2": "atan2f",
        "sinh": "sinhf",
        "cosh": "coshf",
        "tanh": "tanhf",
        "asinh": "asinhf",
        "acosh": "acoshf",
        "atanh": "atanhf",
    }
