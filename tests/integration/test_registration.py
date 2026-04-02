import subprocess
import sys

import numpy as np
import pytest

from dualpy.core import (
    FUNC_REGISTRY,
    UFUNC_REGISTRY,
    DualArray,
    register_func,
    register_ufunc,
)

pytestmark = pytest.mark.integration


class TestUfuncRegistry:
    def test_add_is_registered(self):
        assert "add" in UFUNC_REGISTRY

    def test_multiply_is_registered(self):
        assert "multiply" in UFUNC_REGISTRY

    def test_sin_is_registered(self):
        assert "sin" in UFUNC_REGISTRY

    def test_exp_is_registered(self):
        assert "exp" in UFUNC_REGISTRY

    def test_register_custom_ufunc(self):
        @register_ufunc("_test_custom_ufunc")
        def custom(*inputs, **kwargs):
            return inputs[0]

        assert "_test_custom_ufunc" in UFUNC_REGISTRY

    def test_registry_keys_are_strings(self):
        for key in UFUNC_REGISTRY:
            assert isinstance(key, str)


class TestFuncRegistry:
    def test_array_is_registered(self):
        assert "array" in FUNC_REGISTRY

    def test_register_custom_func(self):
        @register_func("_test_custom_func")
        def custom(*args, **kwargs):
            return args[0]

        assert "_test_custom_func" in FUNC_REGISTRY

    def test_registry_keys_are_strings(self):
        for key in FUNC_REGISTRY:
            assert isinstance(key, str)


class TestUnregisteredOps:
    def test_unregistered_ufunc_raises(self):
        da = DualArray(np.array([1, 2]), np.array([1, 1]))
        with pytest.raises(NotImplementedError, match="not implemented"):
            np.bitwise_and(da, da)

    def test_unregistered_array_function_raises(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        with pytest.raises(NotImplementedError, match="not implemented"):
            da.__array_function__(np.histogram, (DualArray,), (da,), {})


class TestImportRegistration:
    def test_import_dualpy_registers_ufuncs(self):
        code = (
            "import dualpy; import numpy as np; "
            "from dualpy.core import DualArray; "
            "da = DualArray(np.array(1.0), np.array(1.0)); "
            "r = np.sin(da); "
            "assert isinstance(r, DualArray)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_import_dualpy_registers_routines(self):
        code = (
            "import dualpy; import numpy as np; "
            "from dualpy.core import DualArray; "
            "da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2])); "
            "r = np.sum(da); "
            "assert isinstance(r, DualArray)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestComprehensiveRegistration:
    def test_all_expected_ufuncs_registered(self):
        expected = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "power",
            "matmul",
            "negative",
            "absolute",
            "square",
            "cbrt",
            "reciprocal",
            "maximum",
            "minimum",
            "positive",
            "float_power",
            "fabs",
            "remainder",
            "fmod",
            "copysign",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "arctan2",
            "hypot",
            "degrees",
            "rad2deg",
            "radians",
            "deg2rad",
            "exp",
            "log",
            "sqrt",
            "exp2",
            "expm1",
            "log2",
            "log10",
            "log1p",
            "logaddexp",
            "logaddexp2",
            "sinh",
            "cosh",
            "tanh",
            "arcsinh",
            "arccosh",
            "arctanh",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "equal",
            "not_equal",
            "sign",
            "heaviside",
            "floor",
            "ceil",
            "trunc",
            "rint",
            "logical_and",
            "logical_or",
            "logical_xor",
            "logical_not",
            "isnan",
            "isinf",
            "isfinite",
            "signbit",
        ]
        for name in expected:
            assert name in UFUNC_REGISTRY, f"ufunc '{name}' not registered"

    def test_all_expected_funcs_registered(self):
        expected = [
            "sum",
            "mean",
            "prod",
            "max",
            "min",
            "var",
            "std",
            "cumsum",
            "cumprod",
            "argmax",
            "argmin",
            "nansum",
            "nanmean",
            "nanvar",
            "nanstd",
            "nanmax",
            "nanmin",
            "average",
            "median",
            "all",
            "any",
            "count_nonzero",
            "concatenate",
            "stack",
            "reshape",
            "transpose",
            "dot",
            "inner",
            "outer",
            "tensordot",
            "einsum",
            "trace",
            "cross",
            "vdot",
            "kron",
            "norm",
            "det",
            "inv",
            "solve",
            "cholesky",
            "eigh",
            "svd",
            "qr",
            "lstsq",
            "matrix_power",
            "multi_dot",
            "where",
            "clip",
            "sort",
            "argsort",
            "searchsorted",
            "nonzero",
            "flatnonzero",
            "extract",
            "zeros_like",
            "ones_like",
            "full_like",
            "copy",
            "empty_like",
            "diag",
            "diagonal",
            "meshgrid",
            "triu",
            "tril",
            "atleast_1d",
            "atleast_2d",
            "atleast_3d",
            "column_stack",
            "pad",
            "rot90",
            "take",
            "take_along_axis",
            "insert",
            "delete",
            "append",
            "select",
            "diff",
            "convolve",
            "correlate",
            "interp",
            "trapezoid",
            "sinc",
            "gradient",
        ]
        for name in expected:
            assert name in FUNC_REGISTRY, f"func '{name}' not registered"
