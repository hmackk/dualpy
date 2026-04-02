import subprocess
import sys

import numpy as np
import pytest
from dualpy.core import DualArray

pytestmark = pytest.mark.unit


class TestConstruction:
    def test_primal_only_tangent_defaults_to_zeros(self):
        da = DualArray(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(da.tangent, np.array([0.0, 0.0]))

    def test_primal_and_tangent(self):
        p = np.array([3.0, 4.0])
        t = np.array([0.5, -0.5])
        da = DualArray(p, t)
        np.testing.assert_array_equal(da.primal, p)
        np.testing.assert_array_equal(da.tangent, t)

    def test_reject_non_broadcastable_shapes(self):
        with pytest.raises(ValueError):
            DualArray(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_accept_broadcastable_shapes(self):
        da = DualArray(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0]))
        assert da.primal.shape == (2, 2)
        assert da.tangent.shape == (2,)

    def test_scalar_construction(self):
        da = DualArray(np.array(5.0), np.array(1.0))
        assert da.primal == 5.0
        assert da.tangent == 1.0


class TestRepr:
    def test_repr_contains_values(self):
        da = DualArray(np.array(3.0), np.array(1.0))
        r = repr(da)
        assert "DualArray" in r
        assert "3." in r
        assert "1." in r


class TestProperties:
    def test_shape(self, vector_dual):
        assert vector_dual.shape == (3,)

    def test_dtype(self, vector_dual):
        assert vector_dual.dtype == np.float64

    def test_size(self, vector_dual):
        assert vector_dual.size == 3

    def test_ndim_vector(self, vector_dual):
        assert vector_dual.ndim == 1

    def test_ndim_matrix(self, matrix_dual):
        assert matrix_dual.ndim == 2

    def test_transpose(self, matrix_dual):
        t = matrix_dual.T
        np.testing.assert_array_equal(t.primal, matrix_dual.primal.T)
        np.testing.assert_array_equal(t.tangent, matrix_dual.tangent.T)


class TestIndexing:
    def test_getitem_scalar_index(self, vector_dual):
        elem = vector_dual[0]
        assert isinstance(elem, DualArray)
        assert elem.primal == 1.0
        assert elem.tangent == 0.5

    def test_getitem_slice(self, vector_dual):
        sliced = vector_dual[:2]
        assert isinstance(sliced, DualArray)
        np.testing.assert_array_equal(sliced.primal, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(sliced.tangent, np.array([0.5, -1.0]))

    def test_setitem_dual_array(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        da[0] = DualArray(np.array(9.0), np.array(0.9))
        assert da.primal[0] == 9.0
        assert da.tangent[0] == 0.9

    def test_setitem_plain_scalar(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        da[0] = 9.0
        assert da.primal[0] == 9.0
        assert da.tangent[0] == 0.0

    def test_delitem(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        del da[1]
        np.testing.assert_array_equal(da.primal, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(da.tangent, np.array([0.1, 0.3]))


class TestDunders:
    def test_neg(self, scalar_dual):
        result = -scalar_dual
        assert result.primal == -2.0
        assert result.tangent == -1.0

    def test_abs(self, scalar_dual):
        result = abs(scalar_dual)
        assert result.primal == 2.0

    def test_len(self, vector_dual):
        assert len(vector_dual) == 3


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
