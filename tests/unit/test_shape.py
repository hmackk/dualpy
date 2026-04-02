import numpy as np
import pytest
from dualpy.core import DualArray

from tests.conftest import assert_dual_close

pytestmark = pytest.mark.unit


class TestConcatenate:
    def test_concatenate_dual_arrays(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.concatenate([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )

    def test_mixed_dual_and_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.concatenate([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.0, 0.0]),
        )


class TestStack:
    def test_stack_dual_arrays(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.stack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )

    def test_mixed_dual_and_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.stack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.0, 0.0]]),
        )


class TestReshape:
    def test_reshape_dual_array(self):
        da = DualArray(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )
        result = np.reshape(da, (2, 2))
        assert isinstance(result, DualArray)
        assert result.shape == (2, 2)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )


class TestTranspose:
    def test_2d(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.transpose(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 3.0], [2.0, 4.0]]),
            np.array([[0.1, 0.3], [0.2, 0.4]]),
        )


class TestSwapaxes:
    def test_basic(self):
        da = DualArray(
            np.arange(8.0).reshape(2, 2, 2),
            np.ones((2, 2, 2)) * 0.1,
        )
        result = np.swapaxes(da, 0, 2)
        assert isinstance(result, DualArray)
        assert result.shape == (2, 2, 2)
        np.testing.assert_allclose(result.primal, np.swapaxes(da.primal, 0, 2))


class TestSqueeze:
    def test_basic(self):
        da = DualArray(
            np.array([[[1.0, 2.0]]]),
            np.array([[[0.1, 0.2]]]),
        )
        result = np.squeeze(da)
        assert isinstance(result, DualArray)
        assert result.shape == (2,)
        assert_dual_close(result, np.array([1.0, 2.0]), np.array([0.1, 0.2]))


class TestExpandDims:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.expand_dims(da, axis=0)
        assert isinstance(result, DualArray)
        assert result.shape == (1, 2)


class TestRavel:
    def test_basic(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.ravel(da)
        assert isinstance(result, DualArray)
        assert result.shape == (4,)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )


class TestHstack:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.hstack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.3, 0.4]),
        )

    def test_mixed_dual_and_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.hstack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([0.1, 0.2, 0.0, 0.0]),
        )


class TestVstack:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.vstack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )

    def test_mixed_dual_and_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.vstack([a, b])
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.0, 0.0]]),
        )


class TestDstack:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.dstack([a, b])
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.dstack([[1.0, 2.0], [3.0, 4.0]]))

    def test_mixed_dual_and_ndarray(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = np.array([3.0, 4.0])
        result = np.dstack([a, b])
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.dstack([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(result.tangent[..., 1].ravel(), np.array([0.0, 0.0]))


class TestTile:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.tile(da, 2)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 2.0, 1.0, 2.0]),
            np.array([0.1, 0.2, 0.1, 0.2]),
        )


class TestRepeat:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.repeat(da, 2)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([1.0, 1.0, 2.0, 2.0]),
            np.array([0.1, 0.1, 0.2, 0.2]),
        )


class TestFlip:
    def test_1d(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.flip(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([3.0, 2.0, 1.0]),
            np.array([0.3, 0.2, 0.1]),
        )


class TestFliplr:
    def test_basic(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.fliplr(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[2.0, 1.0], [4.0, 3.0]]),
            np.array([[0.2, 0.1], [0.4, 0.3]]),
        )


class TestFlipud:
    def test_basic(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        result = np.flipud(da)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([[3.0, 4.0], [1.0, 2.0]]),
            np.array([[0.3, 0.4], [0.1, 0.2]]),
        )


class TestRoll:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.roll(da, 1)
        assert isinstance(result, DualArray)
        assert_dual_close(
            result,
            np.array([3.0, 1.0, 2.0]),
            np.array([0.3, 0.1, 0.2]),
        )


class TestBroadcastTo:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.broadcast_to(da, (3, 2))
        assert isinstance(result, DualArray)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result.primal, np.broadcast_to([1.0, 2.0], (3, 2)))


class TestSplit:
    def test_split_returns_dual_arrays(self):
        da = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        parts = np.split(da, 2)
        assert len(parts) == 2
        assert all(isinstance(p, DualArray) for p in parts)
        assert_dual_close(parts[0], np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert_dual_close(parts[1], np.array([3.0, 4.0]), np.array([0.3, 0.4]))

    def test_split_with_axis(self):
        da = DualArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        parts = np.split(da, 2, axis=1)
        assert len(parts) == 2
        assert all(isinstance(p, DualArray) for p in parts)
        assert parts[0].shape == (2, 1)
        assert parts[1].shape == (2, 1)

    def test_split_parts_usable_in_operations(self):
        da = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        parts = np.split(da, 2)
        result = parts[0] + parts[1]
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([4.0, 6.0]), np.array([0.4, 0.6]))


class TestArraySplit:
    def test_array_split_returns_dual_arrays(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        parts = np.array_split(da, 2)
        assert len(parts) == 2
        assert all(isinstance(p, DualArray) for p in parts)
        assert_dual_close(parts[0], np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        assert_dual_close(parts[1], np.array([3.0]), np.array([0.3]))

    def test_array_split_parts_usable_in_operations(self):
        da = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        parts = np.array_split(da, 2)
        result = parts[0] * parts[1]
        assert isinstance(result, DualArray)


class TestAtleast1d:
    def test_scalar(self):
        da = DualArray(np.array(5.0), np.array(0.1))
        result = np.atleast_1d(da)
        assert isinstance(result, DualArray)
        assert result.primal.ndim >= 1

    def test_vector(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.atleast_1d(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([1.0, 2.0]), np.array([0.1, 0.2]))

    def test_multi_arg(self):
        a = DualArray(np.array(1.0), np.array(0.1))
        b = DualArray(np.array([2.0, 3.0]), np.array([0.2, 0.3]))
        result = np.atleast_1d(a, b)
        assert isinstance(result, list)
        assert len(result) == 2


class TestAtleast2d:
    def test_1d(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.atleast_2d(da)
        assert isinstance(result, DualArray)
        assert result.primal.ndim >= 2

    def test_scalar(self):
        da = DualArray(np.array(5.0), np.array(0.1))
        result = np.atleast_2d(da)
        assert result.primal.ndim >= 2


class TestAtleast3d:
    def test_1d(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.atleast_3d(da)
        assert isinstance(result, DualArray)
        assert result.primal.ndim >= 3

    def test_2d(self):
        da = DualArray(np.array([[1.0, 2.0]]), np.array([[0.1, 0.2]]))
        result = np.atleast_3d(da)
        assert result.primal.ndim >= 3


class TestColumnStack:
    def test_basic(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.column_stack([a, b])
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.column_stack([a.primal, b.primal]))
        np.testing.assert_allclose(
            result.tangent, np.column_stack([a.tangent, b.tangent])
        )


class TestPad:
    def test_constant(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.pad(da, 2, mode="constant", constant_values=0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([0, 0, 1, 2, 3, 0, 0]))
        np.testing.assert_allclose(
            result.tangent, np.array([0, 0, 0.1, 0.2, 0.3, 0, 0])
        )

    def test_2d(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.pad(da, 1, mode="constant")
        assert isinstance(result, DualArray)
        assert result.primal.shape == (4, 4)
        assert result.tangent.shape == (4, 4)


class TestRot90:
    def test_basic(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.rot90(da)
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.rot90(p), np.rot90(t))

    def test_k2(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.rot90(da, k=2)
        assert_dual_close(result, np.rot90(p, k=2), np.rot90(t, k=2))


class TestTake:
    def test_basic(self):
        da = DualArray(
            np.array([10.0, 20.0, 30.0, 40.0]), np.array([0.1, 0.2, 0.3, 0.4])
        )
        result = np.take(da, [0, 2])
        assert isinstance(result, DualArray)
        assert_dual_close(result, np.array([10.0, 30.0]), np.array([0.1, 0.3]))

    def test_with_axis(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = np.array([[0.1, 0.2], [0.3, 0.4]])
        da = DualArray(p, t)
        result = np.take(da, [0], axis=0)
        assert isinstance(result, DualArray)


class TestTakeAlongAxis:
    def test_basic(self):
        p = np.array([[10.0, 30.0, 20.0], [60.0, 40.0, 50.0]])
        t = np.array([[0.1, 0.3, 0.2], [0.6, 0.4, 0.5]])
        da = DualArray(p, t)
        idx = np.argsort(p, axis=1)
        result = np.take_along_axis(da, idx, axis=1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.take_along_axis(p, idx, axis=1))
        np.testing.assert_allclose(result.tangent, np.take_along_axis(t, idx, axis=1))


class TestInsert:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        result = np.insert(da, 1, 10.0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 10.0, 2.0, 3.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.0, 0.2, 0.3]))

    def test_dual_value(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        val = DualArray(np.array(5.0), np.array(0.5))
        result = np.insert(da, 1, val)
        np.testing.assert_allclose(result.primal, np.array([1.0, 5.0, 2.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.5, 0.2]))


class TestDelete:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        result = np.delete(da, 1)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.3, 0.4]))

    def test_with_axis(self):
        p = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        t = np.ones_like(p) * 0.1
        da = DualArray(p, t)
        result = np.delete(da, 1, axis=0)
        assert result.primal.shape == (2, 2)


class TestAppend:
    def test_basic(self):
        da = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        result = np.append(da, [3.0, 4.0])
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.2, 0.0, 0.0]))

    def test_with_dual(self):
        a = DualArray(np.array([1.0, 2.0]), np.array([0.1, 0.2]))
        b = DualArray(np.array([3.0, 4.0]), np.array([0.3, 0.4]))
        result = np.append(a, b)
        np.testing.assert_allclose(result.primal, np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.tangent, np.array([0.1, 0.2, 0.3, 0.4]))


class TestSelect:
    def test_basic(self):
        x = DualArray(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.1, 0.2, 0.3, 0.4]))
        y = DualArray(
            np.array([10.0, 20.0, 30.0, 40.0]), np.array([1.0, 2.0, 3.0, 4.0])
        )
        condlist = [x.primal < 2.5, x.primal >= 2.5]
        result = np.select(condlist, [x, y])
        assert isinstance(result, DualArray)

    def test_with_default(self):
        x = DualArray(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        condlist = [np.array([True, False, False])]
        result = np.select(condlist, [x], default=0.0)
        assert isinstance(result, DualArray)
        np.testing.assert_allclose(result.primal, np.array([1.0, 0.0, 0.0]))
