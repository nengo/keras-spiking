import numpy as np
import pytest
import tensorflow as tf

from keras_spiking import regularizers


def range_error(x, t0, t1):
    return np.maximum(0, -(x - t0)) + np.maximum(0, x - t1)


def test_regularizers(allclose, rng):
    x = rng.uniform(-1, 10, size=(3, 4, 5))

    l1 = regularizers.L1(l1=0.1, target=5)
    assert allclose(l1(x), 0.1 * np.sum(np.abs(x - 5)))

    l2 = regularizers.L2(l2=0.1, target=5)
    assert allclose(l2(x), 0.1 * np.sum(np.square(x - 5)))

    l1l2 = regularizers.L1L2(l1=0.1, l2=0.2, target=5)
    assert allclose(
        l1l2(x), 0.1 * np.sum(np.abs(x - 5)) + 0.2 * np.sum(np.square(x - 5))
    )

    x_max = np.max(x, axis=0)

    max_l1 = regularizers.Percentile(target=4, l1=0.3)
    assert allclose(max_l1(x), 0.3 * np.sum(np.abs(x_max - 4)))

    max_l2 = regularizers.Percentile(target=4, l2=0.3)
    assert allclose(max_l2(x), 0.3 * np.sum(np.square(x_max - 4)))

    max_range_l1 = regularizers.Percentile(target=(3, 4), l1=0.3)
    assert allclose(max_range_l1(x), 0.3 * np.sum(range_error(x_max, 3, 4)))


def test_percentile_regularizer(allclose, rng):
    pytest.importorskip("tensorflow_probability")

    x = rng.uniform(-1, 10, size=(3, 2, 20))

    pct = 95
    axis = (0, 1)
    x_pct = np.percentile(x, pct, axis=axis, interpolation="linear")

    pct_l1 = regularizers.Percentile(target=4, percentile=pct, axis=axis, l1=0.3)
    assert allclose(pct_l1(x), 0.3 * np.sum(np.abs(x_pct - 4)))

    pct_range_l2 = regularizers.Percentile(
        target=(3, 4), percentile=pct, axis=axis, l2=0.01
    )
    assert allclose(pct_range_l2(x), 0.01 * np.sum(np.square(range_error(x_pct, 3, 4))))


def test_errors(monkeypatch):
    with pytest.raises(ValueError, match="`minimum` cannot exceed `maximum`"):
        regularizers.RangedRegularizer(target=(2, 1.99))

    with pytest.raises(ValueError, match="tuple with two elements"):
        regularizers.RangedRegularizer(target=(1, 2, 3))

    with pytest.raises(ValueError, match="`percentile` must be in the range"):
        regularizers.Percentile(percentile=-0.1)
    with pytest.raises(ValueError, match="`percentile` must be in the range"):
        regularizers.Percentile(percentile=101)

    with pytest.warns(UserWarning, match="weight is zero"):
        regularizers.Percentile()

    with pytest.raises(ValueError, match="< 100 requires tensorflow-probability"):
        monkeypatch.setattr(regularizers, "HAS_TFP", False)
        regularizers.Percentile(percentile=50)


@pytest.mark.parametrize(
    "Reg, args",
    (
        (
            regularizers.RangedRegularizer,
            dict(target=(3, 4), regularizer=tf.keras.regularizers.L1L2(l1=0.1)),
        ),
        (regularizers.L1, dict(l1=0.1, target=5)),
        (regularizers.L2, dict(l2=0.1, target=5)),
        (regularizers.L1L2, dict(l1=0.1, l2=0.2, target=5)),
        (regularizers.Percentile, dict(target=3, percentile=60, axis=(0, 2), l2=0.1)),
    ),
)
def test_regularizers_serialize(Reg, args, allclose, rng):
    if args.get("percentile", 100) != 100:
        pytest.importorskip("tensorflow_probability")

    reg = Reg(**args)

    x = rng.uniform(-1, 10, size=(3, 4, 5))

    reg_reloaded = type(reg).from_config(reg.get_config())

    assert allclose(reg(x), reg_reloaded(x))
