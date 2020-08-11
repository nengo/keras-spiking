import numpy as np
import pytest

from keras_spiking import regularizers


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


@pytest.mark.parametrize(
    "reg",
    (
        regularizers.L1(l1=0.1, target=5),
        regularizers.L2(l2=0.1, target=5),
        regularizers.L1L2(l1=0.1, l2=0.2, target=5),
    ),
)
def test_regularizers_serialize(reg, allclose, rng):
    x = rng.uniform(-1, 10, size=(3, 4, 5))

    reg_reloaded = type(reg).from_config(reg.get_config())

    assert allclose(reg(x), reg_reloaded(x))
