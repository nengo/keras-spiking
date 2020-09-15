import pytest

from keras_spiking import config, layers


@pytest.fixture(scope="function")
def reset_defaults():
    dt = config.default.dt
    yield
    config.default.dt = dt


@pytest.mark.parametrize(
    "Layer",
    (
        (lambda dt=None: layers.Lowpass(tau=0.01, dt=dt)),
        (lambda dt=None: layers.SpikingActivation("relu", dt=dt)),
        (lambda dt=None: layers.Alpha(tau=0.01, dt=dt)),
    ),
)
def test_dt(Layer, reset_defaults):
    layer = Layer()
    assert layer.dt == 0.001

    config.default.dt = 1

    # changing the default after the fact does not affect already instantiated layers
    assert layer.dt == 0.001

    # newly created layers are affected
    layer1 = Layer()
    assert layer1.dt == 1

    # manually overriding the default works properly
    layer2 = Layer(dt=2)
    assert layer2.dt == 2
