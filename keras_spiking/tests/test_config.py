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
        (lambda **kwargs: layers.Lowpass(tau_initializer=0.01, **kwargs)),
        (lambda **kwargs: layers.SpikingActivation("relu", **kwargs)),
        (lambda **kwargs: layers.Alpha(tau_initializer=0.01, **kwargs)),
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
