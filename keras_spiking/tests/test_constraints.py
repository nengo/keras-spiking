import numpy as np
import pytest
import tensorflow as tf
from packaging import version

from keras_spiking.constraints import Mean


def simple_dense_model(n_features, n_units, axis=0):
    inp = tf.keras.Input((None, n_features))
    dense = tf.keras.layers.Dense(
        units=n_units,
        kernel_constraint=Mean(axis=axis),
        bias_constraint=Mean(),
    )(inp)
    out = tf.keras.layers.Dense(n_features)(dense)
    return tf.keras.Model(inp, out)


@pytest.mark.parametrize("axis", (0, 1))
def test_mean_constraint(axis, rng):
    n_features = 3
    n_units = 10
    x, y = rng.rand(2, 32, 100, n_features).astype(np.float32)

    model = simple_dense_model(n_features=n_features, n_units=n_units, axis=axis)
    model.compile(loss="mse", optimizer=tf.optimizers.SGD(0.1))
    model.fit(x, x - 1, epochs=1, verbose=0)

    kernel_weights = model.layers[1].weights[0].numpy()
    bias_weights = model.layers[1].weights[1].numpy()

    assert kernel_weights.shape == (n_features, n_units)
    assert bias_weights.shape == (n_units,)

    learned_kernel = np.asarray(np.unique(kernel_weights, axis=axis))
    if axis == 0:
        assert learned_kernel.shape == (1, n_units)
    elif axis == 1:
        assert learned_kernel.shape == (n_features, 1)
    else:
        assert False

    learned_bias = np.unique(bias_weights)
    assert len(learned_bias) == 1


def test_save_load_mean_constraint(tmp_path):
    model = simple_dense_model(n_features=10, n_units=20)

    model.save(str(tmp_path))

    # Note: If custom_objects does not include the custom constraint then it will be
    # missing in the loaded model. It will also be missing in earlier TF versions.
    model_load = tf.keras.models.load_model(
        str(tmp_path), custom_objects={"Mean": Mean}
    )
    if version.parse(tf.__version__) < version.parse("2.4.0"):
        pytest.xfail(f"TensorFlow {tf.__version__} does not load custom constraints")

    assert len(model_load.layers[1].weights) == 2
    assert isinstance(model_load.layers[1].weights[0].constraint, Mean)
    assert isinstance(model_load.layers[1].weights[1].constraint, Mean)
