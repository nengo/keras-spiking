import numpy as np
import pytest
import tensorflow as tf


from keras_spiking import layers


@pytest.mark.parametrize("activation", (tf.nn.relu, tf.nn.tanh, "relu"))
def test_activations(activation, rng, allclose):
    x = rng.randn(32, 10, 2)

    ground = tf.keras.activations.get(activation)(x)

    # behaviour equivalent to base activation during training
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False
    )(x, training=True)
    assert allclose(y, ground)

    # not equivalent during inference
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False
    )(x, training=False)
    assert not allclose(y, ground)

    # equivalent during inference, with large enough dt
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False, dt=1e8
    )(x, training=False)
    assert allclose(y, ground)

    # not equivalent during training if using spiking_aware_training
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=True
    )(x, training=True)
    assert not allclose(y, ground)

    # equivalent with large enough dt
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=True, dt=1e8,
    )(x, training=True)
    assert allclose(y, ground)


def test_seed(seed, allclose):
    x = np.ones((2, 100, 10)) * 100

    # layers with the same seed produce the same output
    y0 = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)(x)

    y1 = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)(x)

    assert allclose(y0, y1)

    # layers with different seeds produce different output
    y2 = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed + 1)(x)

    assert not allclose(y0, y2)

    # the same layer called multiple times will produce the same output (if the seed
    # is set)
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)
    assert allclose(layer(x), layer(x))

    # layer will produce different output each time if seed not set
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True)
    assert not allclose(layer(x), layer(x))


def test_spiking_aware_training(rng, allclose):
    layer = layers.SpikingActivation(tf.nn.relu, spiking_aware_training=False)
    layer_sat = layers.SpikingActivation(tf.nn.relu, spiking_aware_training=True)
    with tf.GradientTape(persistent=True) as g:
        x = tf.constant(rng.uniform(-1, 1, size=(10, 20, 32)))
        g.watch(x)
        y = layer(x, training=True)
        y_sat = layer_sat(x, training=True)
        y_ground = tf.nn.relu(x)[:, -1]

    # forward pass is different
    assert allclose(y, y_ground)
    assert not allclose(y_sat, y_ground)

    # gradients are the same
    assert allclose(g.gradient(y, x), g.gradient(y_ground, x))
    assert allclose(g.gradient(y_sat, x), g.gradient(y_ground, x))


def test_spiking_swap_functional(allclose):
    inp = tf.keras.Input((None, 1))
    x0 = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=10, activation=tf.nn.leaky_relu)
    )(inp)
    x1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=10))(inp)
    x1 = layers.SpikingActivation(
        tf.nn.leaky_relu, return_sequences=True, spiking_aware_training=False
    )(x1)

    model = tf.keras.Model(inp, (x0, x1))
    model.compile(loss="mse", optimizer=tf.optimizers.SGD(0.1))

    model.fit(
        np.ones((32, 1, 1)),
        [np.ones((32, 1, 10)) * np.arange(1, 100, 10)] * 2,
        epochs=200,
        verbose=0,
    )

    y0, y1 = model.predict(np.ones((1, 1000, 1)))
    assert allclose(y0, np.arange(1, 100, 10), atol=1)
    assert allclose(
        np.sum(y1 * 0.001, axis=1, keepdims=True), np.arange(1, 100, 10), atol=1
    )


def test_stateful(allclose, rng, seed):
    layer = layers.SpikingActivation(
        tf.nn.relu, stateful=False, return_state=True, return_sequences=True, seed=seed
    )
    layer_stateful = layers.SpikingActivation(
        tf.nn.relu, stateful=True, return_state=True, return_sequences=True, seed=seed
    )

    x = rng.uniform(size=(32, 100, 32))
    # note: need to set initial state to zero due to bug in TF, see
    # https://github.com/tensorflow/tensorflow/issues/42193
    _, s = layer(x, initial_state=[tf.zeros((32, 32))])

    # non-stateful layers start from the same state each time
    assert allclose(s, layer(x, initial_state=[tf.zeros((32, 32))])[1])

    # stateful layers persist state between calls
    states = [layer_stateful(x[:, i * 10 : (i + 1) * 10])[1] for i in range(10)]
    assert allclose(s, states[-1])

    # reset_states resets to initial conditions
    layer_stateful.reset_states()
    assert allclose(layer_stateful(x[:, :10])[1], states[0])

    # can override initial state
    layer_stateful.reset_states(states=states[-2])
    assert allclose(layer_stateful(x[:, -10:])[1], states[-1])


def test_unroll(allclose, seed, rng):
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)
    layer_unroll = layers.SpikingActivation(
        tf.nn.relu, return_sequences=True, seed=seed, unroll=True
    )

    x = rng.uniform(size=(32, 100, 32))
    assert allclose(layer(x), layer_unroll(x))


def test_time_major(allclose, rng, seed):
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)
    layer_time_major = layers.SpikingActivation(
        tf.nn.relu, return_sequences=True, seed=seed, time_major=True
    )

    x = rng.uniform(size=(32, 100, 32))
    assert allclose(
        layer(x), np.transpose(layer_time_major(np.transpose(x, (1, 0, 2))), (1, 0, 2))
    )


@pytest.mark.parametrize("use_cell", (True, False))
def test_save_load(use_cell, allclose, tmpdir, seed):
    inp = tf.keras.Input((None, 32))
    if use_cell:
        out = tf.keras.layers.RNN(
            layers.SpikingActivationCell(units=32, activation=tf.nn.relu, seed=seed)
        )(inp)
    else:
        out = layers.SpikingActivation(tf.nn.relu, seed=seed)(inp)

    model = tf.keras.Model(inp, out)

    model.save(str(tmpdir))

    model_load = tf.keras.models.load_model(
        str(tmpdir),
        custom_objects={"SpikingActivationCell": layers.SpikingActivationCell}
        if use_cell
        else {"SpikingActivation": layers.SpikingActivation},
    )

    assert allclose(
        model.predict(np.ones((32, 10, 32))), model_load.predict(np.ones((32, 10, 32)))
    )
