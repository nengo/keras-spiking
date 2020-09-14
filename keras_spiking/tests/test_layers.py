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
    assert not allclose(y, ground, record_rmse=False, print_fail=0)

    # equivalent during inference, with large enough dt
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=False, dt=1e8
    )(x, training=False)
    assert allclose(y, ground)

    # not equivalent during training if using spiking_aware_training
    y = layers.SpikingActivation(
        activation, return_sequences=True, spiking_aware_training=True
    )(x, training=True)
    assert not allclose(y, ground, record_rmse=False, print_fail=0)

    # equivalent with large enough dt
    y = layers.SpikingActivation(
        activation,
        return_sequences=True,
        spiking_aware_training=True,
        dt=1e8,
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

    assert not allclose(y0, y2, record_rmse=False, print_fail=0)

    # the same layer called multiple times will produce the same output (if the seed
    # is set)
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True, seed=seed)
    assert allclose(layer(x), layer(x))

    # layer will produce different output each time if seed not set
    layer = layers.SpikingActivation(tf.nn.relu, return_sequences=True)
    assert not allclose(layer(x), layer(x), record_rmse=False, print_fail=0)


def test_spiking_aware_training(rng, allclose):
    layer = layers.SpikingActivation(tf.nn.relu, spiking_aware_training=False)
    layer_sat = layers.SpikingActivation(tf.nn.relu, spiking_aware_training=True)
    with tf.GradientTape(persistent=True) as g:
        x = tf.constant(rng.uniform(-1, 1, size=(10, 20, 32)))
        g.watch(x)
        y = layer(x, training=True)[:, -1]
        y_sat = layer_sat(x, training=True)[:, -1]
        y_ground = tf.nn.relu(x)[:, -1]

    # forward pass is different
    assert allclose(y, y_ground)
    assert not allclose(y_sat, y_ground, record_rmse=False, print_fail=0)

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


@pytest.mark.parametrize(
    "Layer",
    (
        lambda **kwargs: layers.SpikingActivation(tf.nn.relu, seed=0, **kwargs),
        lambda **kwargs: layers.Lowpass(0.01, **kwargs),
    ),
)
def test_stateful(Layer, allclose, rng):
    layer = Layer(stateful=False, return_state=True, return_sequences=True)
    layer_stateful = Layer(stateful=True, return_state=True, return_sequences=True)

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


@pytest.mark.parametrize(
    "Layer",
    (
        lambda **kwargs: layers.SpikingActivation(tf.nn.relu, seed=0, **kwargs),
        lambda **kwargs: layers.Lowpass(0.01, **kwargs),
    ),
)
def test_unroll(Layer, allclose, rng):
    layer = Layer(return_sequences=True)
    layer_unroll = Layer(return_sequences=True, unroll=True)

    x = rng.uniform(size=(32, 100, 32))
    assert allclose(layer(x), layer_unroll(x))


@pytest.mark.parametrize(
    "Layer",
    (
        lambda **kwargs: layers.SpikingActivation(tf.nn.relu, seed=0, **kwargs),
        lambda **kwargs: layers.Lowpass(0.01, **kwargs),
    ),
)
def test_time_major(Layer, allclose, rng):
    layer = Layer(return_sequences=True)
    layer_time_major = Layer(return_sequences=True, time_major=True)

    x = rng.uniform(size=(32, 100, 32))
    assert allclose(
        layer(x), np.transpose(layer_time_major(np.transpose(x, (1, 0, 2))), (1, 0, 2))
    )


@pytest.mark.parametrize("use_cell", (True, False))
def test_save_load(use_cell, allclose, tmpdir, seed):
    inp = tf.keras.Input((None, 32))
    if use_cell:
        out = tf.keras.layers.RNN(
            layers.SpikingActivationCell(size=32, activation=tf.nn.relu, seed=seed),
            return_sequences=True,
        )(inp)
        out = tf.keras.layers.RNN(
            layers.LowpassCell(tau=0.01, size=32), return_sequences=True
        )(out)
        out = tf.keras.layers.RNN(
            layers.AlphaCell(tau=0.01, size=32), return_sequences=True
        )(out)
    else:
        out = layers.SpikingActivation(tf.nn.relu, seed=seed)(inp)
        out = layers.Lowpass(tau=0.01)(out)
        out = layers.Alpha(tau=0.01)(out)

    model = tf.keras.Model(inp, out)

    model.save(str(tmpdir))

    model_load = tf.keras.models.load_model(
        str(tmpdir),
        custom_objects={
            "SpikingActivationCell": layers.SpikingActivationCell,
            "LowpassCell": layers.LowpassCell,
            "AlphaCell": layers.AlphaCell,
        }
        if use_cell
        else {
            "SpikingActivation": layers.SpikingActivation,
            "Lowpass": layers.Lowpass,
            "Alpha": layers.Alpha,
        },
    )

    assert allclose(
        model.predict(np.ones((32, 10, 32))), model_load.predict(np.ones((32, 10, 32)))
    )


@pytest.mark.parametrize("kind", ("lowpass", "alpha"))
@pytest.mark.parametrize("dt", (0.001, 0.03))
def test_lowpass_alpha_tau(kind, dt, allclose, rng):
    """Verify that the keras-spiking filter matches the Nengo implementation"""
    nengo = pytest.importorskip("nengo")

    units = 32
    steps = 100
    tau = 0.1
    if kind == "lowpass":
        layer = layers.Lowpass(tau=tau, dt=dt)
        synapse = nengo.Lowpass(tau)
    elif kind == "alpha":
        layer = layers.Alpha(tau=tau, dt=dt)
        synapse = nengo.Alpha(tau)

    x = rng.randn(10, steps, units).astype(np.float32)
    y = layer(x)

    x_nengo = np.moveaxis(x, 1, 0).reshape(steps, -1)
    y_nengo = synapse.filt(x_nengo, axis=0, dt=dt)
    y_nengo = np.moveaxis(y_nengo.reshape(steps, -1, units), 0, 1)

    assert allclose(y, y_nengo, atol=1e-6)


@pytest.mark.parametrize("Layer", [layers.Lowpass, layers.Alpha])
def test_filter_apply_during_training(Layer, allclose, rng):
    x = rng.randn(10, 100, 32)

    # apply_during_training=False:
    #   confirm `output == input` for training=True, but not training=False
    layer = Layer(tau=0.1, apply_during_training=False, return_sequences=True)
    assert allclose(layer(x, training=True), x)
    assert not allclose(layer(x, training=False), x, record_rmse=False, print_fail=0)

    # apply_during_training=True:
    #   confirm `output != input` for both values of `training`, and
    #   output is equal for both values of `training`
    layer = Layer(tau=0.1, apply_during_training=True, return_sequences=True)
    assert not allclose(layer(x, training=True), x, record_rmse=False, print_fail=0)
    assert not allclose(layer(x, training=False), x, record_rmse=False, print_fail=0)
    assert allclose(layer(x, training=True), layer(x, training=False))


def test_lowpass_trainable(allclose):
    inp = tf.keras.Input((None, 1))
    layer_trained = layers.Lowpass(0.01, apply_during_training=True)
    layer_skip = layers.Lowpass(0.01, apply_during_training=False)
    layer_untrained = layers.Lowpass(0.01, apply_during_training=True, trainable=False)

    model = tf.keras.Model(
        inp, [layer_trained(inp), layer_skip(inp), layer_untrained(inp)]
    )

    model.compile(loss="mse", optimizer=tf.optimizers.SGD(0.5))
    model.fit(np.zeros((1, 1, 1)), [np.ones((1, 1, 1))] * 3, epochs=10, verbose=0)

    # trainable layer should learn to output 1
    ys = model.predict(np.zeros((1, 1, 1)))
    assert allclose(ys[0], 1)
    assert not allclose(ys[1], 1, record_rmse=False, print_fail=0)
    assert not allclose(ys[2], 1, record_rmse=False, print_fail=0)

    # for trainable layer, smoothing * initial_level should go to 1
    assert allclose(
        tf.nn.sigmoid(layer_trained.layer.cell.smoothing)
        * layer_trained.layer.cell.initial_level,
        1,
    )

    # other layers should stay at initial value
    assert allclose(layer_skip.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_untrained.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_skip.layer.cell.tau_var.numpy(), layer_skip.layer.cell.tau)
    assert allclose(
        layer_untrained.layer.cell.tau_var.numpy(),
        layer_untrained.layer.cell.tau,
    )


def test_alpha_trainable(allclose):
    n_train = 32 * 100
    steps = 5
    units = 2
    tau0 = 0.001

    inp = tf.keras.Input((None, units))
    layer_trained = layers.Alpha(tau0, apply_during_training=True)
    layer_skip = layers.Alpha(tau0, apply_during_training=False)
    layer_untrained = layers.Alpha(tau0, apply_during_training=True, trainable=False)

    model = tf.keras.Model(
        inp, [layer_trained(inp), layer_skip(inp), layer_untrained(inp)]
    )

    # initial `initial_level` should be near 0
    assert allclose(layer_trained.layer.cell.initial_level.numpy(), 0)

    # train model
    model.compile(
        loss="mse", optimizer=tf.optimizers.SGD(0.5, momentum=0.9, nesterov=True)
    )
    model.fit(
        np.zeros((n_train, steps, units)),
        [np.ones((n_train, steps, units))] * 3,
        epochs=10,
        verbose=0,
    )

    # trainable layer should learn to output 1 at all timesteps given 0 at all timesteps
    ys = model.predict(np.zeros((1, steps, units)))
    assert allclose(ys[0], 1, atol=3e-2)
    assert not allclose(ys[1], 1, atol=0.1, record_rmse=False, print_fail=0)
    assert not allclose(ys[2], 1, atol=0.1, record_rmse=False, print_fail=0)

    # learned `initial_level` should be near 1
    assert allclose(layer_trained.layer.cell.initial_level.numpy(), 1, atol=3e-2)

    # learned tau should be larger (tau trains slowly)
    smoothing = layer_trained.layer.cell.tau_var
    assert np.all(tf.nn.softplus(smoothing) > 3 * tau0)

    # other layers should stay at initial value
    assert allclose(layer_skip.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_untrained.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_skip.layer.cell.tau_var.numpy(), layer_skip.layer.cell.tau)
    assert allclose(
        layer_untrained.layer.cell.tau_var.numpy(),
        layer_untrained.layer.cell.tau,
    )


def test_lowpass_alpha_validation():
    with pytest.raises(ValueError, match="tau must be a positive number"):
        layers.LowpassCell(tau=0, size=1)

    with pytest.raises(ValueError, match="tau must be a positive number"):
        # note: error won't be raised until layer is applied (when LowpassCell is built)
        layers.Lowpass(tau=0)(np.zeros((1, 1, 1)))

    with pytest.raises(ValueError, match="tau must be a positive number"):
        layers.AlphaCell(tau=0, size=1)

    with pytest.raises(ValueError, match="tau must be a positive number"):
        # note: error won't be raised until layer is applied (when AlphaCell is built)
        layers.Alpha(tau=0)(np.zeros((1, 1, 1)))


@pytest.mark.parametrize(
    "Layer",
    (
        lambda dt: layers.SpikingActivation("relu", dt=dt, seed=0),
        lambda dt: layers.Lowpass(0.1, dt=dt),
        lambda dt: layers.Alpha(0.1, dt=dt),
    ),
)
def test_dt_update(Layer, rng, allclose):
    x = rng.rand(32, 100, 64)

    dt = tf.Variable(0.01)

    layer = Layer(dt=dt)

    assert allclose(layer(x), Layer(dt=0.01)(x))

    dt.assign(0.05)

    assert allclose(layer(x), Layer(dt=0.05)(x))


@pytest.mark.parametrize("layer", ("spikingactivation", "lowpass", "alpha"))
def test_multid_input(layer, rng, seed, allclose):
    x = rng.rand(32, 10, 1, 2, 3)

    if layer == "spikingactivation":
        y0 = layers.SpikingActivation("relu", seed=seed, dt=1)(
            np.reshape(x, (32, 10, -1))
        )
        y1 = layers.SpikingActivation("relu", seed=seed, dt=1)(x)
        y2 = tf.keras.layers.RNN(
            layers.SpikingActivationCell(y1.shape[2:], "relu", seed=seed, dt=1),
            return_sequences=True,
        )(x)
    elif layer == "lowpass":
        y0 = layers.Lowpass(0.1, dt=1)(np.reshape(x, (32, 10, -1)))
        y1 = layers.Lowpass(0.1, dt=1)(x)
        y2 = tf.keras.layers.RNN(
            layers.LowpassCell(y1.shape[2:], 0.1, dt=1), return_sequences=True
        )(x)
    elif layer == "alpha":
        y0 = layers.Alpha(0.1, dt=1)(np.reshape(x, (32, 10, -1)))
        y1 = layers.Alpha(0.1, dt=1)(x)
        y2 = tf.keras.layers.RNN(
            layers.AlphaCell(y1.shape[2:], 0.1, dt=1), return_sequences=True
        )(x)

    assert allclose(np.reshape(y0, x.shape), y1)
    assert allclose(y1, y2)
