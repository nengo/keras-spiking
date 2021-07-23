import numpy as np
import pytest
import tensorflow as tf
from packaging import version

from keras_spiking import constraints, layers


@pytest.mark.parametrize("activation", (tf.nn.relu, tf.nn.tanh, "relu"))
def test_activations(activation, rng, allclose):
    x = rng.randn(32, 10, 2).astype(np.float32)

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
    x = np.ones((2, 100, 10), dtype=np.float32) * 100

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
        x = tf.constant(rng.uniform(-1, 1, size=(10, 20, 32)).astype(np.float32))
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

    x = rng.uniform(size=(32, 100, 32)).astype(np.float32)

    # note: need to set initial state to zero due to bug in TF<2.4, see
    # https://github.com/tensorflow/tensorflow/issues/42193
    initial_state = (
        [tf.zeros((32, 32))]
        if version.parse(tf.__version__) < version.parse("2.4.0")
        else None
    )

    _, s = layer(x, initial_state=initial_state)

    # non-stateful layers start from the same state each time
    assert allclose(s, layer(x, initial_state=initial_state)[1])

    # stateful layers persist state between calls
    states = [layer_stateful(x[:, i * 10 : (i + 1) * 10])[1] for i in range(10)]
    assert allclose(s, states[-1])

    # reset_states resets to initial conditions
    layer_stateful.reset_states(states=initial_state)
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

    x = rng.uniform(size=(32, 100, 32)).astype(np.float32)
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

    x = rng.uniform(size=(32, 100, 32)).astype(np.float32)
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
            layers.LowpassCell(tau_initializer=0.01, size=32), return_sequences=True
        )(out)
        out = tf.keras.layers.RNN(
            layers.AlphaCell(tau_initializer=0.01, size=32), return_sequences=True
        )(out)
    else:
        out = layers.SpikingActivation(tf.nn.relu, seed=seed)(inp)
        out = layers.Lowpass(tau_initializer=0.01)(out)
        out = layers.Alpha(tau_initializer=0.01)(out)

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
        layer = layers.Lowpass(tau_initializer=tau, dt=dt)
        synapse = nengo.Lowpass(tau)
    elif kind == "alpha":
        layer = layers.Alpha(tau_initializer=tau, dt=dt)
        synapse = nengo.Alpha(tau)

    x = rng.randn(10, steps, units).astype(np.float32)
    y = layer(x)

    x_nengo = np.moveaxis(x, 1, 0).reshape(steps, -1)
    y_nengo = synapse.filt(x_nengo, axis=0, dt=dt)
    y_nengo = np.moveaxis(y_nengo.reshape(steps, -1, units), 0, 1)

    assert allclose(y, y_nengo, atol=1e-6)


@pytest.mark.parametrize("Layer", [layers.Lowpass, layers.Alpha])
def test_filter_apply_during_training(Layer, allclose, rng):
    x = rng.randn(10, 100, 32).astype(np.float32)

    # apply_during_training=False:
    #   confirm `output == input` for training=True, but not training=False
    layer = Layer(
        tau_initializer=0.1, apply_during_training=False, return_sequences=True
    )
    assert allclose(layer(x, training=True), x)
    assert not allclose(layer(x, training=False), x, record_rmse=False, print_fail=0)

    # apply_during_training=True:
    #   confirm `output != input` for both values of `training`, and
    #   output is equal for both values of `training`
    layer = Layer(
        tau_initializer=0.1, apply_during_training=True, return_sequences=True
    )
    assert not allclose(layer(x, training=True), x, record_rmse=False, print_fail=0)
    assert not allclose(layer(x, training=False), x, record_rmse=False, print_fail=0)
    assert allclose(layer(x, training=True), layer(x, training=False))


@pytest.mark.parametrize("Layer", (layers.Lowpass, layers.Alpha))
def test_filter_trainable(Layer, allclose):
    tau = 0.1
    n_steps = 10
    tolerance = 1e-3 if Layer is layers.Lowpass else 3e-2

    inputs = np.ones((1, n_steps, 1), dtype=np.float32) * 0.5
    # we'll train the layers to match the output of this lowpass filter (with
    # a different tau/initial level)
    target_layer = Layer(
        tau_initializer=tau / 2,
        level_initializer=tf.initializers.constant(0.1),
    )
    targets = target_layer(inputs).numpy()

    inp = tf.keras.Input((None, 1))
    layer_trained = Layer(tau, apply_during_training=True)
    layer_skip = Layer(tau, apply_during_training=False)
    layer_untrained = Layer(tau, apply_during_training=True, trainable=False)

    model = tf.keras.Model(
        inp, [layer_trained(inp), layer_skip(inp), layer_untrained(inp)]
    )

    model.compile(loss="mse", optimizer=tf.optimizers.Adam(0.01))
    model.fit(inputs, [targets] * 3, epochs=150, verbose=0)

    # trainable layer should learn to output target
    ys = model.predict(inputs)
    assert allclose(ys[0], targets, atol=tolerance)
    assert not allclose(ys[1], targets, record_rmse=False, print_fail=0, atol=tolerance)
    assert not allclose(ys[2], targets, record_rmse=False, print_fail=0, atol=tolerance)

    # for trainable layer, parameters should match the target layer
    for w0, w1 in zip(layer_trained.weights, target_layer.weights):
        assert allclose(w0.numpy(), w1.numpy(), atol=tolerance)

    # other layers should stay at initial value
    assert allclose(layer_skip.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_untrained.layer.cell.initial_level.numpy(), 0)
    assert allclose(layer_skip.layer.cell.tau.numpy(), tau)
    assert allclose(layer_untrained.layer.cell.tau.numpy(), tau)


@pytest.mark.parametrize(
    "Layer",
    (
        lambda dt: layers.SpikingActivation("relu", dt=dt, seed=0),
        lambda dt: layers.Lowpass(0.1, dt=dt),
        lambda dt: layers.Alpha(0.1, dt=dt),
    ),
)
def test_dt_update(Layer, rng, allclose):
    x = rng.rand(32, 100, 64).astype(np.float32)

    dt = tf.Variable(0.01)

    layer = Layer(dt=dt)

    assert allclose(layer(x), Layer(dt=0.01)(x))

    dt.assign(0.05)

    assert allclose(layer(x), Layer(dt=0.05)(x))


@pytest.mark.parametrize("layer", ("spikingactivation", "lowpass", "alpha"))
def test_multid_input(layer, rng, seed, allclose):
    x = rng.rand(32, 10, 1, 2, 3).astype(np.float32)

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


@pytest.mark.parametrize(
    "layer_cls", [layers.LowpassCell, layers.Lowpass, layers.AlphaCell, layers.Alpha]
)
def test_filter_constraints(layer_cls, rng):
    n_features = 3
    initial_tau = 0.1
    x = rng.rand(32, 100, n_features).astype(np.float32)

    inp = tf.keras.Input((None, n_features))
    kwargs = dict(
        size=n_features,
        tau_initializer=initial_tau,
        tau_constraint=constraints.Mean(),
        initial_level_constraint=tf.keras.constraints.UnitNorm(axis=-1),
    )
    if issubclass(layer_cls, layers.KerasSpikingCell):
        layer = tf.keras.layers.RNN(layer_cls(**kwargs), return_sequences=True)
    elif issubclass(layer_cls, layers.KerasSpikingLayer):
        kwargs.pop("size")  # automatically determined in build method
        layer = layer_cls(**kwargs, return_sequences=True)
    else:
        assert False
    x0 = layer(inp)

    model = tf.keras.Model(inp, x0)
    model.compile(loss="mse", optimizer=tf.optimizers.SGD(0.1))

    model.fit(x, x, epochs=1, verbose=0)

    tau_weights = model.layers[1].weights[1]
    if layer_cls in (layers.Lowpass, layers.LowpassCell):
        assert tau_weights.shape == (1, n_features)
    elif layer_cls in (layers.Alpha, layers.AlphaCell):
        assert tau_weights.shape == (n_features,)
    else:
        assert False
    learned_tau = np.unique(tau_weights.numpy())
    assert len(learned_tau) == 1

    assert 0.1 * initial_tau < learned_tau.item() < 0.9 * initial_tau

    initial_level_weights = model.layers[1].weights[0]
    if layer_cls in (layers.Lowpass, layers.LowpassCell):
        assert initial_level_weights.shape == (1, n_features)
    elif layer_cls in (layers.Alpha, layers.AlphaCell):
        assert initial_level_weights.shape == (n_features,)
    assert np.allclose(np.linalg.norm(initial_level_weights.numpy()), 1)
