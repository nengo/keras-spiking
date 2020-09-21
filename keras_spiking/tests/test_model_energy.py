import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    TimeDistributed,
)

from keras_spiking.layers import SpikingActivation
from keras_spiking.model_energy import ModelEnergy


@pytest.mark.parametrize("use_time", [False, True])
@pytest.mark.parametrize("use_skip", [False, True])
def test_summary(use_time, use_skip, rng):
    tf.random.set_seed(0)

    TimeWrapper = TimeDistributed if use_time else (lambda x: x)
    PoolLayer = GlobalAveragePooling3D if use_time else GlobalAveragePooling2D
    image_shape = [22, 22, 3]

    input_data = rng.uniform(-1, 1, size=[4] + ([10] if use_time else []) + image_shape)

    inp = tf.keras.Input(([None] if use_time else []) + image_shape)
    x1 = TimeWrapper(Conv2D(filters=4, kernel_size=(1, 1), activation="relu"))(inp)
    x = TimeWrapper(Conv2D(filters=32, kernel_size=(3, 3)))(x1)
    x = TimeWrapper(Activation("relu"))(x)
    x = TimeWrapper(Dropout(0.1))(x)
    x = PoolLayer()(x)
    y = Dense(units=10, activation=tf.nn.relu)(x)
    if use_skip:
        x = PoolLayer()(x1)
        x = Dense(units=10, activation=tf.nn.relu)(x)
        y = tf.keras.layers.Add()([y, x])
    model = tf.keras.Model(inp, [y])

    with tf.device("CPU"):
        energy = ModelEnergy(model, example_data=input_data)

    summary = energy.summary_string(
        timesteps_per_inference=10,
        line_length=200,  # long, so we don't have to worry about line wrapping
        columns=(
            "name",
            "output_shape",
            "params",
            "connections",
            "neurons",
            "rate",
            "energy cpu",
            "energy spinnaker2",
            "synop_energy loihi",
            "neuron_energy loihi",
            "energy loihi",
        ),
    )

    summary_lines = summary.split("\n")
    seps = [i for i, line in enumerate(summary_lines) if set(line) == set("=")]

    # check the totals lines at the end of the summary
    summary_lines = summary_lines[seps[-1] + 1 :]
    assert summary_lines[0].startswith("Total energy per inference [Joules/inf] (cpu)")
    assert summary_lines[1].startswith(
        "Total energy per inference [Joules/inf] (spinnaker2)"
    )
    assert summary_lines[2].startswith(
        "Total energy per inference [Joules/inf] (loihi)"
    )
    assert summary_lines[3].startswith("* These are estimates only")
    assert summary_lines[4].startswith("* This model contains non-spiking activations")

    if use_time:
        # make sure that spiking models don't produce the warning
        model = tf.keras.Model(inp, SpikingActivation("relu")(inp))
        energy = ModelEnergy(model)
        summary = energy.summary_string(columns=("name", "energy cpu"), line_length=200)
        summary_lines = summary.split("\n")
        assert summary_lines[-1].startswith("* These are estimates only")
        assert not any(
            line.startswith("* This model contains non-spiking activations")
            for line in summary_lines
        )


@pytest.mark.parametrize("timesteps", [1, 8, None])
def test_device_energy(timesteps, rng):
    TimeWrapper = TimeDistributed if timesteps != 1 else (lambda x: x)
    input_shape = ([timesteps] if timesteps != 1 else []) + [17, 17, 8]
    input_data = rng.uniform(
        -1, 1, size=[2] + ([5] + input_shape[1:] if timesteps is None else input_shape)
    )

    inp = tf.keras.Input(input_shape)
    x = TimeWrapper(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))(inp)
    y = TimeWrapper(Dense(units=10, activation="relu"))(x)
    model = tf.keras.Model(inp, [y])

    with tf.device("CPU"):
        energy = ModelEnergy(model, example_data=input_data)

    assert energy.total_energy(
        "loihi", timesteps_per_inference=46
    ) == 2 * energy.total_energy("loihi", timesteps_per_inference=23)


def test_layer_repeat(rng):
    input_shape = [10]

    layer = Dense(units=10, activation="relu")

    inp = tf.keras.Input(input_shape)
    model1 = tf.keras.Model(inp, [layer(inp)])
    model2 = tf.keras.Model(inp, [layer(layer(inp))])

    with tf.device("CPU"):
        energy1 = ModelEnergy(model1)
        energy2 = ModelEnergy(model2)
    for stat in ["connections", "neurons"]:
        assert energy2.layer_stats[layer][stat] == 2 * energy1.layer_stats[layer][stat]


@pytest.mark.parametrize(
    "spiking, timesteps",
    [(False, 1), (False, 8), (False, None), (True, 8), (True, None)],
)
@pytest.mark.parametrize("dims", [1, 3])
def test_activation_stats(dims, spiking, timesteps):
    input_shape = [5, 7, 6][-dims:]
    if timesteps != 1:
        input_shape = [timesteps] + input_shape
    TimeWrapper = tf.keras.layers.TimeDistributed if timesteps == 8 else lambda x: x

    inp = tf.keras.Input(input_shape)
    layer = SpikingActivation("relu") if spiking else TimeWrapper(Activation("relu"))
    model = tf.keras.Model(inp, [layer(inp)])

    energy = ModelEnergy(model)

    output_shape = layer.output_shape[2 if timesteps != 1 or spiking else 1 :]
    assert energy.layer_stats[layer]["connections"] == 0
    assert energy.layer_stats[layer]["neurons"] == np.prod(output_shape)


@pytest.mark.parametrize("timesteps", [1, 8, None])
@pytest.mark.parametrize("strides", [(1, 1, 1), (2, 1, 2)])
@pytest.mark.parametrize("act", [None, "relu"])
@pytest.mark.parametrize("dims", [1, 2, 3])
def test_conv_stats(dims, act, strides, timesteps):
    TimeWrapper = TimeDistributed if timesteps != 1 else (lambda x: x)
    Layer = getattr(tf.keras.layers, "Conv%dD" % dims)
    input_filters = 4
    input_shape = [5, 7, 6][-dims:] + [input_filters]
    kernel_size = [4, 5, 3][-dims:]
    strides = strides[-dims:]

    if timesteps != 1:
        input_shape = [timesteps] + input_shape

    inp = tf.keras.Input(input_shape)
    layer = TimeWrapper(
        Layer(
            filters=12,
            kernel_size=kernel_size,
            activation=act,
            strides=strides,
        )
    )
    model = tf.keras.Model(inp, [layer(inp)])

    energy = ModelEnergy(model)

    output_shape = layer.output_shape[2 if timesteps != 1 else 1 :]
    synops = np.prod(output_shape) * np.prod(kernel_size) * input_filters
    neurons = 0 if act is None else np.prod(output_shape)
    assert energy.layer_stats[layer]["connections"] == synops
    assert energy.layer_stats[layer]["neurons"] == neurons


@pytest.mark.parametrize("timesteps", [1, 8, None])
@pytest.mark.parametrize("act", [None, "relu"])
@pytest.mark.parametrize("input_shape", [[10], [5, 10]])
def test_dense_stats(input_shape, act, timesteps):
    TimeWrapper = TimeDistributed if timesteps != 1 else (lambda x: x)

    inp = tf.keras.Input(([timesteps] if timesteps != 1 else []) + input_shape)
    layer = TimeWrapper(Dense(20, activation=act))
    model = tf.keras.Model(inp, [layer(inp)])

    energy = ModelEnergy(model)

    input_filters = input_shape[-1]
    output_shape = layer.output_shape[2 if timesteps != 1 else 1 :]
    synops = np.prod(output_shape) * input_filters
    neurons = 0 if act is None else np.prod(output_shape)
    assert energy.layer_stats[layer]["connections"] == synops
    assert energy.layer_stats[layer]["neurons"] == neurons


def test_modelenergy_errors():
    class DummyLayer(tf.keras.layers.Layer):
        pass

    # stats for non-layer type
    with pytest.raises(ValueError, match="Cannot compute stats for layer of type"):
        ModelEnergy.compute_layer_stats(object())

    # stats for unknown layer type
    dummy_layer = DummyLayer()
    dummy_layer(tf.keras.Input(10))
    with pytest.warns(Warning, match="Cannot compute stats for layer of type"):
        ModelEnergy.compute_layer_stats(dummy_layer)

    # layer registered twice
    @ModelEnergy.register_layer(DummyLayer)
    def dummylayer_stats(layer, **_):
        return {}

    with pytest.warns(Warning, match="Layer 'DummyLayer' already registered"):

        @ModelEnergy.register_layer(DummyLayer)  # noqa: F811
        def dummylayer_stats(layer, **_):  # noqa: F811
            return {"notastat": 5}

    # device registered twice
    ModelEnergy.register_device(
        "dummydevice", energy_per_synop=1e-12, energy_per_neuron=1e-10, spiking=True
    )
    with pytest.warns(Warning, match="Device 'dummydevice' already registered"):
        ModelEnergy.register_device(
            "dummydevice", energy_per_synop=1e-12, energy_per_neuron=1e-10, spiking=True
        )

    with pytest.raises(ValueError, match="invalid stat 'notastat'"):
        ModelEnergy.compute_layer_stats(dummy_layer)

    # timestep mismatch
    inp = tf.keras.Input([5, 10])
    layer = Dense(20, activation="relu")
    model_t5 = tf.keras.Model(inp, [layer(inp)])

    # energy for unknown device
    energy = ModelEnergy(model_t5)
    with pytest.raises(ValueError, match="Energy specs unknown for device"):
        energy.total_energy("notadevice")

    # layer without defined input
    with pytest.raises(ValueError, match="never been applied to any inputs"):
        ModelEnergy.compute_layer_stats(Dense(10))

    # invalid column requested
    with pytest.raises(ValueError, match="Unknown column type 'notacolumn'"):
        energy.summary(columns=["name", "notacolumn"])

    # layer with undefined shape
    inp = tf.keras.Input([None, None, 5])
    layer = Dense(20)
    model = tf.keras.Model(inp, layer(inp))
    with pytest.raises(
        ValueError, match="output shape \\(None, 20\\) contains undefined elements"
    ):
        ModelEnergy(model)

    # computing energy on spiking device with no example data
    with pytest.raises(ValueError, match="example_data must be given"):
        energy.total_energy("loihi")


@pytest.mark.parametrize("time", (None, 5))
def test_activation_distribution(time):
    inp = tf.keras.Input((time, 10))
    rate = tf.keras.layers.ReLU()
    rate_distributed = tf.keras.layers.TimeDistributed(tf.keras.layers.ReLU())
    spiking = SpikingActivation("relu")
    rate(inp)
    rate_distributed(inp)
    spiking(inp)

    assert (
        ModelEnergy.compute_layer_stats(rate)["neurons"] == 10 if time is None else 50
    )
    assert ModelEnergy.compute_layer_stats(rate_distributed)["neurons"] == 10
    assert ModelEnergy.compute_layer_stats(spiking)["neurons"] == 10
