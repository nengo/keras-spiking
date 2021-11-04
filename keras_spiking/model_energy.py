"""
Estimate energy usage on various devices for Keras models.
"""

import textwrap
import warnings
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    InputLayer,
    Layer,
    LeakyReLU,
    ReLU,
    Reshape,
    TimeDistributed,
)

from keras_spiking import compat, config
from keras_spiking.layers import SpikingActivation

# TODO: update docstring examples


class ModelEnergy:
    """Compute statistics and device energy estimates for a Keras model.

    Computes the following statistics on each layer:

    * "connections": The number of connections from all input elements to all
      activation units. The number of synaptic operations per second ("synops") will be
      computed by multiplying this number by the average firing rate of the input to
      the layer.
    * "neurons": The number of neuron updates per timestep performed by the
      layer. The number of neuron updates per inference will be computed by
      multiplying this number by the number of timesteps per inference.

    Using expected average firing rates for each layer in the network, along with the
    above statistics, this class can estimate the energy usage on one of the following
    types of devices (see `.total_energy` and `.summary`):

    * "cpu": Estimate energy usage on a CPU (Intel i7-4960X), assuming each synop/neuron
      update is one MAC [1]_.
    * "gpu": Estimate energy usage on a GPU (Nvidia GTX Titan Black), assuming each
      synop/neuron update is one MAC [1]_. Note that this assumes significant
      parallelism (e.g., inputs being processed in large batches).
    * "arm": Estimate energy usage on an ARM Cortex-A, assuming each synop/neuron update
      is one MAC [1]_.
    * "loihi": Estimate energy usage on the Intel Loihi chip [2]_.
    * "spinnaker" and "spinnaker2": Estimate energy usage on SpiNNaker or
      SpiNNaker 2 [3]_.

    Note: on non-spiking devices ("cpu"/"gpu"/"arm") this assumes
    the model is being run as a traditional non-spiking ANN (computing every synapse
    each timestep), not taking advantage of spike-based computation.
    This estimate is therefore independent of ``example_data``.
    On spiking devices ("loihi"/"spinnaker1"/"spinnaker2"), we assume
    that the model has been fully converted to a spiking implementation in some way,
    even if ``model`` contains non-spiking elements.

    For example:

    .. testcode::

        inp = tf.keras.Input(784)
        dense = tf.keras.layers.Dense(units=128, activation="relu")(inp)
        model = tf.keras.Model(inp, dense)

        energy = keras_spiking.ModelEnergy(model)
        energy.summary(line_length=80)

    .. testoutput::

        Layer (type)        |Output shape |Param #|Conn #|Neuron #|J/inf (cpu)
        --------------------|-------------|-------|------|--------|-----------
        input_3 (InputLayer)|[(None, 784)]|      0|     0|       0|          0
        dense_2 (Dense)     |  (None, 128)| 100480|100352|     128|    0.00086
        ======================================================================
        Total energy per inference [Joules/inf] (cpu): 8.64e-04
        ...

    Additional devices or different energy assumptions for a given device can be added
    with `.register_device`, e.g.

    .. testcode::

        keras_spiking.ModelEnergy.register_device(
            "my-cpu", energy_per_synop=1e-10, energy_per_neuron=2e-9, spiking=False
        )
        energy.summary(
            columns=("name", "energy cpu", "energy my-cpu"), line_length=80
        )

    .. testoutput::

        Layer (type)        |J/inf (cpu)|J/inf (my-cpu)
        --------------------|-----------|--------------
        input_3 (InputLayer)|          0|             0
        dense_2 (Dense)     |    0.00086|         1e-05
        ===============================================
        Total energy per inference [Joules/inf] (cpu): 8.64e-04
        Total energy per inference [Joules/inf] (my-cpu): 1.03e-05
        ...

    Notes
    -----

    It is important to keep in mind that actual power usage will be heavily dependent
    on the specific details of the underlying software and hardware implementation.
    The numbers provided by `.ModelEnergy` should be taken as very rough estimates only,
    and they rely on a number of assumptions:

    - **Device specifications**: In order to estimate the energy used by a model on a
      particular device, we need to know how much energy is used per synaptic
      operation/neuron update. We rely on published data for these numbers (see our
      sources below).
      Energy numbers in practice can differ significantly from published results.
    - **Overhead**: We do not account for any overhead in the energy estimates
      (e.g., the cost of transferring data on and off a device). We only estimate the
      energy usage of internal model computations (synaptic operations and neuron
      updates).
      In practice, overhead can be a significant contributor to the energy usage of
      a model.
    - **Spiking implementation**: We assume that the model being estimated has been
      fully converted to a spiking implementation when estimating the energy usage on
      a spiking device (even if the input model has non-spiking elements). For example,
      if the model contains ``tf.keras.layers.Activation("relu")`` layers (non-spiking),
      we assume that on a spiking device those layers will be converted to something
      equivalent to ``keras_spiking.SpikingActivation("relu")``, and that any connecting
      layers (e.g. ``tf.keras.layers.Dense``) are applied in an event-based fashion
      (i.e., processing only occurs when the input layer emits a spike). In practice,
      it is not trivial to map a neural network to a spiking device in this way, and
      implementation details can significantly affect energy usage.
      [Nengo](https://www.nengo.ai/nengo/) and [NengoDL](https://www.nengo.ai/nengo-dl/)
      are designed to make this easier.

    Parameters
    ----------
    model : ``tf.keras.Model``
        The model to compute statistics and energy estimates for.
    example_data : array_like
        Input used to estimate average firing rates of each layer (used to estimate
        the number of synaptic events). It is passed directly to ``model.predict`` (see
        ``tf.keras.Model.predict`` for all acceptable types of input data). This is
        required to estimate energy on spiking devices, but does not affect non-spiking
        devices.

    References
    ----------
    .. [1] Degnan, Brian, Bo Marr, and Jennifer Hasler. "Assessing trends in performance
       per watt for signal processing applications." IEEE Transactions on Very Large
       Scale Integration (VLSI) Systems 24.1 (2015): 58-66.
       https://ieeexplore.ieee.org/abstract/document/7054508
    .. [2] Davies, Mike, et al. "Loihi: A neuromorphic manycore processor with on-chip
       learning." IEEE Micro 38.1 (2018): 82-99.
       https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
    .. [3] Höppner, Sebastian, et al. "Dynamic power management for neuromorphic
       many-core systems." IEEE Transactions on Circuits and Systems I: Regular Papers
       66.8 (2019): 2973-2986. https://arxiv.org/abs/1903.08941
    """

    devices = {
        # https://ieeexplore.ieee.org/abstract/document/7054508
        # TODO: CPU neuron energy depends on neuron type
        "cpu": dict(spiking=False, energy_per_synop=8.6e-9, energy_per_neuron=8.6e-9),
        "gpu": dict(spiking=False, energy_per_synop=0.3e-9, energy_per_neuron=0.3e-9),
        "arm": dict(spiking=False, energy_per_synop=0.9e-9, energy_per_neuron=0.9e-9),
        # https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
        "loihi": dict(
            spiking=True,
            energy_per_synop=(23.6 + 3.5) * 1e-12,
            energy_per_neuron=81e-12,
        ),
        # https://arxiv.org/abs/1903.08941
        "spinnaker": dict(
            spiking=True, energy_per_synop=13.3e-9, energy_per_neuron=26e-9
        ),
        "spinnaker2": dict(
            spiking=True, energy_per_synop=450e-12, energy_per_neuron=2.19e-9
        ),
    }

    layer_stats_computers = {}

    def __init__(self, model, example_data=None):
        self.model = model
        self.example_data = example_data

        self._compute_model_stats()
        if example_data is None:
            self.layer_rates = {}
        else:
            self._compute_model_rates()

    @classmethod
    def compute_layer_stats(cls, layer, node=None):
        """Compute statistics for a given layer.

        Examples
        --------

        .. testcode::

            inp = tf.keras.Input([None, 10])
            layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5, "relu"))
            model = tf.keras.Model(inp, [layer(inp)])

            print(keras_spiking.ModelEnergy.compute_layer_stats(layer))

        .. testoutput::

            {'connections': 50, 'neurons': 5, 'spiking': False}
        """

        valid_stats = ("neurons", "connections", "spiking")

        for layer_cls in type(layer).__mro__:
            if layer_cls in cls.layer_stats_computers:
                if node is None:
                    if len(layer.inbound_nodes) == 0:
                        raise ValueError(
                            f"Layer {layer.name} has never been applied to any inputs, "
                            "so its stats are undefined"
                        )
                    node = layer.inbound_nodes[0]

                stats = cls.layer_stats_computers[layer_cls](node)
                for key in stats:
                    if key not in valid_stats:
                        raise ValueError(
                            f"{layer_cls.__name__} stat calculator returned an invalid "
                            f"stat '{key}'; must be one of {valid_stats}"
                        )
                return stats

        raise ValueError(
            f"Cannot compute stats for layer of type '{type(layer).__name__}'"
        )

    @classmethod
    def register_layer(cls, layer_class):
        """Decorator to register a statistic calculator for a layer.

        The input to the decorated function is a node from a layer for which we want
        to compute statistics.

        The decorated function should return a dictionary with the following entries:

          * "connections": The number of connections from all input elements to all
            activation units. Defaults to 0.
          * "neurons": The number of neuron updates per timestep performed by this
            layer. Defaults to 0.
          * "spiking": Whether or not this layer could be implemented on a spiking
            device as-is (e.g. ``tf.keras.layers.ReLU`` returns ``spiking=False``
            because non-spiking nonlinearities like ReLU can't be directly implemented
            on a spiking device). Defaults to True.

        Examples
        --------
        If we know that the ``tf.keras.layers.UpSampling2D`` layer uses one synaptic
        operation per output element, we can register the following function:

        .. testcode::

           @keras_spiking.ModelEnergy.register_layer(tf.keras.layers.UpSampling2D)
           def upsampling2d_stats(node):
               # note: ignore the batch dimension when computing output size
               output_size = np.prod(node.output_shapes[1:])

               return dict(connections=output_size, neurons=0)

           # use our registered stat calculator
           inp = tf.keras.Input([4, 4, 3])
           layer = tf.keras.layers.UpSampling2D(size=(2, 2))
           model = tf.keras.Model(inp, [layer(inp)])

           print(keras_spiking.ModelEnergy.compute_layer_stats(layer))

        .. testoutput::

           {'connections': 192, 'neurons': 0}

        We see that the synaptic operations is 192, which equals the number of pixels
        in the upsampled image size of ``(8, 8)``, times the number of channels (3).
        """

        def register_class(stats_fn):
            if layer_class in cls.layer_stats_computers:
                warnings.warn(
                    f"Layer '{layer_class.__name__}' already registered. Overwriting."
                )
            cls.layer_stats_computers[layer_class] = stats_fn
            return stats_fn

        return register_class

    @classmethod
    def register_device(cls, device_name, energy_per_synop, energy_per_neuron, spiking):
        """Register a new device type for estimating energy consumption.

        Parameters
        ----------
        device_name : str
            The string to use to refer to the device.
        energy_per_synop : float
            The energy (in Joules) used by a single synaptic operation on the device.
        energy_per_neuron : float
            The energy (in Joules) used by a single neuron update on the device.
        spiking : bool
            Whether the device is spiking (event-based), and thus only computes
            synaptic updates for incoming spikes, rather than on every timestep.
        """
        if device_name in cls.devices:
            warnings.warn(f"Device '{device_name}' already registered. Overwriting.")
        cls.devices[device_name] = dict(
            energy_per_synop=energy_per_synop,
            energy_per_neuron=energy_per_neuron,
            spiking=spiking,
        )

    def _compute_model_stats(self):
        """Compute statistics for ``self.model``."""

        self.node_stats = {}
        self.layer_stats = {}

        for nodes in self.model._nodes_by_depth.values():
            for node in nodes:
                assert node not in self.node_stats

                self.node_stats[node] = self.compute_layer_stats(
                    compat.node_layer(node), node=node
                )

        # add up stats for all nodes to get total stats for layer
        for layer in self.model.layers:
            self.layer_stats[layer] = {"neurons": 0, "connections": 0, "spiking": True}
        for node, stats in self.node_stats.items():
            for key, val in stats.items():
                if key in ("neurons", "connections"):
                    self.layer_stats[compat.node_layer(node)][key] += val
                elif key == "spiking":
                    self.layer_stats[compat.node_layer(node)][key] &= val
                else:
                    raise NotImplementedError

    def _compute_model_rates(self):
        """Compute layer input firing rates for the given data."""

        self.layer_rates = {}

        layer_inputs = [
            [
                node.input_tensors
                for node in layer.inbound_nodes
                if node in self.node_stats
            ]
            for layer in self.model.layers
        ]

        layer_model = tf.keras.Model(self.model.inputs, layer_inputs)
        rates = layer_model.predict(self.example_data)

        for layer, rates_in in zip(self.model.layers, rates):
            # sum over nodes of each layer
            self.layer_rates[layer] = sum(
                # compute mean input value (over batch/time/units)
                # using abs because we only care about how many events there are
                # per second (not whether those events are "positive" or "negative")
                np.mean(np.abs(x))
                for x in rates_in
            )

    def layer_energy(
        self,
        layer,
        device,
        timesteps_per_inference=1,
        dt=None,
    ):
        """Estimate the energy used by one layer.

        Parameters
        ----------
        layer : ``tf.keras.layers.Layer``
            Layer to estimate energy for. Note that if the same layer is being reused
            multiple times in the model, this will return the total energy
            summed over all the applications.
        device : str
            Device to estimate energy for. Can be a supported device
            (see `.ModelEnergy` for a list), or another device added with
            `.ModelEnergy.register_device`.
        timesteps_per_inference : int
            Timesteps used per inference (for example, if the model is classifying
            images and we want to present each image for 10 timesteps).
        dt : float
            The length of one timestep, in seconds, used by the device. Used to compute
            the number of synaptic events based on the firing rates (in Hz).
            Can differ from the ``dt`` used on any ``keras_spiking`` layers in the
            model. If None, uses
            `keras_spiking.default.dt <keras_spiking.config.DefaultManager>`
            (which is 0.001 seconds by default).

        Returns
        -------
        synop_energy : float
            Estimated energy used (in Joules) for synaptic computations per inference.
        neuron_energy : float
            Estimated energy used (in Joules) for neuron updates per inference.
        """

        dt = config.default.dt if dt is None else dt

        # get device stats
        device = str(device).lower()
        if device not in self.devices:
            raise ValueError(f"Energy specs unknown for device '{device}'")
        device_stats = self.devices[device]

        if device_stats["spiking"] and self.example_data is None:
            raise ValueError(
                "ModelEnergy.example_data must be given in order to calculate energy "
                "on a spiking device"
            )

        # assume input rate is an event every timestep if not on a spiking device
        input_rate = (
            (1 / dt) if not device_stats["spiking"] else self.layer_rates[layer]
        )
        # assume we're only simulating for a single timestep if not on a spiking device
        timesteps_per_inference = (
            1 if not device_stats["spiking"] else timesteps_per_inference
        )

        # energy/op * ops/event * events/s * s/timestep * timesteps/inference
        # = energy/inference
        synop_energy = (
            device_stats["energy_per_synop"]
            * self.layer_stats[layer]["connections"]
            * input_rate
            * dt
            * timesteps_per_inference
        )
        # energy/op * ops/timestep * timesteps/inference = energy/inference
        neuron_energy = (
            device_stats["energy_per_neuron"]
            * self.layer_stats[layer]["neurons"]
            * timesteps_per_inference
        )

        return synop_energy, neuron_energy

    def total_energy(self, device, timesteps_per_inference=1, dt=None):
        """Estimate the energy usage for a whole model.

        Parameters
        ----------
        device : str
            Device to estimate energy for. Can be a supported device
            (see `.ModelEnergy` for a list), or another device added with
            `.ModelEnergy.register_device`.
        timesteps_per_inference : int
            Timesteps used per inference (for example, if the model is classifying
            images and we want to present each image for 10 timesteps).
        dt : float
            The length of one timestep, in seconds, used by the device. Used to compute
            the number of synaptic events based on the firing rates (in Hz).
            Can differ from the ``dt`` used on any ``keras_spiking`` layers in the
            model. If None, uses
            `keras_spiking.default.dt <keras_spiking.config.DefaultManager>`
            (which is 0.001 seconds by default).

        Returns
        -------
        energy : float
            Total estimated energy used by the model (in Joules) per inference.
        """

        return sum(
            part
            # sum over layers
            for layer in self.model.layers
            # sum neuron+synaptic energy
            for part in self.layer_energy(
                layer, device, timesteps_per_inference=timesteps_per_inference, dt=dt
            )
        )

    def summary(  # noqa: C901
        self,
        columns=(
            "name",
            "output_shape",
            "params",
            "connections",
            "neurons",
            "energy cpu",
        ),
        timesteps_per_inference=1,
        dt=None,
        line_length=98,
        print_warnings=True,
    ):
        """Print a per-layer summary of computation statistics and energy estimates.

        Parameters
        ----------
        columns : list or tuple of string
            Columns to display. Can be any combination of the following:

            * "name": The layer name.
            * "output_shape": The output shape of the layer.
            * "params": The number of parameters in the layer.
            * "connections": The number of synaptic connections from inputs to
              the neurons of this layer (see `.ModelEnergy` for the definition of
              "connections").
            * "neurons": The number of neuron updates performed by the layer each
              timestep (see `.ModelEnergy` for the definition of "neuron update").
            * "rate": The average input firing rate to the layer, in spikes per
              second. Note that this is only relevant for layers that perform synaptic
              operations; for other layers (e.g. an activation layer that gets input
              from a convolutional layer), this number has no effect.
            * "synop_energy <device>": The estimated energy in Joules per inference used
              by the layer on <device> for synaptic operations.
            * "neuron_energy <device>": The estimated energy in Joules per inference
              used by the layer on <device> for neuron updates.
            * "energy <device>": The total estimated energy in Joules per inference used
              by the layer on <device>.

            Here, <device> can be any of the supported devices (see `.ModelEnergy`).
            Additional devices can be added with `.ModelEnergy.register_device`.
        timesteps_per_inference : int
            Timesteps used per inference (for example, if the model is classifying
            images and we want to present each image for 10 timesteps).
        dt : float
            The length of one timestep, in seconds, used by the device. Used to compute
            the number of synaptic events based on the firing rates (in Hz).
            Can differ from the ``dt`` used on any ``keras_spiking`` layers in the
            model. If None, uses
            `keras_spiking.default.dt <keras_spiking.config.DefaultManager>`
            (which is 0.001 seconds by default).
        line_length : int
            The length of each printed line.
        print_warnings : bool
            Set to False to disable the warnings regarding assumptions made in the
            energy calculations.
        """
        print(
            self.summary_string(
                columns=columns,
                timesteps_per_inference=timesteps_per_inference,
                dt=dt,
                line_length=line_length,
                print_warnings=print_warnings,
            )
        )

    def summary_string(  # noqa: C901
        self,
        columns=(
            "name",
            "output_shape",
            "params",
            "connections",
            "neurons",
            "energy cpu",
        ),
        timesteps_per_inference=1,
        dt=None,
        line_length=98,
        print_warnings=True,
    ):
        """Returns a per-layer summary of computation statistics and energy estimates.

        The same as `.summary`, except returns the summary as a string, rather than
        printing it.

        For documentation on parameters and other features, see `.summary`.
        """

        dt = config.default.dt if dt is None else dt

        def layer_output_shape(layer, _):
            try:
                return str(layer.output_shape)
            # exceptions are copied from `tf.keras.Model.summary`
            except AttributeError:  # pragma: no cover
                return "multiple"
            except RuntimeError:  # pragma: no cover
                # output_shape unknown in Eager mode.
                return "?"

        def layer_energy(layer, device, kind="all"):
            synop_energy, neuron_energy = self.layer_energy(
                layer, device, timesteps_per_inference=timesteps_per_inference, dt=dt
            )
            if kind == "all":
                energy = synop_energy + neuron_energy
            elif kind == "synop_energy":
                energy = synop_energy
            elif kind == "neuron_energy":
                energy = neuron_energy
            else:
                raise NotImplementedError

            return f"{energy:.2g}"

        col_value = dict(
            name=lambda layer, _: f"{layer.name} ({layer.__class__.__name__})",
            output_shape=layer_output_shape,
            params=lambda layer, _: str(layer.count_params()),
            rate=lambda layer, _: f"{self.layer_rates.get(layer, np.nan):.2g}",
            connections=lambda layer, _: str(self.layer_stats[layer]["connections"]),
            neurons=lambda layer, _: str(self.layer_stats[layer]["neurons"]),
            energy=layer_energy,
            synop_energy=partial(layer_energy, kind="synop_energy"),
            neuron_energy=partial(layer_energy, kind="neuron_energy"),
        )
        col_names = dict(
            name=lambda _: "Layer (type)",
            output_shape=lambda _: "Output shape",
            params=lambda _: "Param #",
            rate=lambda _: "Rate [Hz]",
            connections=lambda _: "Conn #",
            neurons=lambda _: "Neuron #",
            energy=lambda device: f"J/inf ({device})",
            synop_energy=lambda device: f"Synop J/inf ({device})",
            neuron_energy=lambda device: f"Neuron J/inf ({device})",
        )

        for c in columns:
            if c.split()[0] not in col_names:
                raise ValueError(
                    f"Unknown column type '{c}'; must be one of {list(col_names)}"
                )

        columns = [(col.split()[0], " ".join(col.split()[1:])) for col in columns]

        lines = []
        lines.append([col_names[col](argstr) for col, argstr in columns])
        for layer in self.model.layers:
            lines.append([col_value[col](layer, argstr) for col, argstr in columns])

        # --- compute column widths
        max_widths = np.array([[len(s) for s in line] for line in lines]).max(axis=0)
        usable_width = line_length - len(columns) + 1
        base_width = usable_width // len(columns)
        col_widths = np.minimum(base_width, max_widths)

        usable_width = min(usable_width, max_widths.sum())
        while col_widths.sum() < usable_width:
            diff_ratio = (max_widths - col_widths) / col_widths
            i = np.argmax(diff_ratio)
            col_widths[i] += 1

        col_aligns = ["<" if col in ("name",) else ">" for col, _ in columns]

        # --- print lines
        line_strs = []
        for i, line in enumerate(lines):
            parts = []
            for part, align, width in zip(line, col_aligns, col_widths):
                fstr = "{:" + ("<" if i == 0 else align) + str(width) + "}"
                part = fstr.format(part[:width])
                parts.append(part)

            line_strs.append("|".join(parts))

            if i == 0:
                line_strs.append("|".join("-" * width for width in col_widths))

        # --- print totals
        has_energy = False
        has_spiking = False
        for col, device in columns:
            if col == "energy":
                if not has_energy:
                    line_strs.append("=" * len(line_strs[0]))
                    has_energy = True

                if self.devices[device]["spiking"]:
                    has_spiking = True

                energy = self.total_energy(
                    device, dt=dt, timesteps_per_inference=timesteps_per_inference
                )
                line_strs.append(
                    f"Total energy per inference [Joules/inf] ({device}): {energy:0.2e}"
                )

        if print_warnings and has_energy:
            line_strs.append(
                textwrap.fill(
                    "* These are estimates only; see the documentation for a list of "
                    "the assumptions being made. https://bit.ly/3c3aKKH",
                    width=line_length,
                    subsequent_indent="  ",
                )
            )
        if (
            print_warnings
            and has_spiking
            and not all(
                self.layer_stats[layer]["spiking"] for layer in self.model.layers
            )
        ):
            line_strs.append(
                textwrap.fill(
                    "* This model contains non-spiking activations that would not "
                    "actually behave in the manner we assume in these calculations; "
                    "we assume these layers will be converted to spiking equivalents. "
                    "Consider using `keras_spiking.SpikingActivation` to make this "
                    "conversion explicit.",
                    width=line_length,
                    subsequent_indent="  ",
                )
            )

        return "\n".join(line_strs)


def _output_size(node):
    output_shape = node.output_shapes[1:]
    if output_shape[0] is None:
        # we'll assume this represents time
        output_shape = output_shape[1:]
    if any(x is None for x in output_shape):
        raise ValueError(
            f"Cannot compute stats for '{compat.node_layer(node).name}' layer because "
            f"the output shape {output_shape} contains undefined elements"
        )
    return np.prod(output_shape)


def _act_stats(activation, n_neurons):
    return (
        {"neurons": 0, "spiking": True}
        if activation in ("linear", tf.keras.activations.linear, None)
        else {"neurons": n_neurons, "spiking": False}
    )


@ModelEnergy.register_layer(Layer)
def layer_stats(node, **_):
    """Fallback for computing stats on an unknown layer (assumes none)."""

    # layer types that we know don't affect neurons/connections, so we don't register
    # a stat computer for them
    whitelist = (
        Add,
        Dropout,
        Flatten,
        GlobalAveragePooling1D,
        GlobalAveragePooling2D,
        GlobalAveragePooling3D,
        InputLayer,
        Reshape,
    )

    if not isinstance(compat.node_layer(node), whitelist):
        warnings.warn(
            "Cannot compute stats for layer of type "
            f"'{type(compat.node_layer(node)).__name__}'."
            "Use `ModelEnergy.register_layer` to register this layer."
        )
    return {}


@ModelEnergy.register_layer(Activation)
@ModelEnergy.register_layer(ReLU)
@ModelEnergy.register_layer(LeakyReLU)
def act_stats(node):
    """Compute activation layer stats."""
    return _act_stats(
        getattr(compat.node_layer(node), "activation", "relu"), _output_size(node)
    )


@ModelEnergy.register_layer(Conv1D)
@ModelEnergy.register_layer(Conv2D)
@ModelEnergy.register_layer(Conv3D)
def conv_stats(node):
    """Compute ``Conv1D``/``Conv2D``/``Conv3D`` layer stats."""

    kernel_size = np.prod(compat.node_layer(node).kernel.shape[:-1])
    output_size = _output_size(node)

    result = {"connections": int(output_size * kernel_size)}
    result.update(_act_stats(compat.node_layer(node).activation, output_size))
    return result


@ModelEnergy.register_layer(Dense)
def dense_stats(node):
    """Compute ``Dense`` layer stats."""

    output_size = _output_size(node)
    spatial_size = output_size // node.output_shapes[-1]
    kernel_size = np.prod(compat.node_layer(node).kernel.shape)

    result = {"connections": spatial_size * kernel_size}
    result.update(_act_stats(compat.node_layer(node).activation, output_size))
    return result


@ModelEnergy.register_layer(SpikingActivation)
def spikingactivation_stats(node):
    """Compute `.SpikingActivation` layer stats."""
    # note: using input shape because we know that will always have a temporal axis
    # (whether return_sequences=true/false)
    return {"neurons": np.prod(node.input_shapes[2:]), "spiking": True}


@ModelEnergy.register_layer(TimeDistributed)
def timedistributed_stats(node):
    """Compute ``TimeDistributed`` layer stats.

    Calls `.ModelEnergy.compute_layer_stats` on the wrapped layer.
    """

    # using a layer inside a TimeDistributed wrapper doesn't
    # update it's inbound_nodes. so we'll call the wrapped layer on the wrapper's
    # input to generate a node we can compute stats for. note that the wrapper's input
    # has the extra time dimension, so we slice out just the first timestep
    # (it doesn't matter what the values are)
    _ = compat.node_layer(node).layer(node.input_tensors[:, 0])

    return ModelEnergy.compute_layer_stats(
        layer=compat.node_layer(node).layer,
        node=compat.node_layer(node).layer.inbound_nodes[-1],
    )
