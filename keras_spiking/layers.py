"""
Components for building spiking models in Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import smart_cond


class KerasSpikingCell(tf.keras.layers.Layer):
    """
    Base class for RNN cells in KerasSpiking.

    The important feature of this class is that it allows cells to define a different
    implementation to be used in training versus inference.

    Parameters
    ----------
    output_size : int
        Dimensionality of cell output.
    state_size : int
        Dimensionality of cell state.
    dt : float
        Length of time (in seconds) represented by one time step.
    always_use_inference : bool
        If True, this layer will use its `.call_inference` behaviour during training,
        rather than `.call_training`.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        output_size,
        state_size=None,
        dt=0.001,
        always_use_inference=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dt = dt
        self.always_use_inference = always_use_inference

        self.output_size = output_size
        self.state_size = self.output_size if state_size is None else state_size

    def call(self, inputs, states, training=None):
        """
        Call function that defines a different forward pass during training
        versus inference.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        return smart_cond.smart_cond(
            tf.logical_and(tf.cast(training, tf.bool), not self.always_use_inference),
            lambda: self.call_training(inputs, states),
            lambda: self.call_inference(inputs, states),
        )

    def call_training(self, inputs, states):
        """Compute layer output when training and ``always_use_inference=False``."""

        raise NotImplementedError("Subclass must implement `call_training`")

    def call_inference(self, inputs, states):
        """Compute layer output when testing or ``always_use_inference=True``."""

        raise NotImplementedError("Subclass must implement `call_inference`")


class KerasSpikingLayer(tf.keras.layers.Layer):
    """
    Base class for KerasSpiking layers.

    The main role of this class is to wrap a KerasSpikingCell in a
    ``tf.keras.layers.RNN``.

    Parameters
    ----------
    dt : float
        Length of time (in seconds) represented by one time step.
    return_sequences : bool
        Whether to return the full sequence of output values (default),
        or just the values on the last timestep.
    return state : bool
        Whether to return the state in addition to the output.
    stateful : bool
        If False (default), each time the layer is called it will begin from the same
        initial conditions. If True, each call will resume from the terminal state of
        the previous call (``my_layer.reset_states()`` can be called to reset the state
        to initial conditions).
    unroll : bool
        If True, the network will be unrolled, else a symbolic loop will be used.
        Unrolling can speed up computations, although it tends to be more
        memory-intensive. Unrolling is only suitable for short sequences.
    time_major : bool
        The shape format of the input and output tensors. If True, the inputs and
        outputs will be in shape ``(timesteps, batch, ...)``, whereas in the False case,
        it will be ``(batch, timesteps, ...)``. Using ``time_major=True`` is a bit more
        efficient because it avoids transposes at the beginning and end of the layer
        calculation. However, most TensorFlow data is batch-major, so by default this
        layer accepts input and emits output in batch-major form.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        dt=0.001,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dt = dt
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.layer = None

    def build_cell(self, input_shapes):
        """Create and return the RNN cell."""
        raise NotImplementedError("Subclass must implement `build_cell`")

    def build(self, input_shapes):
        """
        Builds the RNN/cell layers contained within this layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shapes)

        # we initialize these here, rather than in ``__init__``, so that we can
        # determine ``units`` automatically
        cell = self.build_cell(input_shapes)
        self.layer = tf.keras.layers.RNN(
            cell,
            return_sequences=self.return_sequences,
            return_state=self.return_state,
            stateful=self.stateful,
            unroll=self.unroll,
            time_major=self.time_major,
        )

        self.layer.build(input_shapes)

    def call(self, inputs, training=None, initial_state=None, constants=None):
        """
        Apply this layer to inputs.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """
        return self.layer.call(
            inputs, training=training, initial_state=initial_state, constants=constants
        )

    def reset_states(self, states=None):
        """
        Reset the internal state of the layer (only necessary if ``stateful=True``).

        Parameters
        ----------
        states : `~numpy.ndarray`
            Optional state array that can be used to override the values returned by
            `.SpikingActivationCell.get_initial_state`.
        """
        self.layer.reset_states(states=states)

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        cfg = super().get_config()
        cfg.update(
            dict(
                dt=self.dt,
                return_sequences=self.return_sequences,
                return_state=self.return_state,
                stateful=self.stateful,
                unroll=self.unroll,
                time_major=self.time_major,
            )
        )

        return cfg


class SpikingActivationCell(KerasSpikingCell):
    """
    RNN cell for converting an arbitrary activation function to a spiking equivalent.

    Neurons will spike at a rate proportional to the output of the base activation
    function. For example, if the activation function is outputting a value of 10, then
    the wrapped SpikingActivationCell will output spikes at a rate of 10Hz (i.e., 10
    spikes per 1 simulated second, where 1 simulated second is equivalent to ``1/dt``
    time steps). Each spike will have height ``1/dt`` (so that the integral of the
    spiking output will be the same as the integral of the base activation output).
    Note that if the base activation is outputting a negative value then the spikes
    will have height ``-1/dt``. Multiple spikes per timestep are also possible, in
    which case the output will be ``n/dt`` (where ``n`` is the number of spikes).

    Notes
    -----

    This cell needs to be wrapped in a ``tf.keras.layers.RNN``, like

    .. testcode::

        my_layer = tf.keras.layers.RNN(
            keras_spiking.SpikingActivationCell(units=10, activation=tf.nn.relu)
        )

    Parameters
    ----------
    units : int
        Dimensionality of layer.
    activation : callable
        Activation function to be converted to spiking equivalent.
    dt : float
        Length of time (in seconds) represented by one time step.
    seed : int
        Seed for random state initialization.
    spiking_aware_training : bool
        If True (default), use the spiking activation function
        for the forward pass and the base activation function for the backward pass.
        If False, use the base activation function for the forward and
        backward pass during training.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        units,
        activation,
        *,
        dt=0.001,
        seed=None,
        spiking_aware_training=True,
        **kwargs
    ):
        super().__init__(
            output_size=units,
            dt=dt,
            always_use_inference=spiking_aware_training,
            **kwargs,
        )
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.seed = seed
        self.spiking_aware_training = spiking_aware_training

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        Set up initial spiking state.

        Initial state is chosen from a uniform distribution, seeded based on the seed
        passed on construction (if one was given).

        Note: state will be initialized automatically, user does not need to call this
        themselves.
        """
        seed = (
            tf.random.uniform((), maxval=np.iinfo(np.int32).max, dtype=tf.int32)
            if self.seed is None
            else self.seed
        )

        # TODO: we could make the initial voltages trainable
        return tf.random.stateless_uniform(
            (batch_size, self.output_size), seed=(seed, seed), dtype=dtype
        )

    def call_training(self, inputs, states):
        return self.activation(inputs), states

    @tf.custom_gradient
    def call_inference(self, inputs, states):
        """
        Compute spiking output, with custom gradient for spiking aware training.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Input to the activation function.
        voltage : ``tf.Tensor``
            Spiking voltage state.

        Returns
        -------
        spikes : ``tf.Tensor``
            Output spike values (0 or ``n/dt`` for each element in ``inputs``, where
            ``n`` is the number of spikes).
        voltage : ``tf.Tensor``
            Updated voltage state.
        grad : callable
            Custom gradient function for spiking aware training.
        """
        voltage = states[0]

        with tf.GradientTape() as g:
            g.watch(inputs)
            rates = self.activation(inputs)
        voltage = voltage + rates * self.dt
        n_spikes = tf.floor(voltage)
        voltage -= n_spikes
        spikes = n_spikes / self.dt

        def grad(grad_spikes, grad_voltage):
            return (
                g.gradient(rates, inputs) * grad_spikes,
                None,
            )

        return (spikes, (voltage,)), grad

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        cfg = super().get_config()
        cfg.update(
            dict(
                units=self.units,
                activation=tf.keras.activations.serialize(self.activation),
                dt=self.dt,
                seed=self.seed,
                spiking_aware_training=self.spiking_aware_training,
            )
        )

        return cfg


class SpikingActivation(KerasSpikingLayer):
    """
    Layer for converting an arbitrary activation function to a spiking equivalent.

    Neurons will spike at a rate proportional to the output of the base activation
    function. For example, if the activation function is outputting a value of 10, then
    the wrapped SpikingActivationCell will output spikes at a rate of 10Hz (i.e., 10
    spikes per 1 simulated second, where 1 simulated second is equivalent to ``1/dt``
    time steps). Each spike will have height ``1/dt`` (so that the integral of the
    spiking output will be the same as the integral of the base activation output).
    Note that if the base activation is outputting a negative value then the spikes
    will have height ``-1/dt``. Multiple spikes per timestep are also possible, in
    which case the output will be ``n/dt`` (where ``n`` is the number of spikes).

    When applying this layer to an input, make sure that the input has a time axis
    (the ``time_major`` option controls whether it comes before or after the batch
    axis). The spiking output will be computed along the time axis.
    The number of simulation timesteps will depend on the length of that time axis.
    The number of timesteps does not need to be the same during
    training/evaluation/inference. In particular, it may be more efficient
    to use one timestep during training and multiple timesteps during inference
    (often with ``spiking_aware_training=False``, and ``apply_during_training=False``
    on any `.Lowpass` layers).

    Notes
    -----
    This is equivalent to
    ``tf.keras.layers.RNN(SpikingActivationCell(...) ...)``, it just takes care of
    the RNN construction automatically.

    Parameters
    ----------
    activation : callable
        Activation function to be converted to spiking equivalent.
    dt : float
        Length of time (in seconds) represented by one time step.
    seed : int
        Seed for random state initialization.
    spiking_aware_training : bool
        If True (default), use the spiking activation function
        for the forward pass and the base activation function for the backward pass.
        If False, use the base activation function for the forward and
        backward pass during training.
    return_sequences : bool
        Whether to return the full sequence of output spikes (default),
        or just the spikes on the last timestep.
    return state : bool
        Whether to return the state in addition to the output.
    stateful : bool
        If False (default), each time the layer is called it will begin from the same
        initial conditions. If True, each call will resume from the terminal state of
        the previous call (``my_layer.reset_states()`` can be called to reset the state
        to initial conditions).
    unroll : bool
        If True, the network will be unrolled, else a symbolic loop will be used.
        Unrolling can speed up computations, although it tends to be more
        memory-intensive. Unrolling is only suitable for short sequences.
    time_major : bool
        The shape format of the input and output tensors. If True, the inputs and
        outputs will be in shape ``(timesteps, batch, ...)``, whereas in the False case,
        it will be ``(batch, timesteps, ...)``. Using ``time_major=True`` is a bit more
        efficient because it avoids transposes at the beginning and end of the layer
        calculation. However, most TensorFlow data is batch-major, so by default this
        layer accepts input and emits output in batch-major form.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        activation,
        *,
        dt=0.001,
        seed=None,
        spiking_aware_training=True,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs
    ):
        super().__init__(
            dt=dt,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs,
        )

        self.activation = tf.keras.activations.get(activation)
        self.seed = seed
        self.spiking_aware_training = spiking_aware_training

    def build_cell(self, input_shapes):
        return SpikingActivationCell(
            units=input_shapes[-1],
            activation=self.activation,
            dt=self.dt,
            seed=self.seed,
            spiking_aware_training=self.spiking_aware_training,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                activation=tf.keras.activations.serialize(self.activation),
                seed=self.seed,
                spiking_aware_training=self.spiking_aware_training,
            )
        )

        return cfg


class LowpassCell(KerasSpikingCell):
    """
    RNN cell for a lowpass filter.

    The initial filter state and filter time constants are both trainable parameters.
    However, if ``apply_during_training=False`` then the parameters are not part
    of the training loop, and so will never be updated.

    Notes
    -----

    This cell needs to be wrapped in a ``tf.keras.layers.RNN``, like

    .. testcode::

        my_layer = tf.keras.layers.RNN(keras_spiking.LowpassCell(units=10, tau=0.01))

    Parameters
    ----------
    units : int
        Dimensionality of layer.
    tau : float
        Time constant of filter (in seconds).
    dt : float
        Length of time (in seconds) represented by one time step.
    apply_during_training : bool
        If False, this layer will effectively be ignored during training (this
        often makes sense in concert with the swappable training behaviour in, e.g.,
        `.SpikingActivation`, since if the activations are not spiking during training
        then we often don't need to filter them either).
    level_initializer : str or ``tf.keras.initializers.Initializer``
        Initializer for filter state.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        units,
        tau,
        *,
        dt=0.001,
        # TODO: better name for this parameter?
        apply_during_training=True,
        level_initializer="zeros",
        **kwargs
    ):
        super().__init__(
            output_size=units,
            dt=dt,
            always_use_inference=apply_during_training,
            **kwargs,
        )

        if tau <= 0:
            raise ValueError("tau must be a positive number")

        self.units = units
        self.tau = tau
        self.apply_during_training = apply_during_training
        self.level_initializer = tf.initializers.get(level_initializer)

        # apply ZOH discretization
        tau = np.exp(-dt / tau)

        # compute inverse sigmoid of tau, so that when we apply the sigmoid
        # later we'll get the tau value specified
        self.smoothing_init = np.log(tau / (1 - tau))

    def build(self, input_shapes):
        """Build parameters associated with this layer."""

        super().build(input_shapes)

        self.initial_level = self.add_weight(
            name="initial_level",
            shape=(1, self.output_size),
            initializer=self.level_initializer,
            trainable=self.apply_during_training,
        )

        self.smoothing = self.add_weight(
            name="level_smoothing",
            shape=(1, self.state_size),
            initializer=tf.initializers.constant(
                np.ones(self.state_size) * self.smoothing_init
            ),
            trainable=self.apply_during_training,
        )

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get initial filter state."""
        return tf.tile(self.initial_level, (batch_size, 1))

    def call_inference(self, inputs, states):
        smoothing = tf.nn.sigmoid(self.smoothing)
        x = (1 - smoothing) * inputs + smoothing * states[0]
        return x, (x,)

    def call_training(self, inputs, states):
        return inputs, states

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""

        cfg = super().get_config()
        cfg.update(
            dict(
                units=self.units,
                tau=self.tau,
                dt=self.dt,
                apply_during_training=self.apply_during_training,
                level_initializer=tf.keras.initializers.serialize(
                    self.level_initializer
                ),
            )
        )

        return cfg


class Lowpass(KerasSpikingLayer):
    """
    Layer implementing a lowpass filter.

    The initial filter state and filter time constants are both trainable parameters.
    However, if ``apply_during_training=False`` then the parameters are not part
    of the training loop, and so will never be updated.

    When applying this layer to an input, make sure that the input has a time axis
    (the ``time_major`` option controls whether it comes before or after the batch
    axis).

    Notes
    -----
    This is equivalent to
    ``tf.keras.layers.RNN(LowpassCell(...) ...)``, it just takes care of
    the RNN construction automatically.

    Parameters
    ----------
    tau : float
        Time constant of filter (in seconds).
    dt : float
        Length of time (in seconds) represented by one time step.
    apply_during_training : bool
        If False, this layer will effectively be ignored during training (this
        often makes sense in concert with the swappable training behaviour in, e.g.,
        `.SpikingActivation`, since if the activations are not spiking during training
        then we often don't need to filter them either).
    level_initializer : str or ``tf.keras.initializers.Initializer``
        Initializer for filter state.
    return_sequences : bool
        Whether to return the full sequence of filtered output (default),
        or just the output on the last timestep.
    return state : bool
        Whether to return the state in addition to the output.
    stateful : bool
        If False (default), each time the layer is called it will begin from the same
        initial conditions. If True, each call will resume from the terminal state of
        the previous call (``my_layer.reset_states()`` can be called to reset the state
        to initial conditions).
    unroll : bool
        If True, the network will be unrolled, else a symbolic loop will be used.
        Unrolling can speed up computations, although it tends to be more
        memory-intensive. Unrolling is only suitable for short sequences.
    time_major : bool
        The shape format of the input and output tensors. If True, the inputs and
        outputs will be in shape ``(timesteps, batch, ...)``, whereas in the False case,
        it will be ``(batch, timesteps, ...)``. Using ``time_major=True`` is a bit more
        efficient because it avoids transposes at the beginning and end of the layer
        calculation. However, most TensorFlow data is batch-major, so by default this
        layer accepts input and emits output in batch-major form.
    kwargs : dict
        Passed on to `tf.keras.layers.Layer
        <https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer>`_.
    """

    def __init__(
        self,
        tau,
        *,
        dt=0.001,
        apply_during_training=True,
        level_initializer="zeros",
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs
    ):
        super().__init__(
            dt=dt,
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            unroll=unroll,
            time_major=time_major,
            **kwargs,
        )

        self.tau = tau
        self.apply_during_training = apply_during_training
        self.level_initializer = tf.keras.initializers.get(level_initializer)

    def build_cell(self, input_shapes):
        return LowpassCell(
            units=input_shapes[-1],
            tau=self.tau,
            dt=self.dt,
            apply_during_training=self.apply_during_training,
            level_initializer=self.level_initializer,
        )

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                tau=self.tau,
                apply_during_training=self.apply_during_training,
                level_initializer=tf.keras.initializers.serialize(
                    self.level_initializer
                ),
            )
        )

        return cfg
