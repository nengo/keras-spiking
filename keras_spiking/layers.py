"""
Components for building spiking models in Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import smart_cond


class SpikingActivationCell(tf.keras.layers.Layer):
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
        dt=0.001,
        seed=None,
        # TODO: should this default to True or False?
        spiking_aware_training=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.dt = dt
        self.seed = seed
        self.spiking_aware_training = spiking_aware_training

        self.output_size = (units,)
        self.state_size = (units,)

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
            (batch_size, self.units), seed=(seed, seed), dtype=dtype
        )

    def call(self, inputs, states, training=None):
        """
        Compute layer output.
        """

        if training is None:
            training = tf.keras.backend.learning_phase()

        voltage = states[0]

        return smart_cond.smart_cond(
            tf.logical_and(tf.cast(training, tf.bool), not self.spiking_aware_training),
            lambda: (self.activation(inputs), voltage),
            lambda: self._compute_spikes(inputs, voltage),
        )

    @tf.custom_gradient
    def _compute_spikes(self, inputs, voltage):
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

        return (spikes, voltage), grad

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


class SpikingActivation(tf.keras.layers.Layer):
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
        Whether to return the last output in the output sequence (default), or the
        full sequence.
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
        dt=0.001,
        seed=None,
        spiking_aware_training=True,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unroll=False,
        time_major=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.activation = tf.keras.activations.get(activation)
        self.dt = dt
        self.seed = seed
        self.spiking_aware_training = spiking_aware_training
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.unroll = unroll
        self.time_major = time_major
        self.layer = None

    def build(self, input_shapes):
        """
        Builds the RNN/SpikingActivationCell layers contained within this layer.

        Notes
        -----
        This method should not be called manually; rather, use the implicit layer
        callable behaviour (like ``my_layer(inputs)``), which will apply this method
        with some additional bookkeeping.
        """

        super().build(input_shapes)

        # we initialize these here, rather than in ``__init__``, so that we can
        # determine ``units`` automatically
        self.layer = tf.keras.layers.RNN(
            SpikingActivationCell(
                activation=self.activation,
                units=input_shapes[-1],
                dt=self.dt,
                seed=self.seed,
                spiking_aware_training=self.spiking_aware_training,
            ),
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
                activation=tf.keras.activations.serialize(self.activation),
                dt=self.dt,
                seed=self.seed,
                spiking_aware_training=self.spiking_aware_training,
                return_sequences=self.return_sequences,
                return_state=self.return_state,
                stateful=self.stateful,
                unroll=self.unroll,
                time_major=self.time_major,
            )
        )

        return cfg
