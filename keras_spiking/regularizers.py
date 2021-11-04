"""
Regularization methods designed to work with spiking layers.
"""

import warnings

import tensorflow as tf

try:
    import tensorflow_probability as tfp

    HAS_TFP = True
except ImportError:  # pragma: no cover
    HAS_TFP = False


class RangedRegularizer(tf.keras.regularizers.Regularizer):
    """
    A regularizer that penalizes values that fall outside a range.

    This allows regularized values to fall anywhere within the range, as opposed to
    standard regularizers that penalize any departure from some fixed point.

    Parameters
    ----------
    target : float or tuple
        The value that we want the regularized outputs to be driven towards. Can be a
        float, in which case all outputs will be driven towards that value, or a tuple
        specifying a range ``(min, max)``, in which case outputs outside that range
        will be driven towards that range (but outputs within the range will not be
        penalized).
    regularizer: ``tf.keras.regularizers.Regularizer``
        Regularization penalty that will be applied to the outputs with respect to
        ``target``.
    """

    def __init__(self, target=0, regularizer=tf.keras.regularizers.L1L2(l2=0.01)):
        super().__init__()

        self.regularizer = regularizer
        self.target = target
        if isinstance(target, (list, tuple)):
            if len(target) != 2:
                raise ValueError(
                    "Target ranges should be specified as a tuple with two elements "
                    f"`(min, max)` (got {len(target)} elements)"
                )
            self.minimum, self.maximum = target
        else:
            self.minimum = self.maximum = target

        if self.minimum > self.maximum:
            raise ValueError("`minimum` cannot exceed `maximum`")

    def __call__(self, x):
        if self.minimum == self.maximum:
            error = x - self.maximum
        else:
            error = tf.nn.relu(self.minimum - x) + tf.nn.relu(x - self.maximum)

        error = self.regularizer(error)

        return error

    def get_config(self):
        """Return config (for serialization during model saving/loading)."""

        return dict(
            target=self.target,
            regularizer=tf.keras.regularizers.serialize(self.regularizer),
        )

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the serialized ``config``."""

        config["regularizer"] = tf.keras.regularizers.deserialize(config["regularizer"])
        return cls(**config)


class L1L2(RangedRegularizer):
    """
    A version of ``tf.keras.regularizers.L1L2`` that allows the user to specify a
    nonzero target output.

    Parameters
    ----------
    l1 : float
        Weight on L1 regularization penalty.
    l2 : float
        Weight on L2 regularization penalty.
    target : float or tuple
        The value that we want the regularized outputs to be driven towards. Can be a
        float, in which case all outputs will be driven towards that value, or a tuple
        specifying a range ``(min, max)``, in which case outputs outside that range
        will be driven towards that range (but outputs within the range will not be
        penalized).
    """

    def __init__(self, l1=0.0, l2=0.0, target=0, **kwargs):
        super().__init__(
            target=target,
            regularizer=tf.keras.regularizers.L1L2(l1=l1, l2=l2, **kwargs),
        )

        if l1 == 0 and l2 == 0:
            warnings.warn("Regularization weight is zero, it will have no effect")

    def get_config(self):
        """Return config (for serialization during model saving/loading)."""

        cfg = self.regularizer.get_config()
        cfg["target"] = self.target

        return cfg

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the serialized ``config``."""

        return cls(**config)


class L1(L1L2):
    """
    A version of ``tf.keras.regularizers.L1`` that allows the user to specify a
    nonzero target output.

    Parameters
    ----------
    l1 : float
        Weight on L1 regularization penalty.
    target : float or tuple
        The value that we want the regularized outputs to be driven towards. Can be a
        float, in which case all outputs will be driven towards that value, or a tuple
        specifying a range ``(min, max)``, in which case outputs outside that range
        will be driven towards that range (but outputs within the range will not be
        penalized).
    """

    def __init__(self, l1=0.01, target=0, **kwargs):
        super().__init__(l1=l1, target=target, **kwargs)


class L2(L1L2):
    """
    A version of ``tf.keras.regularizers.L2`` that allows the user to specify a
    nonzero target output.

    Parameters
    ----------
    l2 : float
        Weight on L2 regularization penalty.
    target : float or tuple
        The value that we want the regularized outputs to be driven towards. Can be a
        float, in which case all outputs will be driven towards that value, or a tuple
        specifying a range ``(min, max)``, in which case outputs outside that range
        will be driven towards that range (but outputs within the range will not be
        penalized).
    """

    def __init__(self, l2=0.01, target=0, **kwargs):
        super().__init__(l2=l2, target=target, **kwargs)


class Percentile(L1L2):
    """
    A regularizer that penalizes a percentile of a tensor.

    This regularizer finds the requested ``percentile`` of the data over the ``axis``,
    and then applies a regularizer to the percentile values with respect to ``target``.
    This can be useful as it is makes the computed regularization penalty more invariant
    to outliers.

    Parameters
    ----------
    percentile : float
        Percentile to compute over the ``axis``. Defaults to 100, which is equivalent to
        taking the maximum across the specified ``axis``.

        .. note:: For ``percentile != 100``, requires
           `tensorflow-probability <https://www.tensorflow.org/probability/install>`_.
    axis : int or tuple of int
        Axis or axes to take the percentile over.
    target : float or tuple
        The value that we want the regularized outputs to be driven towards. Can be a
        float, in which case all outputs will be driven towards that value, or a tuple
        specifying a range ``(min, max)``, in which case outputs outside that range
        will be driven towards that range (but outputs within the range will not be
        penalized).
    l1 : float
        Weight on L1 regularization penalty applied to percentiles.
    l2 : float
        Weight on L2 regularization penalty applied to percentiles.

    Examples
    --------

    In the following example, we use `.Percentile` to ensure the neuron activities
    (a.k.a., firing rates) fall in the desired range of 5-10 Hz when computing the
    product of two inputs.

    .. testcode::

        train_x = np.random.uniform(-1, 1, size=(1024 * 100, 2))
        train_y = train_x[:, :1] * train_x[:, 1:]
        test_x = np.random.uniform(-1, 1, size=(128, 2))
        test_y = test_x[:, :1] * test_x[:, 1:]

        # train using one timestep, to speed things up
        train_seq = train_x[:, None]

        # test using 10 timesteps
        n_steps = 10
        test_seq = np.tile(test_x[:, None], (1, n_steps, 1))

        inp = x = tf.keras.Input((None, 2))
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50))(x)
        x = spikes = keras_spiking.SpikingActivation(
            "relu",
            dt=1,
            activity_regularizer=keras_spiking.regularizers.Percentile(
                target=(5, 10), l2=0.01
            ),
        )(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inp, (x, spikes))

        model.compile(
            # note: we use a dict to specify loss/metrics because we only want to
            # apply these to the final dense output, not the spike layer
            optimizer="rmsprop", loss={"dense_1": "mse"}, metrics={"dense_1": "mae"}
        )
        model.fit(train_seq, train_y, epochs=5)

        outputs, spikes = model.predict(test_seq)

        # estimate rates by averaging over time
        rates = spikes.mean(axis=1)
        max_rates = rates.max(axis=0)
        print("Max rates: %s, %s" % (max_rates.mean(), max_rates.std()))

        error = np.mean(np.abs(outputs - test_y))
        print("MAE: %s" % (error,))


    .. testoutput::
       :hide:

       ...
       Max rates: ...
       MAE: ...
    """

    def __init__(
        self,
        percentile=100,
        axis=0,
        target=0,
        l1=0,
        l2=0,
    ):
        super().__init__(target=target, l1=l1, l2=l2)

        self.axis = axis
        self.percentile = float(percentile)

        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("`percentile` must be in the range [0, 100]")

        if self.percentile != 100 and not HAS_TFP:
            raise ValueError(
                "`percentile` < 100 requires tensorflow-probability to be installed"
            )

    def __call__(self, x):
        percentile = (
            tf.math.reduce_max(x, axis=self.axis)
            if self.percentile == 100
            else tfp.stats.percentile(
                x, self.percentile, axis=self.axis, interpolation="linear"
            )
        )

        return super().__call__(percentile)

    def get_config(self):
        """Return config (for serialization during model saving/loading)."""

        cfg = super().get_config()
        cfg.update(
            dict(
                percentile=self.percentile,
                axis=self.axis,
            )
        )
        return cfg
