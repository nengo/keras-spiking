"""
Callbacks for use with KerasSpiking models.
"""

import tensorflow as tf


class DtScheduler(tf.keras.callbacks.Callback):
    """
    A callback for updating Layer ``dt`` attributes during training.

    This uses the same scheduler interface as `TensorFlow's learning rate schedulers
    <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules>`_,
    so any of those built-in schedules can be used to adjust ``dt``, or a custom
    function implementing the same interface.

    When using this functionality, ``dt`` should be initialized as a ``tf.Variable``,
    and that Variable should be passed as the ``dt`` parameter to any Layers that
    should be affected by this callback.

    For example:

    .. testcode::

        dt = tf.Variable(1.0)

        inp = tf.keras.Input((None, 10))
        x = keras_spiking.SpikingActivation("relu", dt=dt)(inp)
        x = keras_spiking.Lowpass(0.1, dt=dt)(x)
        model = tf.keras.Model(inp, x)

        callback = keras_spiking.callbacks.DtScheduler(
            dt, tf.optimizers.schedules.ExponentialDecay(
                1.0, decay_steps=5, decay_rate=0.9
            )
        )

        model.compile(loss="mse", optimizer="sgd")
        model.fit(
            np.ones((100, 2, 10)),
            np.ones((100, 2, 10)),
            epochs=10,
            batch_size=20,
            callbacks=[callback],
        )

    .. testoutput::
        :hide:

        ...

    Parameters
    ----------
    dt : ``tf.Variable``
        Variable representing ``dt`` that has been passed to other Layers.
    scheduler : ``tf.optimizers.schedules.LearningRateSchedule``
        A schedule class that will update ``dt`` based on the training step (one
        training step is one minibatch worth of training).
    verbose : bool
        If True, print out some information about ``dt`` updates during training.

    Notes
    -----
    Because Variable values persist over time, any changes made to ``dt`` by this
    callback will persist after training completes. For example, if you call ``fit``
    with this callback and then ``predict`` later on, that ``predict`` call will be
    using the last ``dt`` value set by this callback.
    """

    def __init__(self, dt, scheduler, verbose=False):
        super().__init__()

        if not isinstance(dt, tf.Variable):
            raise TypeError("DtScheduler requires `dt` to be stored as a `tf.Variable`")

        self.dt = dt
        self.scheduler = scheduler
        self.curr_epoch = None
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        """Keep track of the current epoch so we can count the total number of steps."""

        self.curr_epoch = epoch
        if self.verbose:
            print(f"DtScheduler epoch={epoch} dt={self.dt.numpy():.4f}")

    def on_train_batch_begin(self, batch, logs=None):
        """Update ``dt`` variable based on the current training step."""

        assert self.curr_epoch is not None
        step = self.curr_epoch * self.params["steps"] + batch
        self.dt.assign(self.scheduler(step))
