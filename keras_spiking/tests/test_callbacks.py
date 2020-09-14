import pytest
import tensorflow as tf

from keras_spiking import callbacks, layers


def test_dt_scheduling(allclose, capsys):
    dt_var = tf.Variable(0.0, trainable=False)
    scheduler = tf.optimizers.schedules.PiecewiseConstantDecay(
        [0, 16, 32], [1.0, 0.5, 0.25, 0.1]
    )

    class DtChecker(tf.keras.callbacks.Callback):
        def __init__(self, var):
            self.var = var
            self.curr_epoch = None

        def on_epoch_begin(self, epoch, logs=None):
            self.curr_epoch = epoch

        def on_train_batch_end(self, batch, logs=None):
            step = self.curr_epoch * self.params["steps"] + batch

            if step == 0:
                assert allclose(self.var.numpy(), 1)
            elif step <= 16:
                assert allclose(self.var.numpy(), 0.5)
            elif step <= 32:
                assert allclose(self.var.numpy(), 0.25)
            else:
                assert allclose(self.var.numpy(), 0.1)

    inp = x = tf.keras.Input((None, 10))
    x = tf.keras.layers.Dense(10)(x)
    x = layers.SpikingActivation("relu", dt=dt_var)(x)
    x = layers.Lowpass(0.1, dt=dt_var)(x)
    x = layers.Alpha(0.1, dt=dt_var)(x)
    model = tf.keras.Model(inp, x)
    model.compile(optimizer="sgd", loss="mse")

    model.fit(
        tf.ones((32, 5, 10)),
        tf.ones((32, 5, 10)),
        callbacks=[
            DtChecker(dt_var),
            callbacks.DtScheduler(dt_var, scheduler, verbose=True),
        ],
        epochs=10,
        batch_size=2,
    )

    assert "DtScheduler epoch=0 dt=0.0000" in capsys.readouterr().out


def test_dt_schedule_errors():
    with pytest.raises(TypeError, match="stored as a `tf.Variable`"):
        callbacks.DtScheduler(tf.constant(0), None)
