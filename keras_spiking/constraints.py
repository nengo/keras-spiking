"""
Custom constraints for weight tensors in Keras models.
"""

import tensorflow as tf


class Mean(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be their mean.

    Parameters
    ----------
    axis : int
        Axis used to compute the mean and repeat its value. Defaults to the last axis.
    """

    def __init__(self, axis=-1):
        self.axis = axis

    def __call__(self, w):
        return tf.repeat(
            input=tf.math.reduce_mean(w, axis=self.axis, keepdims=True),
            repeats=tf.shape(w)[self.axis],
            axis=self.axis,
        )

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""
        return {"axis": self.axis}
