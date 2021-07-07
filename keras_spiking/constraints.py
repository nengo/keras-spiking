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
    non_neg : boolean
        Whether to constrain the resulting mean to be non-negative as well.
    """

    def __init__(self, axis=-1, non_neg=False):
        self.axis = axis
        self.non_neg = non_neg
        self._post_constraint = tf.keras.constraints.get("non_neg" if non_neg else None)

    def __call__(self, w):
        r = tf.repeat(
            input=tf.math.reduce_mean(w, axis=self.axis, keepdims=True),
            repeats=tf.shape(w)[self.axis],
            axis=self.axis,
        )
        if self._post_constraint is not None:
            r = self._post_constraint(r)
        return r

    def get_config(self):
        """Return config of layer (for serialization during model saving/loading)."""
        return {"axis": self.axis, "non_neg": self.non_neg}
