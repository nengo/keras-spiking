"""
Regularization methods designed to work with spiking layers.
"""

import tensorflow as tf


class L1L2(tf.keras.regularizers.L1L2):
    """
    A version of ``tf.keras.regularizers.L1L2`` that allows the user to specify a
    nonzero target output.

    Parameters
    ----------
    l1 : float
        Weight on L1 regularization penalty.
    l2 : float
        Weight on L2 regularization penalty.
    target : float
        Target output value (values will be penalized based on their distance from
        this point).
    """

    def __init__(self, l1=0.0, l2=0.0, target=0, **kwargs):
        super().__init__(l1=l1, l2=l2, **kwargs)

        self.target = target

    def __call__(self, x):
        return super().__call__(x - self.target)

    def get_config(self):
        """Return config (for serialization during model saving/loading)."""

        cfg = super().get_config()
        cfg["target"] = self.target

        return cfg


class L1(L1L2):
    """
    A version of ``tf.keras.regularizers.L1`` that allows the user to specify a
    nonzero target output.

    Parameters
    ----------
    l1 : float
        Weight on L1 regularization penalty.
    target : float
        Target output value (values will be penalized based on their distance from
        this point).
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
    target : float
        Target output value (values will be penalized based on their distance from
        this point).
    """

    def __init__(self, l2=0.01, target=0, **kwargs):
        super().__init__(l2=l2, target=target, **kwargs)
