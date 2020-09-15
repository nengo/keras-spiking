"""
Configuration options for KerasSpiking layers.
"""


class DefaultManager:
    """
    Manages the default parameter values for KerasSpiking layers.

    Notes
    -----
    Do not instantiate this class directly, instead access it through
    ``keras_spiking.default``.

    Parameters
    ----------
    dt : float
        Length of time (in seconds) represented by one time step. Defaults to 0.001s.
    """

    def __init__(self, dt=0.001):
        self.dt = dt


default = DefaultManager()
