Release history
===============

.. Changelog entries should follow this format:

   version (release date)
   ----------------------

   **section**

   - One-line description of change (link to GitHub issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Fixed
   - Deprecated
   - Removed

0.3.0 (November 8, 2021)
------------------------

*Compatible with TensorFlow 2.1.0 - 2.7.0*

**Added**

- ``LowpassCell``, ``Lowpass``, ``AlphaCell``, and ``Alpha`` layers now accept both
  ``initial_level_constraint`` and ``tau_constraint`` to customize how their
  respective parameters are constrained during training. (`#21`_)

**Changed**

- The ``tau`` time constants for ``LowpassCell``, ``Lowpass``, ``AlphaCell``, and
  ``Alpha`` are now always clipped to be positive in the forward pass rather than
  constraining the underlying trainable weights in between gradient updates. (`#21`_)
- Renamed the ``Lowpass/Alpha`` ``tau`` parameter to ``tau_initializer``, and it now
  accepts ``tf.keras.initializers.Initializer`` objects (in addition to floats, as
  before).  Renamed the ``tau_var`` weight attribute to ``tau``. (`#21`_)

**Fixed**

- ``SpikingActivation``, ``Lowpass``, and ``Alpha`` layers will now correctly use
  ``keras_spiking.default.dt``. (`#20`_)

.. _#20: https://github.com/nengo/keras-spiking/pull/20
.. _#21: https://github.com/nengo/keras-spiking/pull/21

0.2.0 (February 18, 2021)
-------------------------

*Compatible with TensorFlow 2.1.0 - 2.4.0*

**Added**

- Added the ``keras_spiking.Alpha`` filter, which provides second-order lowpass
  filtering for better noise removal for spiking layers. (`#4`_)
- Added ``keras_spiking.callbacks.DtScheduler``, which can be used to update layer
  ``dt`` parameters during training. (`#5`_)
- Added ``keras_spiking.default.dt``, which can be used to set the default ``dt``
  for all layers that don't directly specify ``dt``. (`#5`_)
- Added ``keras_spiking.regularizers.RangedRegularizer``, which can be used to apply
  some other regularizer (e.g. ``tf.keras.regularizers.L2``) with respect to some
  non-zero target point, or a range of acceptable values. This functionality has also
  been added to ``keras_spiking.regularizers.L1L2/L1/L2`` (so they can now be applied
  with respect to a single reference point or a range). (`#6`_)
- Added ``keras_spiking.regularizers.Percentile`` which computes a percentile across a
  number of examples, and regularize that statistic. (`#6`_)
- Added ``keras_spiking.ModelEnergy`` to estimate energy usage for Keras Models. (`#7`_)

**Changed**

- ``keras_spiking.SpikingActivation`` and ``keras_spiking.Lowpass`` now return sequences
  by default. This means that these layers will now have outputs that have the same
  number of timesteps as their inputs. This makes it easier to process create
  multi-layer spiking networks, where time is preserved throughout the network.
  The spiking fashion-MNIST example has been updated accordingly. (`#3`_)
- Layers now support multi-dimensional inputs (e.g., output of ``Conv2D`` layers).
  (`#5`_)

**Fixed**

- KerasSpiking layers' ``reset_state`` now resets to the value of ``get_initial_state``
  (as documented in the docstring), rather than all zeros. (`#12`_)
- Fixed a bug with ``keras_spiking.Alpha`` on TensorFlow 2.1, where a symbolic tensor
  in the initial state shape could not be converted to a Numpy array. (`#16`_)

.. _#3: https://github.com/nengo/keras-spiking/pull/3
.. _#4: https://github.com/nengo/keras-spiking/pull/4
.. _#5: https://github.com/nengo/keras-spiking/pull/5
.. _#6: https://github.com/nengo/keras-spiking/pull/6
.. _#7: https://github.com/nengo/keras-spiking/pull/7
.. _#12: https://github.com/nengo/keras-spiking/pull/12
.. _#16: https://github.com/nengo/keras-spiking/pull/16

0.1.0 (August 14, 2020)
-----------------------

*Compatible with TensorFlow 2.1.0 - 2.3.0*

Initial release
