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

0.1.1 (unreleased)
------------------

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

.. _#3: https://github.com/nengo/keras-spiking/pull/3
.. _#4: https://github.com/nengo/keras-spiking/pull/4
.. _#5: https://github.com/nengo/keras-spiking/pull/5
.. _#6: https://github.com/nengo/keras-spiking/pull/6
.. _#12: https://github.com/nengo/keras-spiking/pull/12

0.1.0 (August 14, 2020)
-----------------------

*Compatible with TensorFlow 2.1.0 - 2.3.0*

Initial release
