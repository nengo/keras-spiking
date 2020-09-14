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

**Added**

- Added the ``keras_spiking.Alpha`` filter, which provides second-order lowpass
  filtering for better noise removal for spiking layers. (`#4`_)
- Added ``keras_spiking.callbacks.DtScheduler``, which can be used to update layer
  ``dt`` parameters during training. (`#5`_)

**Changed**

- ``keras_spiking.SpikingActivation`` and ``keras_spiking.Lowpass`` now return sequences
  by default. This means that these layers will now have outputs that have the same
  number of timesteps as their inputs. This makes it easier to process create
  multi-layer spiking networks, where time is preserved throughout the network.
  The spiking fashion-MNIST example has been updated accordingly. (`#3`_)
- Layers now support multi-dimensional inputs (e.g., output of ``Conv2D`` layers).
  (`#5`_)

.. _#3: https://github.com/nengo/keras-spiking/pull/3
.. _#4: https://github.com/nengo/keras-spiking/pull/4
.. _#5: https://github.com/nengo/keras-spiking/pull/5

0.1.0 (August 14, 2020)
-----------------------

Initial release
