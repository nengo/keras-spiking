Installation
============

Installing KerasSpiking
------------------------
We recommend using ``pip`` to install KerasSpiking:

.. code-block:: bash

  pip install keras-spiking

That's it!

Requirements
------------
KerasSpiking works with Python 3.6 or later.  ``pip`` will do its best to install
all of KerasSpiking's requirements automatically.  However, if anything
goes wrong during this process you can install the requirements manually and
then try to ``pip install keras-spiking`` again.

Developer installation
----------------------
If you want to modify KerasSpiking, or get the very latest updates, you will need to
perform a developer installation:

.. code-block:: bash

  git clone https://github.com/nengo/keras-spiking.git
  pip install -e ./keras-spiking

Installing TensorFlow
---------------------
Use ``pip install tensorflow`` to install the latest version of TensorFlow. GPU support
is included in this package as of version 2.1.0.

Note that if you are using one of the non-standard TensorFlow packages (e.g.
``tensorflow-gpu``, ``tensorflow-cpu``, or ``tf-nightly``), then
``pip install keras-spiking`` will install the ``tensorflow`` package
over top of your existing TensorFlow installation,
which is probably not what you want.
To avoid this, you can install with the ``--no-deps`` option:

.. code-block:: bash

  pip install --no-deps keras-spiking

This will install only the KerasSpiking package, and you will need to manually ``pip``
install any other requirements.
This option can also be used with the developer installation method above.

In order to use TensorFlow with GPU support you will need to install the appropriate
Nvidia drivers and CUDA/cuDNN. The precise steps for accomplishing this will depend
on your system. On Linux the correct Nvidia drivers (as of TensorFlow 2.2.0) can be
installed via ``sudo apt install nvidia-driver-440``, and on Windows simply using the
most up-to-date drivers should work.  For CUDA/cuDNN we recommend using
`conda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_ to
simplify the process. ``conda install tensorflow-gpu`` will install TensorFlow as
well as all the CUDA/cuDNN requirements.  If you run into any problems, see the
`TensorFlow GPU installation instructions <https://www.tensorflow.org/install/gpu>`_
for more details.

It is also possible to build TensorFlow from source.  This is significantly
more complicated but allows you to customize the installation to your
computer, which can improve simulation speeds.

`Instructions for installing on Ubuntu or Mac OS
<https://www.tensorflow.org/install/source>`_.

`Instructions for installing on Windows
<https://www.tensorflow.org/install/source_windows>`_.
