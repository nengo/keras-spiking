.. image:: https://img.shields.io/travis/com/nengo/keras-spiking/master.svg
  :target: https://travis-ci.com/nengo/keras-spiking
  :alt: Travis-CI build status

.. image:: https://img.shields.io/codecov/c/github/nengo/keras-spiking/master.svg
  :target: https://codecov.io/gh/nengo/keras-spiking
  :alt: Test coverage

*************
Keras Spiking
*************

Keras Spiking provides tools for incorporating spiking neuron activation functions
directly within the Keras framework. The main feature is
``keras_spiking.SpikingActivation``, which can be used to transform
any activation function into a spiking equivalent. For example, we can translate a
non-spiking model, such as

.. code-block:: python

    inp = tf.keras.Input((5,))
    dense = tf.keras.layers.Dense(10)(inp)
    act = tf.keras.layers.Activation("relu")(dense)
    model = tf.keras.Model(inp, act)

into the spiking equivalent:

.. code-block:: python

    # add time dimension to inputs
    inp = tf.keras.Input((None, 5))
    # use Keras' TimeDistributed wrapper to apply Dense over time
    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))(inp)
    # replace Activation with SpikingActivation
    act = keras_spiking.SpikingActivation("relu")(dense)
    model = tf.keras.Model(inp, act)

Models with SpikingActivation layers can be optimized and evaluated in the same way as
any other Keras model. They will automatically take advantage of Keras Spiking's
"spiking aware training": using the spiking activations on the forward pass and the
non-spiking (differentiable) activation function on the backwards pass.

Keras Spiking also includes various tools to assist in the training of spiking models,
such as additional `regularizers
<https://www.nengo.ai/keras-spiking/reference.html#module-keras_spiking.regularizers>`_
and `filtering layers
<https://www.nengo.ai/keras-spiking/reference.html#module-keras_spiking.layers>`_.

If you are interested in building and optimizing spiking neuron models, you may also
be interested in `NengoDL <https://www.nengo.ai/nengo-dl>`_. See
`this page <https://www.nengo.ai/keras-spiking/nengo-dl-comparison.html>`_ for a
comparison of the different use cases supported by these two packages.

**Documentation**

Check out the `documentation <https://www.nengo.ai/keras-spiking/>`_ for

- `Installation instructions
  <https://www.nengo.ai/keras-spiking/installation.html>`_
- `More detailed example introducing the features of Keras Spiking
  <https://www.nengo.ai/keras-spiking/examples/spiking-fashion-mnist.html>`_
- `API reference <https://www.nengo.ai/keras-spiking/reference.html>`_
