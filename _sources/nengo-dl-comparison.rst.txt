KerasSpiking versus NengoDL
===========================

If you are interested in combining spiking neurons and deep learning methods, you may
be familiar with `NengoDL <https://www.nengo.ai/nengo-dl>`_ (and wondering what the
difference is between KerasSpiking and NengoDL).

The short answer is that KerasSpiking is designed to be a lightweight, minimal
implementation of spiking behaviour that integrates very transparently into Keras.
It is designed to get you up and running on building a spiking model with very little
overhead.

NengoDL provides much more robust, fully-featured tools for building spiking models.
More neuron types, more synapse types, more complex network architectures, more of
everything basically. However, all of those extra features require a more significant
departure from the underlying TensorFlow/Keras API. There is more of a learning curve to
getting started with NengoDL, and the resulting code looks less like standard
Keras code (although it is still designed to feel familiar to Keras users).

One particularly significant distinction is that KerasSpiking does not really
integrate with the rest of the Nengo ecosystem (e.g., it cannot run models built with
the Nengo API, and models built with KerasSpiking cannot run on other Nengo platforms).
In contrast, NengoDL can run any Nengo model, and models optimized in NengoDL can
be run on other Nengo platforms (such as custom neuromorphic hardware, like NengoLoihi).

In summary, you should use KerasSpiking if you want to get up and running with minimal
departures from the standard Keras API. If you find yourself wishing for more control
or more features to build your model, or you would like to run your model on different
hardware platforms, consider checking out NengoDL.
