"""
Utilities to ease cross-compatibility between different versions of dependencies.
"""

import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) < version.parse("2.3.0"):

    def node_layer(node):
        """Return the layer associated with this node."""
        return node.outbound_layer


else:

    def node_layer(node):
        """Return the layer associated with this node."""
        return node.layer
