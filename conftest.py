import pytest
import tensorflow as tf


@pytest.fixture(scope="function", autouse=True)
def clear_session(request):
    # free up resources between tests
    tf.keras.backend.clear_session()
