from importlib import reload

from keras_spiking import version


def test_version_string():
    # reload file so that it will be seen in code coverage
    reload(version)

    version_string = ".".join(str(x) for x in version.version_info)

    if version.dev is not None:
        version_string += ".dev%d" % version.dev

    assert version_string == version.version
