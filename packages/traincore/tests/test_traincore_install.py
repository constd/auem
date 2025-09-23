"""These tests are trivial and should be removed when more detailed tests are added."""


def test_can_import_base():
    try:
        import traincore  # noqa
    except ImportError:
        raise ImportError("Failed to import traincore")


def test_can_import_datasets():
    pass
