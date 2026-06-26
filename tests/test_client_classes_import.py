def test_new_client_classes_import():
    from src.client import FedIIR, MTGCClient  # noqa: F401

    assert FedIIR is not None
    assert MTGCClient is not None
