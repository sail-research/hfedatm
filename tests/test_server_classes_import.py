def test_new_server_classes_import():
    from src.server import (  # noqa: F401
        FedMAStyle,
        FedIIRServer,
        FedRCHFLGaussian,
        FisherMerging,
        HFedATM,
        MTGC,
        MTGCApprox,
        ModelSoup,
        OTFusion,
        RegMeanAll,
    )
    assert FedIIRServer is not None
    assert MTGC is not MTGCApprox
