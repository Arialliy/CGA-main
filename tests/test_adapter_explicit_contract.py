import pytest
import torch

from model.output_contract import validate_detector_output


def test_validate_detector_output_does_not_guess_first_4d_tensor():
    bad_output = {
        "some_feature": torch.randn(2, 16, 64, 64),
        "another_feature": torch.randn(2, 32, 32, 32),
    }

    with pytest.raises(KeyError, match="logits"):
        validate_detector_output(bad_output, backbone_name="dummy", require_feature=True)


def test_validate_detector_output_requires_explicit_feature_meta():
    bad_output = {
        "logits": torch.randn(2, 1, 64, 64),
        "features": [torch.randn(2, 16, 64, 64)],
        "adapter_meta": {
            "backbone": "dummy",
            "logits_source": "declared_logits",
            "feature_source": "declared_feature",
        },
    }

    with pytest.raises(KeyError, match="feature_meta"):
        validate_detector_output(bad_output, backbone_name="dummy", require_feature=True)


def test_validate_detector_output_rejects_raw_tuple_output():
    with pytest.raises(TypeError, match="must return a dict"):
        validate_detector_output((torch.randn(2, 1, 64, 64),), backbone_name="dummy", require_feature=True)
