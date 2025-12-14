"""Tests for litellm re-exports."""

from unittest.mock import Mock, patch

import pytest

from textile.lite.exports import (
    aimage_generation,
    atranscription,
    batch_completion,
    batch_completion_models,
    batch_completion_models_all_responses,
    get_model_info,
    image_generation,
    moderation,
    supports_function_calling,
    supports_response_schema,
    supports_vision,
    transcription,
)


def test_batch_completion_export():
    """Verify batch_completion is exported."""
    assert callable(batch_completion)


def test_batch_completion_models_export():
    """Verify batch_completion_models is exported."""
    assert callable(batch_completion_models)


def test_batch_completion_models_all_responses_export():
    """Verify batch_completion_models_all_responses is exported."""
    assert callable(batch_completion_models_all_responses)


def test_image_generation_export():
    """Verify image_generation is exported."""
    assert callable(image_generation)


def test_aimage_generation_export():
    """Verify aimage_generation is exported."""
    assert callable(aimage_generation)


def test_transcription_export():
    """Verify transcription is exported."""
    assert callable(transcription)


def test_atranscription_export():
    """Verify atranscription is exported."""
    assert callable(atranscription)


def test_get_model_info_export():
    """Verify get_model_info is exported."""
    assert callable(get_model_info)


def test_supports_function_calling_export():
    """Verify supports_function_calling is exported."""
    assert callable(supports_function_calling)


def test_supports_vision_export():
    """Verify supports_vision is exported."""
    assert callable(supports_vision)


def test_supports_response_schema_export():
    """Verify supports_response_schema is exported."""
    assert callable(supports_response_schema)


@pytest.mark.parametrize("input_type", ["string", "list"])
def test_moderation_default_model(input_type):
    """Moderation provides default model."""
    test_input = "test" if input_type == "string" else ["test1", "test2"]
    with patch("litellm.moderation", return_value=Mock()) as mock_mod:
        moderation(input=test_input)
        mock_mod.assert_called_once()
        assert mock_mod.call_args.kwargs["model"] == "text-moderation-stable"


def test_moderation_custom_model():
    """Moderation allows custom model."""
    with patch("litellm.moderation", return_value=Mock()) as mock_mod:
        moderation(input="test", model="custom-moderator")
        assert mock_mod.call_args.kwargs["model"] == "custom-moderator"


def test_moderation_additional_kwargs():
    """Moderation passes additional kwargs."""
    with patch("litellm.moderation", return_value=Mock()) as mock_mod:
        moderation(input="test", custom_param="value")
        assert mock_mod.call_args.kwargs["custom_param"] == "value"
