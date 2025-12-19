"""Integration tests for multi-transformer pipelines."""

from unittest.mock import Mock, patch

from textile import completion
from textile.transformers.base import ContextTransformer


class MockDecayTransformer(ContextTransformer):
    """Mock transformer for testing integration workflows."""

    def __init__(self, half_life_turns: int = 5, threshold: float = 0.1):
        """Initialize mock decay transformer.

        Args:
            half_life_turns: Turns until prominence decays by half
            threshold: Prominence threshold for pruning
        """
        self.half_life_turns = half_life_turns
        self.threshold = threshold

    def transform(self, context, state):
        """Return context unchanged for integration tests."""
        return context, state


def test_pipeline_with_multiple_transformers(conversation_messages, mock_litellm_response):
    """Test messages → pipeline (2 transformers) → LLM."""
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                transformers = [
                    MockDecayTransformer(half_life_turns=3, threshold=0.2),
                    MockDecayTransformer(half_life_turns=5, threshold=0.1),
                ]
                response = completion(
                    model="gpt-4", messages=conversation_messages, transformers=transformers
                )
                assert response is not None
                mock_llm.assert_called_once()


def test_pipeline_sequential_application(conversation_messages, mock_litellm_response):
    """Test transformers apply sequentially with state threading."""
    mock_transformer_1 = Mock()
    mock_transformer_2 = Mock()
    mock_context_1 = Mock()
    mock_context_1.render.return_value = conversation_messages
    mock_state_1 = Mock(tools=None)
    mock_context_2 = Mock()
    mock_context_2.render.return_value = conversation_messages
    mock_state_2 = Mock(tools=None)
    mock_transformer_1.should_apply.return_value = True
    mock_transformer_1.transform.return_value = (mock_context_1, mock_state_1)
    mock_transformer_1.on_response.return_value = []
    mock_transformer_2.should_apply.return_value = True
    mock_transformer_2.transform.return_value = (mock_context_2, mock_state_2)
    mock_transformer_2.on_response.return_value = []
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4",
                    messages=conversation_messages,
                    transformers=[mock_transformer_1, mock_transformer_2],
                )
                assert response is not None
                mock_transformer_1.transform.assert_called_once()
                mock_transformer_2.transform.assert_called_once()


def test_pipeline_conditional_application(conversation_messages, mock_litellm_response):
    """Test transformers respect should_apply conditions."""
    mock_transformer = Mock()
    mock_transformer.should_apply.return_value = False
    mock_transformer.on_response.return_value = []
    with patch("litellm.completion", return_value=mock_litellm_response):
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = None
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(
                    model="gpt-4", messages=conversation_messages, transformers=[mock_transformer]
                )
                assert response is not None
                mock_transformer.transform.assert_not_called()


def test_pipeline_with_config_transformers(conversation_messages, mock_litellm_response):
    """Test global transformers from config."""
    config_transformer = MockDecayTransformer(half_life_turns=5)
    with patch("litellm.completion", return_value=mock_litellm_response) as mock_llm:
        with patch("textile.lite.completion.get_config") as mock_config:
            mock_config.return_value.transformers = [config_transformer]
            with patch("litellm.get_max_tokens", return_value=4096):
                response = completion(model="gpt-4", messages=conversation_messages)
                assert response is not None
                mock_llm.assert_called_once()
