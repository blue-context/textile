"""Test turn index assignment in completion flow."""

from unittest.mock import patch

import pytest

from textile.lite.completion import completion
from textile.transformers.decay import DecayTransformer


@pytest.mark.parametrize("num_msgs,expected_state_turn", [(1, 0), (3, 2), (7, 6)])
def test_assigns_sequential_turn_indices(
    num_msgs, expected_state_turn, capture_transformer, mock_completion_response
):
    """Verify messages get sequential turn indices and state tracks last turn."""
    msgs = [{"role": "user", "content": f"Msg {i}"} for i in range(num_msgs)]
    transformer, captured = capture_transformer

    with patch("textile.lite.completion.litellm.completion", return_value=mock_completion_response):
        completion(model="gpt-3.5-turbo", messages=msgs, transformers=[transformer])

    actual_indices = [m.turn_index for m in captured["context"].messages]
    assert actual_indices == list(range(num_msgs))
    assert captured["state"].turn_index == expected_state_turn


def test_decay_transformer_filters_old_messages(mock_completion_response):
    """Verify DecayTransformer filters old messages using turn indices."""
    messages = [
        {"role": "user", "content": "Old msg 1"},
        {"role": "assistant", "content": "Old resp 1"},
        {"role": "user", "content": "Old msg 2"},
        {"role": "assistant", "content": "Old resp 2"},
        {"role": "user", "content": "Recent msg"},
        {"role": "assistant", "content": "Recent resp"},
        {"role": "user", "content": "Latest msg"},
    ]
    captured_llm_messages = None

    def capture_llm_call(*args, **kwargs):
        nonlocal captured_llm_messages
        captured_llm_messages = kwargs.get("messages", [])
        return mock_completion_response

    with patch("textile.lite.completion.litellm.completion", side_effect=capture_llm_call):
        completion(
            model="gpt-3.5-turbo",
            messages=messages,
            transformers=[DecayTransformer(half_life_turns=2, threshold=0.2)],
        )
    message_contents = [msg["content"] for msg in captured_llm_messages]
    assert len(captured_llm_messages) < len(messages)
    assert "Latest msg" in message_contents
    assert "Old msg 1" not in message_contents


def test_decay_filters_context_shift_conversation(mock_completion_response):
    """Verify DecayTransformer prioritizes recent context over old context."""
    messages = [
        {"role": "user", "content": "Tell me about electric cars"},
        {"role": "assistant", "content": "Electric cars use batteries..."},
        {"role": "user", "content": "What about Tesla?"},
        {"role": "assistant", "content": "Tesla is a leading EV manufacturer..."},
        {"role": "user", "content": "Let's talk about Python programming"},
        {"role": "assistant", "content": "Python is a high-level language..."},
        {"role": "user", "content": "What are Python decorators?"},
        {"role": "assistant", "content": "Decorators modify functions..."},
        {"role": "user", "content": "What were we just discussing?"},
    ]
    captured_llm_messages = None

    def capture_llm_call(*args, **kwargs):
        nonlocal captured_llm_messages
        captured_llm_messages = kwargs.get("messages", [])
        return mock_completion_response

    with patch("textile.lite.completion.litellm.completion", side_effect=capture_llm_call):
        completion(
            model="gpt-3.5-turbo",
            messages=messages,
            transformers=[DecayTransformer(half_life_turns=2, threshold=0.2)],
        )
    message_contents = [msg["content"] for msg in captured_llm_messages]
    content_text = " ".join(message_contents)
    assert "Tell me about electric cars" not in message_contents
    assert "Tesla" not in content_text
    assert "Python" in content_text


def test_completion_without_transformers_no_regression(mock_completion_response):
    """Verify completion without transformers still works (no turn indices needed)."""
    messages = [{"role": "user", "content": "Simple question"}]

    with patch("textile.lite.completion.litellm.completion", return_value=mock_completion_response) as mock_llm:
        result = completion(model="gpt-3.5-turbo", messages=messages)

    assert mock_llm.called
    assert result.choices[0].message.content == "Test response"
