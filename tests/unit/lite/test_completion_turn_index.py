"""Test turn index assignment in completion flow."""

from unittest.mock import patch

import pytest

from textile.lite.completion import completion


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


def test_completion_without_transformers_no_regression(mock_completion_response):
    """Verify completion without transformers still works (no turn indices needed)."""
    messages = [{"role": "user", "content": "Simple question"}]

    with patch(
        "textile.lite.completion.litellm.completion", return_value=mock_completion_response
    ) as mock_llm:
        result = completion(model="gpt-3.5-turbo", messages=messages)

    assert mock_llm.called
    assert result.choices[0].message.content == "Test response"
