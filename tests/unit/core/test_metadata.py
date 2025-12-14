"""Concise MessageMetadata tests."""

import pytest

from textile.core.metadata import DataclassMetadata, MessageMetadata


class TestGlobalProperties:
    @pytest.mark.parametrize(
        "value,expected", [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0)]
    )
    def test_prominence_clamped_to_one(self, value: float, expected: float) -> None:
        meta = MessageMetadata()
        meta.prominence = value
        assert meta.prominence == expected

    def test_prominence_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="prominence must be >= 0.0"):
            MessageMetadata().prominence = -0.1  # type: ignore[misc]

    def test_prominence_defaults_to_one(self) -> None:
        assert MessageMetadata().prominence == 1.0

    @pytest.mark.parametrize("value", [0, 1, 5, 100])
    def test_turn_index_roundtrip(self, value: int) -> None:
        meta = MessageMetadata()
        meta.turn_index = value
        assert meta.turn_index == value

    def test_turn_index_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="turn_index must be >= 0"):
            MessageMetadata().turn_index = -1  # type: ignore[misc]

    def test_turn_index_defaults_to_zero(self) -> None:
        assert MessageMetadata().turn_index == 0

    @pytest.mark.parametrize("embedding", [[0.1, 0.2], [1.0] * 1536, None])
    def test_embedding_roundtrip(self, embedding: list[float] | None) -> None:
        meta = MessageMetadata()
        meta.embedding = embedding
        assert meta.embedding == embedding


class TestNamespaces:
    def test_set_and_get_namespace(self, sample_metadata: MessageMetadata) -> None:
        from dataclasses import dataclass

        @dataclass
        class TestMeta(DataclassMetadata):
            value: int = 42

        sample_metadata.set_namespace("test", TestMeta(value=100))
        retrieved = sample_metadata.get_namespace("test", TestMeta)
        assert retrieved is not None and retrieved.value == 100

    def test_get_nonexistent_namespace(self, sample_metadata: MessageMetadata) -> None:
        from dataclasses import dataclass

        @dataclass
        class TestMeta(DataclassMetadata):
            value: int = 42

        assert sample_metadata.get_namespace("missing", TestMeta) is None

    def test_has_namespace(self, sample_metadata: MessageMetadata) -> None:
        from dataclasses import dataclass

        @dataclass
        class TestMeta(DataclassMetadata):
            value: int = 42

        assert not sample_metadata.has_namespace("test")
        sample_metadata.set_namespace("test", TestMeta())
        assert sample_metadata.has_namespace("test")


class TestSerialization:
    def test_to_dict_includes_global(self) -> None:
        meta = MessageMetadata()
        meta.prominence = 0.8
        assert meta.to_dict()["global"]["prominence"] == 0.8

    def test_from_dict_restores_global(self) -> None:
        data = {"global": {"prominence": 0.7, "turn_index": 5}, "namespaces": {}}
        meta = MessageMetadata.from_dict(data)
        assert meta.prominence == 0.7 and meta.turn_index == 5

    def test_roundtrip_preserves_data(self) -> None:
        meta = MessageMetadata()
        meta.prominence = 0.9
        meta.turn_index = 3
        restored = MessageMetadata.from_dict(meta.to_dict())
        assert restored.prominence == 0.9 and restored.turn_index == 3
