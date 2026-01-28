"""Tests for ToolSpec and ToolCollection - core data structures."""

from __future__ import annotations

import pytest

from atr import ToolCollection, ToolSpec


class TestToolSpec:
    """Tests for ToolSpec dataclass."""

    def test_toolspec_minimal_creation(self):
        """ToolSpec can be created with just name and description."""
        tool = ToolSpec(name="test_tool", description="A test tool")

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_toolspec_full_creation(self):
        """ToolSpec accepts all optional fields."""
        tool = ToolSpec(
            name="advanced_tool",
            description="A tool with all fields",
            parameters={"type": "object", "properties": {"arg1": {"type": "string"}}},
            source="mcp:filesystem",
            metadata={"custom": "data", "version": 1},
        )

        assert tool.name == "advanced_tool"
        assert tool.description == "A tool with all fields"
        assert tool.parameters == {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
        }
        assert tool.source == "mcp:filesystem"
        assert tool.metadata == {"custom": "data", "version": 1}

    def test_toolspec_parameters_default_none(self):
        """Parameters defaults to None when not provided."""
        tool = ToolSpec(name="simple", description="Simple tool")

        assert tool.parameters is None

    def test_toolspec_source_default_none(self):
        """Source defaults to None when not provided."""
        tool = ToolSpec(name="simple", description="Simple tool")

        assert tool.source is None

    def test_toolspec_metadata_default_empty_dict(self):
        """Metadata defaults to empty dict when not provided."""
        tool = ToolSpec(name="simple", description="Simple tool")

        assert tool.metadata == {}

    def test_toolspec_equality_by_name(self):
        """Two ToolSpecs with same name are equal (regardless of other fields)."""
        tool1 = ToolSpec(name="test", description="First description", source="src1")
        tool2 = ToolSpec(name="test", description="Second description", source="src2")

        assert tool1 == tool2

    def test_toolspec_inequality(self):
        """ToolSpecs with different names are not equal."""
        tool1 = ToolSpec(name="test1", description="desc")
        tool2 = ToolSpec(name="test2", description="desc")

        assert tool1 != tool2

    def test_toolspec_hashable(self):
        """ToolSpec can be used in sets and as dict keys."""
        tool1 = ToolSpec(name="test", description="desc1")
        tool2 = ToolSpec(name="test", description="desc2")
        tool3 = ToolSpec(name="other", description="desc")

        # Same name = same hash
        tool_set = {tool1, tool2, tool3}
        assert len(tool_set) == 2

        # Can use as dict key
        tool_dict = {tool1: "value1"}
        assert tool_dict[tool2] == "value1"

    def test_toolspec_parameters_json_schema(self):
        """Parameters should accept valid JSON Schema dict."""
        json_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "enum": ["utf-8", "ascii"]},
            },
            "required": ["path"],
        }

        tool = ToolSpec(name="read", description="Read file", parameters=json_schema)

        assert tool.parameters == json_schema
        assert tool.parameters["type"] == "object"
        assert "path" in tool.parameters["properties"]

    def test_toolspec_to_summary(self):
        """to_summary returns formatted string."""
        tool = ToolSpec(name="test_tool", description="A test tool description")

        summary = tool.to_summary()

        assert "test_tool" in summary
        assert "A test tool description" in summary
        assert summary.startswith("- ")

    def test_toolspec_to_summary_truncates_long_description(self):
        """Long descriptions are truncated in summaries."""
        long_desc = "A" * 200
        tool = ToolSpec(name="test", description=long_desc)

        summary = tool.to_summary(max_description_length=50)

        assert len(summary) < 200
        assert "..." in summary


class TestToolCollection:
    """Tests for ToolCollection container."""

    def test_toolcollection_from_specs(self, sample_toolspecs: list[ToolSpec]):
        """ToolCollection can be created from list of ToolSpecs."""
        collection = ToolCollection(tools=sample_toolspecs)

        assert len(collection) == len(sample_toolspecs)

    def test_toolcollection_iteration(self, sample_toolspecs: list[ToolSpec]):
        """Can iterate over tools in collection."""
        collection = ToolCollection(tools=sample_toolspecs)

        tools_list = list(collection)
        assert len(tools_list) == len(sample_toolspecs)
        assert all(isinstance(t, ToolSpec) for t in tools_list)

    def test_toolcollection_len(self, sample_toolspecs: list[ToolSpec]):
        """len() returns count of tools."""
        collection = ToolCollection(tools=sample_toolspecs)

        assert len(collection) == 8  # Based on sample_toolspecs fixture

    def test_toolcollection_empty(self):
        """Empty collection works correctly."""
        collection = ToolCollection(tools=[])

        assert len(collection) == 0
        assert list(collection) == []

    def test_toolcollection_empty_default(self):
        """Collection defaults to empty."""
        collection = ToolCollection()

        assert len(collection) == 0

    def test_toolcollection_names(self, sample_toolspecs: list[ToolSpec]):
        """names property returns set of tool names."""
        collection = ToolCollection(tools=sample_toolspecs)

        names = collection.names
        assert isinstance(names, set)
        assert "read_file" in names
        assert "send_email" in names
        assert len(names) == 8

    def test_toolcollection_contains_by_name(self, sample_toolspecs: list[ToolSpec]):
        """Can check if tool name exists in collection."""
        collection = ToolCollection(tools=sample_toolspecs)

        assert "read_file" in collection
        assert "nonexistent_tool" not in collection

    def test_toolcollection_contains_by_toolspec(self, sample_toolspecs: list[ToolSpec]):
        """Can check if ToolSpec exists in collection."""
        collection = ToolCollection(tools=sample_toolspecs)
        existing_tool = sample_toolspecs[0]
        new_tool = ToolSpec(name="nonexistent", description="Not in collection")

        assert existing_tool in collection
        assert new_tool not in collection

    def test_toolcollection_getitem_by_index(self, sample_toolspecs: list[ToolSpec]):
        """Can retrieve tool by index."""
        collection = ToolCollection(tools=sample_toolspecs)

        tool = collection[0]
        assert isinstance(tool, ToolSpec)
        assert tool.name == sample_toolspecs[0].name

    def test_toolcollection_getitem_by_name(self, sample_toolspecs: list[ToolSpec]):
        """Can retrieve tool by name."""
        collection = ToolCollection(tools=sample_toolspecs)

        tool = collection["read_file"]
        assert tool.name == "read_file"

    def test_toolcollection_getitem_nonexistent_raises(
        self, sample_toolspecs: list[ToolSpec]
    ):
        """Getting nonexistent tool raises KeyError."""
        collection = ToolCollection(tools=sample_toolspecs)

        with pytest.raises(KeyError):
            _ = collection["nonexistent"]

    def test_toolcollection_filter_by_names(self, sample_toolspecs: list[ToolSpec]):
        """filter_by_names returns new collection with matching tools."""
        collection = ToolCollection(tools=sample_toolspecs)

        filtered = collection.filter_by_names(["read_file", "write_file"])

        assert len(filtered) == 2
        assert "read_file" in filtered.names
        assert "write_file" in filtered.names
        assert "send_email" not in filtered.names

    def test_toolcollection_filter_by_names_list(self, sample_toolspecs: list[ToolSpec]):
        """filter_by_names accepts list of names."""
        collection = ToolCollection(tools=sample_toolspecs)

        filtered = collection.filter_by_names(["read_file"])

        assert len(filtered) == 1

    def test_toolcollection_filter_by_names_empty(self, sample_toolspecs: list[ToolSpec]):
        """filter_by_names with no matches returns empty collection."""
        collection = ToolCollection(tools=sample_toolspecs)

        filtered = collection.filter_by_names(["nonexistent"])

        assert len(filtered) == 0

    def test_toolcollection_add(self):
        """add() adds a single tool to collection."""
        collection = ToolCollection()
        tool = ToolSpec(name="new_tool", description="New tool")

        collection.add(tool)

        assert len(collection) == 1
        assert "new_tool" in collection.names

    def test_toolcollection_extend(self, sample_toolspecs: list[ToolSpec]):
        """extend() adds multiple tools to collection."""
        collection = ToolCollection()

        collection.extend(sample_toolspecs[:3])

        assert len(collection) == 3

    def test_toolcollection_to_summaries(self, sample_toolspecs: list[ToolSpec]):
        """to_summaries returns formatted string of all tools."""
        collection = ToolCollection(tools=sample_toolspecs[:2])

        summaries = collection.to_summaries()

        assert "read_file" in summaries
        assert "write_file" in summaries
        assert "\n" in summaries  # Multiple lines
