"""Tests for tools exposed at each main SQL workflow stage."""

from vanna.core.agent.agent import Agent
from vanna.core.tool import ToolSchema


def _schema(name: str) -> ToolSchema:
    return ToolSchema(name=name, description=name, parameters={})


def _filtered_names(stage: str) -> set[str]:
    schemas = [
        _schema("search_business_metadata"),
        _schema("search_saved_correct_tool_uses"),
        _schema("csv_to_json"),
        _schema("run_sql"),
        _schema("save_question_tool_args"),
        _schema("visualize_data"),
        _schema("unrelated_tool"),
    ]
    return {
        schema.name
        for schema in Agent._tool_schemas_for_workflow_stage(schemas, stage)
    }


def test_visualize_data_is_available_during_sql_generation() -> None:
    assert "visualize_data" in _filtered_names("sql_generation")


def test_visualize_data_is_available_after_successful_query() -> None:
    names = _filtered_names("successful_query_save")

    assert names == {"save_question_tool_args", "visualize_data"}


def test_visualize_data_is_not_available_during_final_answer() -> None:
    assert "visualize_data" not in _filtered_names("final_answer")
