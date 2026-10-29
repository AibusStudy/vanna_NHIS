"""Tool for visualizing DataFrame data from CSV files."""

from typing import Any, Optional, Type
import logging
import pandas as pd
from pydantic import BaseModel, Field

from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.components import (
    UiComponent,
    ChartComponent,
    NotificationComponent,
    ComponentType,
    SimpleTextComponent,
)

from .file_system import FileSystem, LocalFileSystem
from vanna.integrations.chartjs import ChartJsChartGenerator, VisualizationDashboardPlanner

logger = logging.getLogger(__name__)


class VisualizeDataArgs(BaseModel):
    """Arguments for visualize_data tool."""

    filename: str = Field(description="Name of the CSV file to visualize")
    title: Optional[str] = Field(
        default=None, description="Optional title for the chart"
    )


class VisualizeDataTool(Tool[VisualizeDataArgs]):
    """Tool that reads CSV files and generates Chart.js visualizations."""

    def __init__(
        self,
        file_system: Optional[FileSystem] = None,
        chart_generator: Optional[ChartJsChartGenerator] = None,
    ):
        """Initialize the tool with a filesystem and Chart.js generator.

        Args:
            file_system: FileSystem implementation for reading CSV files (defaults to LocalFileSystem)
            chart_generator: Chart.js generator used to build an interactive chart
        """
        self.file_system = file_system or LocalFileSystem()
        self.chart_generator = chart_generator or ChartJsChartGenerator()
        self.dashboard_planner = VisualizationDashboardPlanner(self.chart_generator)

    @property
    def name(self) -> str:
        return "visualize_data"

    @property
    def description(self) -> str:
        return "Create a visualization from a CSV file. The tool automatically selects an appropriate chart type based on the data."

    def get_args_schema(self) -> Type[VisualizeDataArgs]:
        return VisualizeDataArgs

    @staticmethod
    def _column_metadata(context: ToolContext) -> dict[str, dict[str, Any]]:
        """Resolve physical columns and generated aliases to Korean business labels."""
        context_metadata = getattr(context, "metadata", {})
        if not isinstance(context_metadata, dict):
            return {}
        main = context_metadata.get("main_workflow", {})
        main = main if isinstance(main, dict) else {}
        metadata = main.get("metadata", {})
        candidates = metadata.get("candidates", []) if isinstance(metadata, dict) else []
        labels: dict[str, dict[str, Any]] = {}
        for candidate in candidates if isinstance(candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            column = candidate.get("column_name")
            if not isinstance(column, str) or not column:
                continue
            logical_name = candidate.get("column_logical_name")
            description = candidate.get("column_desc")
            labels[column] = {
                "logical_name": logical_name if isinstance(logical_name, str) else column,
                "description": description if isinstance(description, str) else "",
                "sql_role": candidate.get("sql_role"),
                "semantic_type": candidate.get("semantic_type"),
                "code_name": candidate.get("code_name"),
                "unit": candidate.get("unit") or candidate.get("column_unit"),
            }

        structured_question = main.get("structured_question", {})
        structured_question = structured_question if isinstance(structured_question, dict) else {}
        for dependency in structured_question.get("semantic_dependency_metadata", []):
            if not isinstance(dependency, dict):
                continue
            column = dependency.get("column_name")
            if isinstance(column, str) and column:
                labels[column] = {
                    "logical_name": dependency.get("column_logical_name") or column,
                    "description": dependency.get("column_desc") or "",
                    "sql_role": dependency.get("sql_role"),
                    "semantic_type": dependency.get("semantic_type"),
                    "code_name": dependency.get("code_name"),
                    "unit": dependency.get("unit") or dependency.get("column_unit"),
                }

        structured = main.get("structured_question", {})
        structured = structured if isinstance(structured, dict) else {}
        for item in structured.get("aggregation", []):
            if not isinstance(item, dict):
                continue
            alias = item.get("alias")
            meaning = item.get("field_meaning")
            if isinstance(alias, str) and alias:
                labels[alias] = {
                    "logical_name": meaning if isinstance(meaning, str) else alias,
                    "description": item.get("original_text") if isinstance(item.get("original_text"), str) else "",
                    "unit": item.get("unit"),
                }
        for item in structured.get("output_fields", []):
            if not isinstance(item, dict):
                continue
            alias = item.get("alias") or item.get("field_name")
            meaning = item.get("field_meaning") or item.get("logical_name")
            if isinstance(alias, str) and alias:
                labels[alias] = {
                    "logical_name": meaning if isinstance(meaning, str) else alias,
                    "description": item.get("original_text") if isinstance(item.get("original_text"), str) else "",
                    "unit": item.get("unit"),
                }
        return labels

    @staticmethod
    def _chart_title(context: ToolContext, fallback: Optional[str]) -> str:
        """Build a concise result title from the structured semantic contract."""
        metadata = getattr(context, "metadata", {})
        main = metadata.get("main_workflow", {}) if isinstance(metadata, dict) else {}
        structured = main.get("structured_question", {}) if isinstance(main, dict) else {}
        if not isinstance(structured, dict):
            return fallback or "데이터 시각화"

        time_conditions = structured.get("time_conditions", [])
        period = ""
        if isinstance(time_conditions, list) and time_conditions:
            condition = time_conditions[0] if isinstance(time_conditions[0], dict) else {}
            start = str(condition.get("start_value") or "")[:4]
            end = str(condition.get("end_value") or "")[:4]
            if start and end:
                period = f"{start}년부터 {end}년까지"
            elif isinstance(condition.get("original_text"), str):
                period = condition["original_text"]

        groups = []
        for item in structured.get("group_by", []):
            if not isinstance(item, dict):
                continue
            label = item.get("original_text") or item.get("field_meaning")
            if isinstance(label, str) and label:
                groups.append(label.rstrip("별") + "별")

        target = ""
        target_entity = structured.get("target_entity", {})
        if isinstance(target_entity, dict):
            target = target_entity.get("entity_meaning") or target_entity.get("original_text") or ""
        if not target:
            aggregations = structured.get("aggregation", [])
            if isinstance(aggregations, list) and aggregations and isinstance(aggregations[0], dict):
                target = aggregations[0].get("field_meaning") or ""
        target = str(target).replace("받은 사람", "수검자").strip()

        parts = [part for part in (period, "·".join(dict.fromkeys(groups)), target) if part]
        if parts:
            return " ".join(parts) + (" 추이" if period else " 현황")
        return fallback or "데이터 시각화"

    async def execute(
        self, context: ToolContext, args: VisualizeDataArgs
    ) -> ToolResult:
        """Read CSV file and generate visualization."""
        try:
            logger.info(f"Starting visualization for file: {args.filename}")

            # Read the CSV file using FileSystem
            csv_content = await self.file_system.read_file(args.filename, context)
            logger.info(f"Read {len(csv_content)} bytes from CSV file")

            # Parse CSV into DataFrame
            import io

            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(
                f"Parsed DataFrame with shape {df.shape}, columns: {df.columns.tolist()}, dtypes: {df.dtypes.to_dict()}"
            )

            # Generate title
            title = self._chart_title(context, args.title)
            # Build an interactive Chart.js payload from the persisted SQL-result CSV.
            logger.info("Generating chart...")
            column_metadata = self._column_metadata(context)
            main_metadata = getattr(context, "metadata", {})
            main_workflow = main_metadata.get("main_workflow", {}) if isinstance(main_metadata, dict) else {}
            structured_question = main_workflow.get("structured_question", {}) if isinstance(main_workflow, dict) else {}
            dashboard = self.dashboard_planner.plan(
                df, title=title, column_metadata=column_metadata,
                structured_question=structured_question if isinstance(structured_question, dict) else {},
            )
            logger.info(
                "Visualization dashboard generated with %s views", len(dashboard["views"])
            )

            # Create result message
            row_count = len(df)
            col_count = len(df.columns)
            result = f"Created visualization from '{args.filename}' ({row_count} rows, {col_count} columns)."

            chart_component = ChartComponent(
                chart_type="visualization_dashboard",
                data=dashboard,
                title=title,
                config={
                    "source_file": args.filename,
                },
            )
            logger.info("ChartComponent created successfully")

            logger.info("Creating ToolResult...")
            tool_result = ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=UiComponent(
                    rich_component=chart_component,
                    simple_component=SimpleTextComponent(text=result),
                ),
                metadata={
                    "filename": args.filename,
                    "rows": row_count,
                    "columns": col_count,
                    "chart_type": "visualization_dashboard",
                    "view_count": len(dashboard["views"]),
                    "visualization_format": "dashboard_json",
                    "source_format": "csv",
                },
            )
            logger.info("ToolResult created successfully")
            return tool_result

        except FileNotFoundError as e:
            logger.error(f"File not found: {args.filename}", exc_info=True)
            error_message = f"File not found: {args.filename}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "file_not_found"},
            )
        except pd.errors.ParserError as e:
            logger.error(f"CSV parse error for {args.filename}", exc_info=True)
            error_message = f"Failed to parse CSV file '{args.filename}': {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "csv_parse_error"},
            )
        except ValueError as e:
            logger.error(f"Visualization error for {args.filename}", exc_info=True)
            error_message = f"Cannot visualize data: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "visualization_error"},
            )
        except Exception as e:
            logger.error(
                f"Unexpected error creating visualization for {args.filename}",
                exc_info=True,
            )
            error_message = f"Error creating visualization: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "general_error"},
            )
