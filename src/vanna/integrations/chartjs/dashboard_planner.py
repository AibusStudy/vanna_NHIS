"""Deterministic multi-view visualization planning for SQL result data."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .chart_generator import ChartJsChartGenerator


class VisualizationDashboardPlanner:
    """Create one shared dataset and several validated views without an LLM loop."""

    def __init__(self, chart_generator: ChartJsChartGenerator | None = None) -> None:
        self.chart_generator = chart_generator or ChartJsChartGenerator()

    def plan(
        self,
        df: pd.DataFrame,
        *,
        title: str,
        column_metadata: dict[str, dict[str, Any]],
        structured_question: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        structured = structured_question or {}
        warnings: list[str] = []
        columns = [self._column_contract(column, df[column], column_metadata) for column in df.columns]
        rows = [
            {str(column): self._json_value(value) for column, value in row.items()}
            for row in df.to_dict(orient="records")
        ]
        dataset = {"columns": columns, "rows": rows, "rowCount": len(rows)}

        primary = self.chart_generator.generate_chart(df, title, column_metadata)
        views: list[dict[str, Any]] = [
            {"id": "statistics-table", "type": "table", "title": f"{title} 통계표"},
            {
                "id": "primary-chart",
                "type": "chartjs",
                "chartType": primary["chart_type"],
                "title": title,
                "data": primary["data"],
                "options": primary["options"],
            },
        ]
        views.extend(self._alternate_views(primary, title))

        geo = self._geo_contract(columns, structured)
        if geo["valid"]:
            views.append({
                "id": "regional-map",
                "type": "map",
                "title": f"{title} 지역 분포",
                "geo": geo,
                "style": {"palette": ChartJsChartGenerator.COLOR_PALETTE, "mode": "choropleth"},
            })
            if geo.get("latitudeField") and geo.get("longitudeField"):
                views.append({
                    "id": "density-heatmap",
                    "type": "heatmap",
                    "title": f"{title} 밀집도",
                    "geo": geo,
                    "style": {"palette": ChartJsChartGenerator.COLOR_PALETTE, "mode": "density"},
                })
        elif geo["requested"]:
            warnings.extend(geo["errors"])

        warnings.extend(self._metric_warnings(columns, structured))
        filters = self._filters(structured)
        return {
            "schemaVersion": 1,
            "title": title,
            "dataset": dataset,
            "views": views,
            "filters": filters,
            "warnings": list(dict.fromkeys(warnings)),
        }

    def _alternate_views(self, primary: dict[str, Any], title: str) -> list[dict[str, Any]]:
        data = primary["data"]
        datasets = data.get("datasets", [])
        labels = data.get("labels", [])
        if not labels or not datasets:
            return []
        views: list[dict[str, Any]] = []
        if primary["chart_type"] == "bar":
            total = [0.0] * len(labels)
            for dataset in datasets:
                values = dataset.get("data", [])
                if all(isinstance(value, (int, float)) for value in values):
                    total = [current + float(value) for current, value in zip(total, values)]
            # A composition chart is only sound for one series. Summing several
            # periods or measures would silently change the statistical meaning.
            if len(datasets) == 1 and total and all(value >= 0 for value in total) and sum(total) > 0 and len(total) <= 12:
                views.append({
                    "id": "composition-chart",
                    "type": "chartjs",
                    "chartType": "doughnut",
                    "title": f"{title} 구성비",
                    "data": {"labels": labels, "datasets": [{
                        "label": "구성비",
                        "data": total,
                        "backgroundColor": [ChartJsChartGenerator.COLOR_PALETTE[index % 6] for index in range(len(total))],
                        "details": [{"name": str(label), "context": "전체 대비 구성", "description": ""} for label in labels],
                    }]},
                    "options": self.chart_generator._options(f"{title} 구성비", "doughnut", ("", ""), total),
                })
            if any(str(label).removesuffix("년").isdigit() for label in labels):
                line_sets = [{**dataset, "fill": False, "tension": 0.28} for dataset in datasets]
                views.append({
                    "id": "trend-chart", "type": "chartjs", "chartType": "line",
                    "title": f"{title} 추이", "data": {"labels": labels, "datasets": line_sets},
                    "options": self.chart_generator._options(f"{title} 추이", "line", ("연도", "값"), total),
                })
        elif primary["chart_type"] == "doughnut":
            views.append({
                "id": "comparison-chart", "type": "chartjs", "chartType": "bar",
                "title": f"{title} 비교", "data": data,
                "options": self.chart_generator._options(f"{title} 비교", "bar", ("분류", "값"), datasets[0].get("data", [])),
            })
        return views

    @staticmethod
    def _column_contract(column: Any, series: pd.Series, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
        name = str(column)
        item = metadata.get(name, {})
        return {
            "field": name,
            "label": item.get("logical_name") or name.replace("_", " ").title(),
            "description": item.get("description") or "",
            "dataType": str(series.dtype),
            "sqlRole": item.get("sql_role"),
            "semanticType": item.get("semantic_type"),
            "codeName": item.get("code_name"),
            "unit": item.get("unit"),
        }

    @staticmethod
    def _geo_contract(columns: list[dict[str, Any]], structured: dict[str, Any]) -> dict[str, Any]:
        requested_views = structured.get("requested_views", [])
        requested = isinstance(requested_views, list) and any(
            str(view).lower() in {"map", "heatmap", "지도", "밀집도"}
            for view in requested_views
        )
        geo_level = structured.get("geo_level")
        region = next((column for column in columns if str(column.get("semanticType") or "").lower() in {"region", "geography", "administrative_area"} or any(token in column["field"].lower() for token in ("sido", "sigungu", "region", "area_cd"))), None)
        latitude = next((column for column in columns if str(column.get("semanticType") or "").lower() == "latitude" or column["field"].lower() in {"lat", "latitude"}), None)
        longitude = next((column for column in columns if str(column.get("semanticType") or "").lower() == "longitude" or column["field"].lower() in {"lng", "lon", "longitude"}), None)
        requested = requested or bool(geo_level) or region is not None or (latitude is not None and longitude is not None)
        errors: list[str] = []
        if requested and region is None and not (latitude and longitude):
            errors.append("지도 생성에 필요한 표준 지역 코드 또는 위도·경도 컬럼이 없습니다.")
        return {
            "valid": region is not None or bool(latitude and longitude),
            "requested": requested,
            "level": geo_level,
            "regionCodeField": region["field"] if region else None,
            "latitudeField": latitude["field"] if latitude else None,
            "longitudeField": longitude["field"] if longitude else None,
            "errors": errors,
        }

    @staticmethod
    def _metric_warnings(columns: list[dict[str, Any]], structured: dict[str, Any]) -> list[str]:
        requested = str(structured.get("original_question") or "")
        has_per_thousand = "1천" in requested or "천 명당" in requested
        if not has_per_thousand:
            return []
        metric_labels = " ".join(str(column.get("label") or "") for column in columns)
        if "기준 인구" not in metric_labels and "분모" not in metric_labels:
            return ["인구 1천 명당 지표의 기준 인구(분모) 컬럼을 확인할 수 없어 비율을 새로 계산하지 않았습니다."]
        return []

    @staticmethod
    def _filters(structured: dict[str, Any]) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        for key in ("time_conditions", "value_conditions"):
            for item in structured.get(key, []) if isinstance(structured.get(key), list) else []:
                if isinstance(item, dict):
                    filters.append({"type": key, "label": item.get("original_text") or item.get("field_meaning"), "value": item.get("value") or item.get("start_value")})
        return filters

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value
