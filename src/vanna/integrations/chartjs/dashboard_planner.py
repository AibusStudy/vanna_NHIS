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

        geo = self._geo_contract(columns, structured)
        plan = structured.get("visualization_plan")
        indicator_contract = plan.get("indicator_contract") if isinstance(plan, dict) else None
        for column in columns:
            if column["field"] == geo.get("regionCodeField"):
                column["label"] = "시도" if geo.get("level") == "시도" else column["label"]
            if column["field"] == geo.get("yearField"):
                column["label"] = "기준연도"
            if column["field"] == geo.get("valueField") and isinstance(indicator_contract, dict):
                column["label"] = str(indicator_contract.get("metric_name") or column["label"])
                geo["valueLabel"] = column["label"]
        chart_df = self._chart_frame(df, geo, structured)
        effective_metadata = dict(column_metadata)
        for column in columns:
            effective_metadata.setdefault(column["field"], {})
            effective_metadata[column["field"]] = {
                **effective_metadata[column["field"]],
                "logical_name": column["label"],
                "description": column["description"],
                "sql_role": column["sqlRole"],
                "semantic_type": column["semanticType"],
                "code_name": column["codeName"],
                "unit": column["unit"],
            }
        primary = self.chart_generator.generate_chart(chart_df, title, effective_metadata)
        requested_views = structured.get("requested_views")
        requested = (
            [str(view).lower() for view in requested_views]
            if isinstance(requested_views, list) and requested_views
            else ["table", primary["chart_type"], "summary"]
        )
        views: list[dict[str, Any]] = []
        for view_type in requested:
            if view_type == "table":
                views.append({"id": "statistics-table", "type": "table", "title": f"{title} 통계표"})
            elif view_type in {"bar", "line", "doughnut"}:
                # 항목이 하나뿐이어도 막대그래프는 동일한 가로 방향을 유지합니다.
                horizontal_bar = view_type == "bar"
                views.append({
                    "id": f"{view_type}-chart",
                    "type": "chartjs",
                    "chartType": view_type,
                    "title": title,
                    "data": primary["data"],
                    "options": {
                        **primary["options"],
                        **({"indexAxis": "y" if horizontal_bar else "x"} if view_type == "bar" else {}),
                    },
                })
            elif view_type == "summary":
                views.append({
                    "id": "summary-metrics",
                    "type": "summary",
                    "title": f"{title} 요약",
                    "data": self._summary_contract(chart_df, columns),
                })
            elif view_type == "map" and geo["valid"]:
                views.append({
                    "id": "regional-map", "type": "map", "title": f"{title} 지역 분포",
                    "geo": geo,
                    "style": {"palette": ChartJsChartGenerator.COLOR_PALETTE, "mode": "choropleth"},
                })
            elif view_type == "heatmap" and geo["valid"] and geo.get("latitudeField") and geo.get("longitudeField"):
                views.append({
                    "id": "density-heatmap", "type": "heatmap", "title": f"{title} 밀집도",
                    "geo": geo,
                    "style": {"palette": ChartJsChartGenerator.COLOR_PALETTE, "mode": "density"},
                })
        if geo["requested"] and not geo["valid"]:
            warnings.extend(geo["errors"])

        warnings.extend(self._metric_warnings(columns, structured))
        filters = self._filters(structured)
        return {
            "schemaVersion": 1,
            "title": title,
            "dataset": dataset,
            "views": views,
            "filters": filters,
            "layout": {
                "type": "split-3" if len(views) == 3 else "grid",
                "areas": [view["id"] for view in views],
            },
            "warnings": list(dict.fromkeys(warnings)),
        }

    @classmethod
    def _summary_contract(
        cls, df: pd.DataFrame, columns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        column_by_field = {column["field"]: column for column in columns}
        measures = [
            field for field in df.select_dtypes(include=["number"]).columns
            if not column_by_field.get(str(field), {}).get("isNumeric") is False
            and str(column_by_field.get(str(field), {}).get("sqlRole") or "").lower()
            not in {"time", "code", "dimension", "identifier"}
        ]
        cards: list[dict[str, Any]] = [{"label": "결과 행 수", "value": len(df), "unit": "건"}]
        if measures:
            field = str(measures[0])
            metadata = column_by_field.get(field, {})
            values = pd.to_numeric(df[field], errors="coerce").dropna()
            if not values.empty:
                if len(values) == 1:
                    cards.insert(0, {"label": metadata.get("label") or field, "value": cls._json_value(values.iloc[0]), "unit": metadata.get("unit")})
                else:
                    cards.extend([
                        {"label": f"최댓값 · {metadata.get('label') or field}", "value": cls._json_value(values.max()), "unit": metadata.get("unit")},
                        {"label": f"최솟값 · {metadata.get('label') or field}", "value": cls._json_value(values.min()), "unit": metadata.get("unit")},
                    ])
        return {"cards": cards[:3]}

    @staticmethod
    def _chart_frame(
        df: pd.DataFrame,
        geo: dict[str, Any],
        structured: dict[str, Any],
    ) -> pd.DataFrame:
        """Remove total rows and non-presentation dimensions from regional charts."""

        frame = df.copy()
        level_field = geo.get("regionLevelField")
        plan = structured.get("visualization_plan")
        include_total = isinstance(plan, dict) and bool(plan.get("include_total"))
        if include_total and level_field in frame.columns:
            level_values = frame[level_field].astype(str).str.replace(r"\.0$", "", regex=True)
            matching = frame[level_values == str(geo.get("detailLevelValue") or "1")]
            if not matching.empty:
                frame = matching

        presentation_fields = [
            geo.get("yearField"), geo.get("regionCodeField"), geo.get("valueField")
        ]
        selected = [field for field in presentation_fields if field in frame.columns]
        return frame[selected] if len(selected) >= 2 else frame

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
        normalized_name = name.lower()
        inferred_geography = any(
            token in normalized_name
            for token in ("si_do", "sido", "sigungu", "si_gun_gu", "region", "area_cd", "emd_cd")
        )
        inferred_year = normalized_name in {"year", "std_year", "std_yyyy", "std_yr"} or normalized_name.endswith("_yr")
        semantic_type = item.get("semantic_type") or (
            "administrative_area" if inferred_geography else "year" if inferred_year else None
        )
        sql_role = item.get("sql_role") or (
            "dimension" if inferred_geography else "time" if inferred_year else None
        )
        return {
            "field": name,
            "label": item.get("logical_name") or name.replace("_", " ").title(),
            "description": item.get("description") or "",
            "dataType": str(series.dtype),
            "sqlRole": sql_role,
            "semanticType": semantic_type,
            "codeName": item.get("code_name"),
            "unit": item.get("unit"),
            "isNumeric": bool(pd.api.types.is_numeric_dtype(series)), # 261023 추가 : 지역 지도 및 위/경도 히트맵 계약
        }

    # 지도 시각화 계약 및 설정
    @staticmethod
    def _geo_contract(columns: list[dict[str, Any]], structured: dict[str, Any]) -> dict[str, Any]:
        requested_views = structured.get("requested_views", [])
        requested = isinstance(requested_views, list) and any(
            str(view).lower() in {"map", "heatmap", "지도", "밀집도"}
            for view in requested_views
        )
        geo_level = structured.get("geo_level")
        region = next((column for column in columns if str(column.get("semanticType") or "").lower() in {"region", "geography", "administrative_area"} or any(token in column["field"].lower() for token in ("si_do", "sido", "sigungu", "si_gun_gu", "region", "area_cd", "emd_cd"))), None)
        latitude = next((column for column in columns if str(column.get("semanticType") or "").lower() == "latitude" or column["field"].lower() in {"lat", "latitude"}), None)
        longitude = next((column for column in columns if str(column.get("semanticType") or "").lower() == "longitude" or column["field"].lower() in {"lng", "lon", "longitude"}), None)
        region_level = next((column for column in columns if column["field"].upper() == "ZN_TYPE_CD"), None)
        year = next((column for column in columns if str(column.get("semanticType") or "").lower() == "year" or column["field"].upper() in {"STD_YYYY", "YEAR"}), None)
        excluded_fields = {
            field
            for field in (
                region["field"] if region else None,
                latitude["field"] if latitude else None,
                longitude["field"] if longitude else None,
                year["field"] if year else None,
            )
            if field
        }
        metric = next(
            (
                column
                for column in columns
                if column.get("isNumeric")
                and column["field"] not in excluded_fields
                and str(column.get("sqlRole") or "").lower() == "measure"
                and str(column.get("semanticType") or "").lower()
                not in {"year", "time", "administrative_area", "region", "geography", "code"}
                and not any(
                    token in column["field"].lower()
                    for token in ("year", "yr", "년도", "연도")
                )
            ),
            None,
        )
        if metric is None:
            metric = next(
                (
                    column
                    for column in columns
                    if column.get("isNumeric")
                    and column["field"] not in excluded_fields
                    and not any(
                        token in column["field"].lower()
                        for token in ("year", "yyyy", "yr", "code", "_cd")
                    )
                ),
                None,
            )
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
            "regionLevelField": region_level["field"] if region_level else None,
            "detailLevelValue": {"시도": "1", "시군구": "2", "읍면동": "3", "중진료권": "4"}.get(str(geo_level), "1"),
            "totalLevelValue": "0",
            "yearField": year["field"] if year else None,
            "valueField": metric["field"] if metric else None,
            "valueLabel": metric["label"] if metric else "값",
            "valueUnit": metric.get("unit") if metric else None,
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
