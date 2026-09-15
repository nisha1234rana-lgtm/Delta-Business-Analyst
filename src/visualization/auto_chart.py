from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =========================================================
# OUTPUT DIRECTORY
# =========================================================

CHART_DIR = (
    PROJECT_ROOT
    / "assets"
    / "generated_charts"
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# HELPERS
# =========================================================

def _safe_dataframe(
    sql_output: dict | None,
) -> pd.DataFrame:
    """
    Extract the SQL result dataframe from the analyst output.
    """

    if not sql_output:
        return pd.DataFrame()

    result = (
        sql_output.get(
            "result",
            {}
        )
        or {}
    )

    data = (
        result.get(
            "data"
        )
    )

    if data is None:
        return pd.DataFrame()

    if isinstance(
        data,
        pd.DataFrame,
    ):
        return data.copy()

    if isinstance(
        data,
        list,
    ):
        return pd.DataFrame(
            data
        )

    if isinstance(
        data,
        dict,
    ):
        return pd.DataFrame(
            [data]
        )

    return pd.DataFrame()


def _humanize(
    column: str,
) -> str:
    """
    Convert machine column names into readable labels.
    """

    label = (
        column
        .replace(
            "_pct",
            " (%)",
        )
        .replace(
            "_pp",
            " (pp)",
        )
        .replace(
            "_min",
            " (min)",
        )
        .replace(
            "_m",
            " ($M)",
        )
        .replace(
            "_",
            " ",
        )
        .strip()
    )

    return (
        label[:1].upper()
        + label[1:]
        if label
        else column
    )


def _slugify(
    text: str,
) -> str:
    """
    Build a safe filename.
    """

    text = (
        text.lower()
        .strip()
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = (
        text.strip(
            "_"
        )
    )

    return (
        text[:90]
        or "chart"
    )


def _is_numeric_series(
    series: pd.Series,
) -> bool:
    """
    Check whether a pandas series is numeric.
    """

    return pd.api.types.is_numeric_dtype(
        series
    )


def _numeric_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return numeric columns.
    """

    return [
        column
        for column in df.columns
        if _is_numeric_series(
            df[column]
        )
    ]


def _categorical_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return non-numeric columns.
    """

    return [
        column
        for column in df.columns
        if not _is_numeric_series(
            df[column]
        )
    ]


def _preferred_time_column(
    df: pd.DataFrame,
) -> str | None:
    """
    Find a likely time axis.
    """

    preferred = [
        "date",
        "flight_date",
        "report_date",
        "year_month",
        "year",
        "quarter",
        "month",
    ]

    columns_lower = {
        column.lower():
            column
        for column in df.columns
    }

    for candidate in preferred:

        if candidate in columns_lower:

            return (
                columns_lower[
                    candidate
                ]
            )

    return None


def _preferred_category_column(
    df: pd.DataFrame,
) -> str | None:
    """
    Find a useful category column for rankings/comparisons.
    """

    preferred = [
        "airport",
        "airport_code",
        "origin",
        "destination",
        "route",
        "operating_airline",
        "operator",
        "carrier",
        "city",
        "year",
        "quarter",
    ]

    columns_lower = {
        column.lower():
            column
        for column in df.columns
    }

    for candidate in preferred:

        if candidate in columns_lower:

            return (
                columns_lower[
                    candidate
                ]
            )

    categorical = (
        _categorical_columns(
            df
        )
    )

    if categorical:
        return categorical[0]

    numeric = (
        _numeric_columns(
            df
        )
    )

    if len(numeric) >= 2:

        first = (
            numeric[0]
        )

        unique_count = (
            df[first]
            .nunique(
                dropna=True
            )
        )

        if unique_count <= 30:
            return first

    return None


def _preferred_metric_column(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
) -> str | None:
    """
    Pick the most likely business metric column.
    """

    exclude = (
        exclude
        or []
    )

    numeric = [
        column
        for column in _numeric_columns(
            df
        )
        if column not in exclude
    ]

    if not numeric:
        return None

    priority_tokens = [
        "rate",
        "pct",
        "margin",
        "change",
        "delay",
        "revenue",
        "income",
        "cost",
        "flights",
        "count",
        "debt",
        "cash",
    ]

    for token in priority_tokens:

        for column in numeric:

            if token in column.lower():
                return column

    return numeric[0]


def _metric_format(
    column: str,
    value: Any,
) -> str:
    """
    Format KPI value based on metric name.
    """

    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    lower = (
        column.lower()
    )

    if (
        lower.endswith(
            "_pct"
        )
        or "rate_pct" in lower
        or "margin_pct" in lower
    ):
        return (
            f"{number:,.2f}%"
        )

    if (
        lower.endswith(
            "_pp"
        )
        or "change_pp" in lower
    ):
        return (
            f"{number:+,.2f} pp"
        )

    if (
        lower.endswith(
            "_m"
        )
        or "revenue_m" in lower
        or "income_m" in lower
        or "cost_m" in lower
        or "debt_m" in lower
        or "cash_m" in lower
    ):
        return (
            f"${number:,.0f}M"
        )

    if (
        "flight" in lower
        or "count" in lower
    ):
        return (
            f"{number:,.0f}"
        )

    if (
        "delay" in lower
        and "min" in lower
    ):
        return (
            f"{number:,.2f} min"
        )

    return (
        f"{number:,.2f}"
    )


# =========================================================
# CHART BUILDERS
# =========================================================

def _build_single_metric(
    df: pd.DataFrame,
    question: str,
) -> dict:
    """
    Build a KPI result instead of forcing a chart
    for a single scalar.
    """

    column = (
        df.columns[
            0
        ]
    )

    value = (
        df.iloc[
            0,
            0,
        ]
    )

    return {
        "kind":
            "metric",

        "title":
            _humanize(
                column
            ),

        "metric_label":
            _humanize(
                column
            ),

        "metric_value":
            _metric_format(
                column,
                value,
            ),

        "figure":
            None,

        "data":
            df,

        "question":
            question,
    }


def _build_time_series(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    question: str,
) -> dict:
    """
    Build a line chart for time-based data.
    """

    figure = px.line(
        df,
        x=x_column,
        y=y_column,
        markers=True,
        title=(
            f"{_humanize(y_column)} "
            f"by {_humanize(x_column)}"
        ),
        labels={
            x_column:
                _humanize(
                    x_column
                ),

            y_column:
                _humanize(
                    y_column
                ),
        },
    )

    figure.update_layout(
        template="plotly_white",
        hovermode="x unified",
    )

    return {
        "kind":
            "chart",

        "chart_type":
            "line",

        "title":
            (
                f"{_humanize(y_column)} "
                f"by {_humanize(x_column)}"
            ),

        "figure":
            figure,

        "data":
            df,

        "question":
            question,
    }


def _build_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    metric_column: str,
    question: str,
) -> dict:
    """
    Build a ranking/comparison bar chart.
    """

    plot_df = (
        df.copy()
    )

    if (
        len(
            plot_df
        )
        > 20
    ):

        plot_df = (
            plot_df
            .head(
                20
            )
        )

    horizontal = (
        len(
            plot_df
        )
        > 6
        or plot_df[
            category_column
        ]
        .astype(str)
        .str.len()
        .mean()
        > 8
    )

    if horizontal:

        figure = px.bar(
            plot_df,
            x=metric_column,
            y=category_column,
            orientation="h",
            title=(
                f"{_humanize(metric_column)} "
                f"by {_humanize(category_column)}"
            ),
            labels={
                category_column:
                    _humanize(
                        category_column
                    ),

                metric_column:
                    _humanize(
                        metric_column
                    ),
            },
        )

        figure.update_layout(
            yaxis={
                "categoryorder":
                    "total ascending"
            },
        )

    else:

        figure = px.bar(
            plot_df,
            x=category_column,
            y=metric_column,
            title=(
                f"{_humanize(metric_column)} "
                f"by {_humanize(category_column)}"
            ),
            labels={
                category_column:
                    _humanize(
                        category_column
                    ),

                metric_column:
                    _humanize(
                        metric_column
                    ),
            },
        )

    figure.update_layout(
        template="plotly_white",
    )

    return {
        "kind":
            "chart",

        "chart_type":
            "bar",

        "title":
            (
                f"{_humanize(metric_column)} "
                f"by {_humanize(category_column)}"
            ),

        "figure":
            figure,

        "data":
            df,

        "question":
            question,
    }


def _build_multi_metric_comparison(
    df: pd.DataFrame,
    category_column: str,
    metric_columns: list[str],
    question: str,
) -> dict:
    """
    Build grouped bars when several metrics are returned.
    """

    selected = (
        metric_columns[
            :4
        ]
    )

    melted = (
        df[
            [
                category_column,
                *selected,
            ]
        ]
        .melt(
            id_vars=[
                category_column
            ],
            var_name="metric",
            value_name="value",
        )
    )

    melted[
        "metric"
    ] = (
        melted[
            "metric"
        ]
        .map(
            _humanize
        )
    )

    figure = px.bar(
        melted,
        x=category_column,
        y="value",
        color="metric",
        barmode="group",
        title=(
            f"Business Metric Comparison "
            f"by {_humanize(category_column)}"
        ),
        labels={
            category_column:
                _humanize(
                    category_column
                ),

            "value":
                "Value",

            "metric":
                "Metric",
        },
    )

    figure.update_layout(
        template="plotly_white",
    )

    return {
        "kind":
            "chart",

        "chart_type":
            "grouped_bar",

        "title":
            (
                f"Business Metric Comparison "
                f"by {_humanize(category_column)}"
            ),

        "figure":
            figure,

        "data":
            df,

        "question":
            question,
    }


# =========================================================
# MAIN VISUALIZATION ROUTER
# =========================================================

def build_visualization(
    question: str,
    sql_output: dict | None,
) -> dict:
    """
    Automatically choose a useful visualization based on
    the SQL result shape.

    Returns one of:
    - metric
    - chart
    - table
    - none

    This function does not alter any backend analytics.
    """

    df = (
        _safe_dataframe(
            sql_output
        )
    )

    if df.empty:

        return {
            "kind":
                "none",

            "title":
                "No visualization available",

            "figure":
                None,

            "data":
                df,

            "question":
                question,
        }

    # -----------------------------------------------------
    # SINGLE KPI
    # -----------------------------------------------------

    if (
        df.shape[
            0
        ] == 1
        and df.shape[
            1
        ] == 1
    ):

        return (
            _build_single_metric(
                df,
                question,
            )
        )

    # -----------------------------------------------------
    # IDENTIFY COLUMNS
    # -----------------------------------------------------

    time_column = (
        _preferred_time_column(
            df
        )
    )

    category_column = (
        _preferred_category_column(
            df
        )
    )

    numeric_columns = (
        _numeric_columns(
            df
        )
    )

    # -----------------------------------------------------
    # TIME SERIES
    # -----------------------------------------------------

    if (
        time_column
        and len(
            df
        ) >= 2
    ):

        metric_column = (
            _preferred_metric_column(
                df,
                exclude=[
                    time_column
                ],
            )
        )

        if metric_column:

            return (
                _build_time_series(
                    df,
                    x_column=time_column,
                    y_column=metric_column,
                    question=question,
                )
            )

    # -----------------------------------------------------
    # CATEGORY RANKING / COMPARISON
    # -----------------------------------------------------

    if (
        category_column
        and len(
            df
        ) >= 2
    ):

        usable_metrics = [
            column
            for column in numeric_columns
            if column
            != category_column
        ]

        if len(
            usable_metrics
        ) == 1:

            return (
                _build_bar_chart(
                    df,
                    category_column=category_column,
                    metric_column=usable_metrics[
                        0
                    ],
                    question=question,
                )
            )

        if len(
            usable_metrics
        ) >= 2:

            # If the result has only a few rows, a grouped
            # comparison is more useful than a large chart.
            if len(
                df
            ) <= 8:

                return (
                    _build_multi_metric_comparison(
                        df,
                        category_column=category_column,
                        metric_columns=usable_metrics,
                        question=question,
                    )
                )

            metric_column = (
                _preferred_metric_column(
                    df,
                    exclude=[
                        category_column
                    ],
                )
            )

            if metric_column:

                return (
                    _build_bar_chart(
                        df,
                        category_column=category_column,
                        metric_column=metric_column,
                        question=question,
                    )
                )

    # -----------------------------------------------------
    # FALLBACK TABLE
    # -----------------------------------------------------

    return {
        "kind":
            "table",

        "title":
            "Structured Analysis",

        "figure":
            None,

        "data":
            df,

        "question":
            question,
    }


# =========================================================
# SAVE CHART
# =========================================================

def save_visualization(
    visualization: dict,
    filename: str | None = None,
) -> Path | None:
    """
    Save a Plotly chart as interactive HTML.

    KPI/table outputs are not saved as chart files.
    """

    if (
        visualization.get(
            "kind"
        )
        != "chart"
    ):

        return None

    figure = (
        visualization.get(
            "figure"
        )
    )

    if figure is None:
        return None

    if not filename:

        filename = (
            _slugify(
                visualization.get(
                    "question",
                    "chart",
                )
            )
            + ".html"
        )

    output_path = (
        CHART_DIR
        / filename
    )

    figure.write_html(
        output_path,
        include_plotlyjs="cdn",
    )

    return output_path


# =========================================================
# CLI TEST
# =========================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Test automatic visualization generation "
                "for the Delta Business Analyst."
            )
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help=(
            "Business question to analyze and visualize."
        ),
    )

    args = (
        parser.parse_args()
    )

    # Lazy import avoids unnecessary dependency coupling
    # when this module is imported by Streamlit later.
    from src.analytics.business_analyst import (
        run_business_analyst,
    )

    question = (
        args.question
        .strip()
    )

    print(
        "=" * 80
    )

    print(
        "DELTA BUSINESS ANALYST"
    )

    print(
        "AUTOMATIC VISUALIZATION TEST"
    )

    print(
        "=" * 80
    )

    print(
        f"\nQuestion:\n"
        f"{question}"
    )

    result = (
        run_business_analyst(
            question
        )
    )

    visualization = (
        build_visualization(
            question=question,
            sql_output=result.get(
                "sql"
            ),
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "VISUALIZATION RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"\nKind: "
        f"{visualization.get('kind')}"
    )

    print(
        f"Title: "
        f"{visualization.get('title')}"
    )

    if (
        visualization.get(
            "kind"
        )
        == "metric"
    ):

        print(
            f"Metric: "
            f"{visualization.get('metric_label')}"
        )

        print(
            f"Value: "
            f"{visualization.get('metric_value')}"
        )

    elif (
        visualization.get(
            "kind"
        )
        == "chart"
    ):

        print(
            f"Chart type: "
            f"{visualization.get('chart_type')}"
        )

        saved_path = (
            save_visualization(
                visualization
            )
        )

        print(
            f"\nSaved chart:\n"
            f"{saved_path}"
        )

    elif (
        visualization.get(
            "kind"
        )
        == "table"
    ):

        print(
            "\nTable preview:"
        )

        print(
            visualization[
                "data"
            ]
            .head(
                10
            )
            .to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo visualization was generated."
        )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
