from __future__ import annotations

from pathlib import Path
import sys
import time
import traceback

import pandas as pd


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
# IMPORT SYSTEM UNDER TEST
# =========================================================

from src.analytics.business_analyst import (
    run_business_analyst,
)


# =========================================================
# OUTPUT PATHS
# =========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "integration"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "integration_benchmark_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "integration_benchmark_summary.csv"
)


# =========================================================
# BENCHMARK CASES
# =========================================================

TEST_CASES = [

    # =====================================================
    # SQL — 6
    # =====================================================

    {
        "id": "INT001",
        "category": "SQL",
        "question":
            "What was Delta's cancellation rate in June 2026?",
        "expected_route": "SQL",
        "expected_years": [2026],
        "expected_quarter": None,
        "expected_month": 6,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    {
        "id": "INT002",
        "category": "SQL",
        "question":
            "Which five airports had the worst on-time performance in 2025?",
        "expected_route": "SQL",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    {
        "id": "INT003",
        "category": "SQL",
        "question":
            "How many Delta-marketed flights operated in 2024?",
        "expected_route": "SQL",
        "expected_years": [2024],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    {
        "id": "INT004",
        "category": "SQL",
        "question":
            "Compare Delta operating margin in Q2 2025 and Q2 2026.",
        "expected_route": "SQL",
        "expected_years": [2025, 2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    {
        "id": "INT005",
        "category": "SQL",
        "question":
            "Which operating carrier had the highest cancellation rate in 2025?",
        "expected_route": "SQL",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    {
        "id": "INT006",
        "category": "SQL",
        "question":
            "How did Delta's Q2 revenue change from 2025 to 2026?",
        "expected_route": "SQL",
        "expected_years": [2025, 2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": False,
        "require_final": True,
    },

    # =====================================================
    # RAG — 5
    # =====================================================

    {
        "id": "INT007",
        "category": "RAG",
        "question":
            "What did management say about premium travel demand?",
        "expected_route": "RAG",
        "expected_years": [],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": False,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT008",
        "category": "RAG",
        "question":
            "What risks did Delta discuss in its 2025 10-K?",
        "expected_route": "RAG",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": False,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT009",
        "category": "RAG",
        "question":
            "What did Delta say about liquidity and debt?",
        "expected_route": "RAG",
        "expected_years": [],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": False,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT010",
        "category": "RAG",
        "question":
            "According to management, how is Delta thinking about premium products?",
        "expected_route": "RAG",
        "expected_years": [],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": False,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT011",
        "category": "RAG",
        "question":
            "What did Delta management say about fuel costs?",
        "expected_route": "RAG",
        "expected_years": [],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": False,
        "require_rag": True,
        "require_final": True,
    },

    # =====================================================
    # HYBRID — 7
    # =====================================================

    {
        "id": "INT012",
        "category": "HYBRID",
        "question":
            "Why did operating margin decline in Q2 2026?",
        "expected_route": "HYBRID",
        "expected_years": [2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT013",
        "category": "HYBRID",
        "question":
            "How did Delta's operating margin change from Q2 2025 to Q2 2026, and what explanation did management provide?",
        "expected_route": "HYBRID",
        "expected_years": [2025, 2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT014",
        "category": "HYBRID",
        "question":
            "How did revenue change in Q2 2026 and what did management say caused the change?",
        "expected_route": "HYBRID",
        "expected_years": [2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT015",
        "category": "HYBRID",
        "question":
            "How did fuel cost change in Q2 2026 and what affected it?",
        "expected_route": "HYBRID",
        "expected_years": [2026],
        "expected_quarter": 2,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT016",
        "category": "HYBRID",
        "question":
            "Which airport had the largest on-time decline in 2025 and what did management say about operational performance?",
        "expected_route": "HYBRID",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT017",
        "category": "HYBRID",
        "question":
            "Did Delta's operational performance worsen in 2025 and how did management explain the year?",
        "expected_route": "HYBRID",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },

    {
        "id": "INT018",
        "category": "HYBRID",
        "question":
            "Which partner airline performed worst in 2025 and did Delta discuss regional carrier performance?",
        "expected_route": "HYBRID",
        "expected_years": [2025],
        "expected_quarter": None,
        "expected_month": None,
        "require_sql": True,
        "require_rag": True,
        "require_final": True,
    },
]


# =========================================================
# HELPERS
# =========================================================

def normalize_years(
    value,
) -> list[int]:

    if value is None:
        return []

    return sorted(
        [
            int(year)
            for year in value
        ]
    )


def sql_has_data(
    sql_output,
) -> bool:

    if not sql_output:
        return False

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
        return False

    try:
        return not data.empty
    except AttributeError:
        return bool(
            data
        )


def rag_has_data(
    rag_output,
) -> bool:

    if rag_output is None:
        return False

    try:
        return len(
            rag_output
        ) > 0
    except TypeError:
        return False


def final_has_answer(
    final_output,
) -> bool:

    if not final_output:
        return False

    answer = (
        final_output.get(
            "answer",
            ""
        )
    )

    if answer is None:
        return False

    answer = str(
        answer
    ).strip()

    return len(
        answer
    ) >= 20


def rag_period_matches(
    rag_output,
    expected_years: list[int],
    expected_quarter: int | None,
) -> bool | None:
    """
    Validate retrieval period only when the test gives
    explicit temporal expectations.

    For comparison questions, the latest year is expected
    because business_analyst intentionally retrieves the
    later comparison period for management explanation.
    """

    if not rag_output:
        return False

    if (
        not expected_years
        and expected_quarter is None
    ):
        return None

    expected_year = (
        max(
            expected_years
        )
        if expected_years
        else None
    )

    first = (
        rag_output[0]
    )

    report_date = str(
        first.get(
            "report_date",
            ""
        )
    )

    if (
        expected_year is not None
        and not report_date.startswith(
            str(
                expected_year
            )
        )
    ):
        return False

    if expected_quarter is None:
        return True

    quarter_months = {
        1: "03",
        2: "06",
        3: "09",
        4: "12",
    }

    expected_month = (
        quarter_months[
            expected_quarter
        ]
    )

    parts = (
        report_date.split(
            "-"
        )
    )

    if len(parts) < 2:
        return False

    return (
        parts[1]
        == expected_month
    )


def status(
    condition,
) -> str:

    if condition is None:
        return "N/A"

    return (
        "PASS"
        if condition
        else "FAIL"
    )


# =========================================================
# RUN BENCHMARK
# =========================================================

def main():

    print(
        "=" * 88
    )

    print(
        "DELTA BUSINESS ANALYST"
    )

    print(
        "FULL INTEGRATION BENCHMARK"
    )

    print(
        "=" * 88
    )

    print(
        f"\nQuestions: "
        f"{len(TEST_CASES)}"
    )

    benchmark_start = (
        time.perf_counter()
    )

    rows = []

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print(
            "\n"
            + "=" * 88
        )

        print(
            f"[{index:02d}/{len(TEST_CASES)}] "
            f"{case['id']} | "
            f"{case['category']}"
        )

        print(
            case[
                "question"
            ]
        )

        print(
            "=" * 88
        )

        started = (
            time.perf_counter()
        )

        error = ""

        try:

            output = (
                run_business_analyst(
                    case[
                        "question"
                    ]
                )
            )

            route = (
                output.get(
                    "route",
                    ""
                )
                .strip()
                .upper()
            )

            routing = (
                output.get(
                    "routing",
                    {}
                )
                or {}
            )

            sql_output = (
                output.get(
                    "sql"
                )
            )

            rag_output = (
                output.get(
                    "rag"
                )
            )

            final_output = (
                output.get(
                    "final"
                )
            )

            route_ok = (
                route
                == case[
                    "expected_route"
                ]
            )

            actual_years = (
                normalize_years(
                    routing.get(
                        "years",
                        []
                    )
                )
            )

            expected_years = (
                normalize_years(
                    case[
                        "expected_years"
                    ]
                )
            )

            years_ok = (
                actual_years
                == expected_years
            )

            quarter_ok = (
                routing.get(
                    "quarter"
                )
                == case[
                    "expected_quarter"
                ]
            )

            month_ok = (
                routing.get(
                    "month"
                )
                == case[
                    "expected_month"
                ]
            )

            if case[
                "require_sql"
            ]:

                sql_ok = (
                    sql_has_data(
                        sql_output
                    )
                )

            else:

                sql_ok = (
                    sql_output
                    is None
                )

            if case[
                "require_rag"
            ]:

                rag_ok = (
                    rag_has_data(
                        rag_output
                    )
                )

            else:

                rag_ok = (
                    rag_output
                    is None
                )

            if case[
                "require_final"
            ]:

                final_ok = (
                    final_has_answer(
                        final_output
                    )
                )

            else:

                final_ok = True

            if case[
                "require_rag"
            ]:

                period_ok = (
                    rag_period_matches(
                        rag_output,
                        expected_years=expected_years,
                        expected_quarter=case[
                            "expected_quarter"
                        ],
                    )
                )

            else:

                period_ok = None

            overall_ok = all(
                check
                for check in [
                    route_ok,
                    years_ok,
                    quarter_ok,
                    month_ok,
                    sql_ok,
                    rag_ok,
                    final_ok,
                    (
                        True
                        if period_ok is None
                        else period_ok
                    ),
                ]
            )

            final_answer = ""

            if final_output:

                final_answer = str(
                    final_output.get(
                        "answer",
                        ""
                    )
                )

        except Exception as exc:

            route = ""
            actual_years = []
            route_ok = False
            years_ok = False
            quarter_ok = False
            month_ok = False
            sql_ok = False
            rag_ok = False
            final_ok = False
            period_ok = False
            overall_ok = False
            final_answer = ""

            error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            traceback.print_exc()

        elapsed = (
            time.perf_counter()
            - started
        )

        row = {
            "id":
                case[
                    "id"
                ],

            "category":
                case[
                    "category"
                ],

            "question":
                case[
                    "question"
                ],

            "expected_route":
                case[
                    "expected_route"
                ],

            "actual_route":
                route,

            "route_pass":
                route_ok,

            "expected_years":
                str(
                    expected_years
                ),

            "actual_years":
                str(
                    actual_years
                ),

            "years_pass":
                years_ok,

            "expected_quarter":
                case[
                    "expected_quarter"
                ],

            "quarter_pass":
                quarter_ok,

            "expected_month":
                case[
                    "expected_month"
                ],

            "month_pass":
                month_ok,

            "sql_pass":
                sql_ok,

            "rag_pass":
                rag_ok,

            "rag_period_pass":
                period_ok,

            "final_answer_pass":
                final_ok,

            "overall_pass":
                overall_ok,

            "latency_seconds":
                round(
                    elapsed,
                    2,
                ),

            "final_answer":
                final_answer,

            "error":
                error,
        }

        rows.append(
            row
        )

        print(
            f"\nRoute:          "
            f"{status(route_ok)} "
            f"({route or 'ERROR'})"
        )

        print(
            f"Years:          "
            f"{status(years_ok)} "
            f"{actual_years}"
        )

        print(
            f"Quarter:        "
            f"{status(quarter_ok)}"
        )

        print(
            f"Month:          "
            f"{status(month_ok)}"
        )

        print(
            f"SQL:            "
            f"{status(sql_ok)}"
        )

        print(
            f"RAG:            "
            f"{status(rag_ok)}"
        )

        print(
            f"RAG period:     "
            f"{status(period_ok)}"
        )

        print(
            f"Final answer:   "
            f"{status(final_ok)}"
        )

        print(
            f"Overall:        "
            f"{status(overall_ok)}"
        )

        print(
            f"Latency:        "
            f"{elapsed:.2f}s"
        )

        if error:

            print(
                f"Error:          "
                f"{error}"
            )

    # =====================================================
    # RESULTS
    # =====================================================

    results_df = (
        pd.DataFrame(
            rows
        )
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    total = len(
        results_df
    )

    overall_passes = int(
        results_df[
            "overall_pass"
        ]
        .sum()
    )

    route_passes = int(
        results_df[
            "route_pass"
        ]
        .sum()
    )

    sql_required = (
        results_df[
            "category"
        ]
        .isin(
            [
                "SQL",
                "HYBRID",
            ]
        )
    )

    rag_required = (
        results_df[
            "category"
        ]
        .isin(
            [
                "RAG",
                "HYBRID",
            ]
        )
    )

    sql_count = int(
        sql_required.sum()
    )

    sql_passes = int(
        results_df.loc[
            sql_required,
            "sql_pass",
        ]
        .sum()
    )

    rag_count = int(
        rag_required.sum()
    )

    rag_passes = int(
        results_df.loc[
            rag_required,
            "rag_pass",
        ]
        .sum()
    )

    final_passes = int(
        results_df[
            "final_answer_pass"
        ]
        .sum()
    )

    period_rows = (
        results_df[
            "rag_period_pass"
        ]
        .notna()
    )

    period_count = int(
        period_rows.sum()
    )

    period_passes = int(
        results_df.loc[
            period_rows,
            "rag_period_pass",
        ]
        .sum()
    )

    total_runtime = (
        time.perf_counter()
        - benchmark_start
    )

    summary_rows = [
        {
            "metric": "Overall",
            "passed": overall_passes,
            "total": total,
            "percent": round(
                100.0
                * overall_passes
                / total,
                2,
            ),
        },

        {
            "metric": "Routing",
            "passed": route_passes,
            "total": total,
            "percent": round(
                100.0
                * route_passes
                / total,
                2,
            ),
        },

        {
            "metric": "SQL execution",
            "passed": sql_passes,
            "total": sql_count,
            "percent": round(
                100.0
                * sql_passes
                / sql_count,
                2,
            ),
        },

        {
            "metric": "RAG retrieval",
            "passed": rag_passes,
            "total": rag_count,
            "percent": round(
                100.0
                * rag_passes
                / rag_count,
                2,
            ),
        },

        {
            "metric": "RAG period",
            "passed": period_passes,
            "total": period_count,
            "percent": (
                round(
                    100.0
                    * period_passes
                    / period_count,
                    2,
                )
                if period_count
                else 0.0
            ),
        },

        {
            "metric": "Final answer present",
            "passed": final_passes,
            "total": total,
            "percent": round(
                100.0
                * final_passes
                / total,
                2,
            ),
        },
    ]

    summary_df = (
        pd.DataFrame(
            summary_rows
        )
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # =====================================================
    # CATEGORY SUMMARY
    # =====================================================

    category_summary = (
        results_df
        .groupby(
            "category"
        )[
            "overall_pass"
        ]
        .agg(
            [
                "sum",
                "count",
            ]
        )
        .reset_index()
    )

    category_summary[
        "percent"
    ] = (
        100.0
        * category_summary[
            "sum"
        ]
        / category_summary[
            "count"
        ]
    ).round(
        2
    )

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 88
    )

    print(
        "INTEGRATION BENCHMARK SUMMARY"
    )

    print(
        "=" * 88
    )

    print(
        "\n"
        + summary_df.to_string(
            index=False
        )
    )

    print(
        "\nCategory performance:"
    )

    print(
        category_summary.to_string(
            index=False
        )
    )

    print(
        f"\nTotal runtime: "
        f"{total_runtime:.2f}s"
    )

    failed = (
        results_df[
            ~results_df[
                "overall_pass"
            ]
        ]
    )

    if not failed.empty:

        print(
            "\n"
            + "=" * 88
        )

        print(
            "FAILED CASES"
        )

        print(
            "=" * 88
        )

        for _, row in failed.iterrows():

            print(
                f"\n{row['id']} | "
                f"{row['category']}"
            )

            print(
                row[
                    "question"
                ]
            )

            print(
                f"Expected route: "
                f"{row['expected_route']}"
            )

            print(
                f"Actual route:   "
                f"{row['actual_route']}"
            )

            print(
                f"Route:          "
                f"{status(row['route_pass'])}"
            )

            print(
                f"Years:          "
                f"{status(row['years_pass'])}"
            )

            print(
                f"Quarter:        "
                f"{status(row['quarter_pass'])}"
            )

            print(
                f"Month:          "
                f"{status(row['month_pass'])}"
            )

            print(
                f"SQL:            "
                f"{status(row['sql_pass'])}"
            )

            print(
                f"RAG:            "
                f"{status(row['rag_pass'])}"
            )

            rag_period_value = (
                row[
                    "rag_period_pass"
                ]
            )

            if pd.isna(
                rag_period_value
            ):
                rag_period_value = None

            print(
                f"RAG period:     "
                f"{status(rag_period_value)}"
            )

            print(
                f"Final answer:   "
                f"{status(row['final_answer_pass'])}"
            )

            if row[
                "error"
            ]:

                print(
                    f"Error:          "
                    f"{row['error']}"
                )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 88
    )

    print(
        f"\nDetailed results:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
