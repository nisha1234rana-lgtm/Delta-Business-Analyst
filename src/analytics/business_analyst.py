from pathlib import Path
import argparse
import sys


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
# CORE IMPORTS
# =========================================================

from src.routing.question_router import (
    route_question,
)

from src.sql_agent.nl_to_sql import (
    answer_sql_question,
)

from src.analytics.answer_generator import (
    build_final_response,
)


# =========================================================
# RAG METADATA RESOLUTION
# =========================================================

def resolve_rag_filters(
    routing: dict,
):
    """
    Convert router metadata into retrieval filters.

    Rules:
    - If one year exists, use it.
    - If multiple years exist, use the latest year.
    - If a quarter exists, use it.
    - Otherwise retrieval stays unfiltered.
    """

    years = (
        routing.get(
            "years",
            [],
        )
        or []
    )

    quarter = (
        routing.get(
            "quarter"
        )
    )

    year_filter = None

    if len(years) == 1:
        year_filter = (
            years[0]
        )

    elif len(years) >= 2:
        year_filter = (
            max(
                years
            )
        )

    return {
        "year":
            year_filter,

        "quarter":
            quarter,
    }


# =========================================================
# RAG EXECUTION
# =========================================================

def run_rag(
    question: str,
    routing: dict,
    top_k: int = 5,
):
    """
    Run SEC-document retrieval.

    Retriever is imported lazily so SQL-only questions
    do not load embedding and reranking models.

    Uses router metadata to prioritize the relevant
    filing period.
    """

    # -----------------------------------------------------
    # LAZY IMPORT
    # -----------------------------------------------------

    from src.rag.reranked_retriever import (
        search,
    )

    # -----------------------------------------------------
    # RESOLVE FILTERS
    # -----------------------------------------------------

    filters = (
        resolve_rag_filters(
            routing
        )
    )

    year_filter = (
        filters[
            "year"
        ]
    )

    quarter_filter = (
        filters[
            "quarter"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DOCUMENT RETRIEVAL"
    )

    print(
        "=" * 80
    )

    print(
        f"\nYear filter: "
        f"{year_filter}"
    )

    print(
        f"Quarter filter: "
        f"{quarter_filter}"
    )

    # -----------------------------------------------------
    # FILTERED RETRIEVAL
    # -----------------------------------------------------

    results = search(
        query=question,
        top_k=top_k,
        year=year_filter,
        quarter=quarter_filter,
    )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    no_results = (
        results is None
        or (
            hasattr(
                results,
                "empty",
            )
            and results.empty
        )
    )

    if no_results:

        print(
            "\nNo evidence found with "
            "period filters."
        )

        print(
            "Retrying without metadata filters..."
        )

        results = search(
            query=question,
            top_k=top_k,
        )

    # -----------------------------------------------------
    # NORMALIZE
    # -----------------------------------------------------

    if results is None:
        return []

    if hasattr(
        results,
        "to_dict",
    ):
        return (
            results.to_dict(
                "records"
            )
        )

    return results


# =========================================================
# BUSINESS ANALYST ORCHESTRATOR
# =========================================================

def run_business_analyst(
    question: str,
):
    """
    Main Delta Business Analyst orchestrator.

    SQL:
        Structured operational / financial analysis.

    RAG:
        SEC filing and management commentary retrieval.

    HYBRID:
        Structured analytics + filing evidence.

    Every route also produces a final user-facing answer.
    """

    # -----------------------------------------------------
    # ROUTE QUESTION
    # -----------------------------------------------------

    routing = (
        route_question(
            question
        )
    )

    route_type = (
        routing[
            "route"
        ]
        .strip()
        .upper()
    )

    output = {
        "question":
            question,

        "route":
            route_type,

        "routing":
            routing,

        "sql":
            None,

        "rag":
            None,

        "final":
            None,
    }

    # =====================================================
    # SQL ROUTE
    # =====================================================

    if route_type == "SQL":

        sql_result = (
            answer_sql_question(
                question
            )
        )

        output[
            "sql"
        ] = sql_result

        output[
            "final"
        ] = build_final_response(
            question=question,
            route=route_type,
            sql_output=output[
                "sql"
            ],
            rag_results=output[
                "rag"
            ],
        )

        return output

    # =====================================================
    # RAG ROUTE
    # =====================================================

    if route_type == "RAG":

        rag_result = (
            run_rag(
                question=question,
                routing=routing,
                top_k=5,
            )
        )

        output[
            "rag"
        ] = rag_result

        output[
            "final"
        ] = build_final_response(
            question=question,
            route=route_type,
            sql_output=output[
                "sql"
            ],
            rag_results=output[
                "rag"
            ],
        )

        return output

    # =====================================================
    # HYBRID ROUTE
    # =====================================================

    if route_type == "HYBRID":

        sql_result = (
            answer_sql_question(
                question
            )
        )

        rag_result = (
            run_rag(
                question=question,
                routing=routing,
                top_k=5,
            )
        )

        output[
            "sql"
        ] = sql_result

        output[
            "rag"
        ] = rag_result

        output[
            "final"
        ] = build_final_response(
            question=question,
            route=route_type,
            sql_output=output[
                "sql"
            ],
            rag_results=output[
                "rag"
            ],
        )

        return output

    # =====================================================
    # UNKNOWN ROUTE
    # =====================================================

    raise ValueError(
        f"Unsupported route type: "
        f"{route_type}"
    )


# =========================================================
# DISPLAY SQL
# =========================================================

def print_sql_result(
    sql_output,
):

    if sql_output is None:
        return

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STRUCTURED ANALYSIS"
    )

    print(
        "=" * 80
    )

    source = (
        sql_output.get(
            "source",
            "unknown",
        )
    )

    pattern = (
        sql_output.get(
            "pattern"
        )
    )

    attempts = (
        sql_output.get(
            "attempts"
        )
    )

    print(
        f"\nSource: "
        f"{source}"
    )

    if pattern:

        print(
            f"Pattern: "
            f"{pattern}"
        )

    if attempts is not None:

        print(
            f"Attempts: "
            f"{attempts}"
        )

    result = (
        sql_output.get(
            "result"
        )
    )

    if result is None:

        print(
            "\nNo SQL result returned."
        )

        return

    data = (
        result.get(
            "data"
        )
    )

    if data is None:

        print(
            "\nNo SQL data returned."
        )

        return

    print(
        "\nResult:"
    )

    try:

        print(
            data.to_string(
                index=False
            )
        )

    except AttributeError:

        print(
            data
        )


# =========================================================
# DISPLAY RAG
# =========================================================

def print_rag_results(
    rag_results,
):

    if rag_results is None:
        return

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DOCUMENT EVIDENCE"
    )

    print(
        "=" * 80
    )

    if not rag_results:

        print(
            "\nNo relevant document "
            "evidence found."
        )

        return

    for index, item in enumerate(
        rag_results,
        start=1,
    ):

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"Evidence {index}"
        )

        print(
            "-" * 80
        )

        form = (
            item.get(
                "form",
                ""
            )
        )

        report_date = (
            item.get(
                "report_date",
                ""
            )
        )

        section = (
            item.get(
                "section",
                ""
            )
        )

        score = (
            item.get(
                "reranker_score"
            )
        )

        chunk_id = (
            item.get(
                "chunk_id",
                ""
            )
        )

        source_url = (
            item.get(
                "source_url",
                ""
            )
        )

        print(
            f"Form: "
            f"{form}"
        )

        print(
            f"Report date: "
            f"{report_date}"
        )

        print(
            f"Section: "
            f"{section}"
        )

        if score is not None:

            try:

                print(
                    f"Reranker score: "
                    f"{float(score):.4f}"
                )

            except (
                TypeError,
                ValueError,
            ):

                print(
                    f"Reranker score: "
                    f"{score}"
                )

        if chunk_id:

            print(
                f"Chunk: "
                f"{chunk_id}"
            )

        if source_url:

            print(
                f"Source URL: "
                f"{source_url}"
            )

        text = (
            item.get(
                "text",
                ""
            )
        )

        print(
            "\nEvidence text:"
        )

        print(
            text[
                :1500
            ]
        )

        if (
            len(
                text
            )
            > 1500
        ):

            print(
                "..."
            )


# =========================================================
# DISPLAY FINAL ANSWER
# =========================================================

def print_final_answer(
    final_output,
):
    """
    Print the final user-facing analyst answer.
    """

    if not final_output:
        return

    answer = (
        final_output.get(
            "answer",
            ""
        )
    )

    sources = (
        final_output.get(
            "sources",
            [],
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 80
    )

    print(
        f"\n{answer}"
    )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    if sources:

        print(
            "\nSources:"
        )

        displayed = set()

        source_number = 1

        for source in sources:

            source_url = (
                source.get(
                    "source_url",
                    ""
                )
            )

            form = (
                source.get(
                    "form",
                    ""
                )
            )

            report_date = (
                source.get(
                    "report_date",
                    ""
                )
            )

            section = (
                source.get(
                    "section",
                    ""
                )
            )

            unique_key = (
                source_url,
                form,
                report_date,
            )

            if unique_key in displayed:
                continue

            displayed.add(
                unique_key
            )

            print(
                f"\n[{source_number}] "
                f"{form} | "
                f"{report_date}"
            )

            if section:

                print(
                    f"    Section: "
                    f"{section}"
                )

            if source_url:

                print(
                    f"    {source_url}"
                )

            source_number += 1


# =========================================================
# CLI
# =========================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Delta Business Analyst "
                "SQL + RAG + Hybrid orchestrator."
            )
        )
    )

    parser.add_argument(
        "question",
        type=str,
        help=(
            "Business question "
            "to analyze."
        ),
    )

    args = (
        parser.parse_args()
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
        "=" * 80
    )

    print(
        f"\nQuestion:\n"
        f"{question}"
    )

    # -----------------------------------------------------
    # RUN ANALYST
    # -----------------------------------------------------

    output = (
        run_business_analyst(
            question
        )
    )

    # =====================================================
    # ROUTING
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ROUTING"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRoute: "
        f"{output['route']}"
    )

    routing = (
        output.get(
            "routing",
            {}
        )
    )

    confidence = (
        routing.get(
            "confidence"
        )
    )

    if confidence is not None:

        try:

            print(
                f"Confidence: "
                f"{float(confidence):.3f}"
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                f"Confidence: "
                f"{confidence}"
            )

    reason = (
        routing.get(
            "reason"
        )
    )

    if reason:

        print(
            f"Reason: "
            f"{reason}"
        )

    years = (
        routing.get(
            "years"
        )
    )

    quarter = (
        routing.get(
            "quarter"
        )
    )

    month = (
        routing.get(
            "month"
        )
    )

    if years:

        print(
            f"Years: "
            f"{years}"
        )

    if quarter:

        print(
            f"Quarter: "
            f"Q{quarter}"
        )

    if month:

        print(
            f"Month: "
            f"{month}"
        )

    # =====================================================
    # STRUCTURED ANALYSIS
    # =====================================================

    print_sql_result(
        output.get(
            "sql"
        )
    )

    # =====================================================
    # DOCUMENT EVIDENCE
    # =====================================================

    print_rag_results(
        output.get(
            "rag"
        )
    )

    # =====================================================
    # FINAL ANSWER
    # =====================================================

    print_final_answer(
        output.get(
            "final"
        )
    )

    # =====================================================
    # ANALYSIS COMPLETE
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 80
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()