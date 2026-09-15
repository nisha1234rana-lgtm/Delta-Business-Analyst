from __future__ import annotations

import re
from typing import Any


# =========================================================
# CORE HELPERS
# =========================================================

def _safe_records(
    sql_output: dict | None,
) -> list[dict]:
    """
    Extract SQL result rows as a list of dictionaries.
    """

    if not sql_output:
        return []

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
        return []

    try:
        return data.to_dict(
            "records"
        )
    except AttributeError:
        pass

    if isinstance(
        data,
        list,
    ):
        return data

    if isinstance(
        data,
        dict,
    ):
        return [
            data
        ]

    return []


def _clean_text(
    text: str,
) -> str:
    """
    Normalize retrieved filing text.
    """

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def _format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    """
    Format a numeric value without adding a business unit.
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

    if number.is_integer():
        return str(
            int(
                number
            )
        )

    return (
        f"{number:.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )


def _humanize_key(
    key: str,
) -> str:
    """
    Convert SQL column names into readable labels.
    """

    label = (
        key
        .replace(
            "_pct",
            " (%)",
        )
        .replace(
            "_pp",
            " (percentage points)",
        )
        .replace(
            "_min",
            " (minutes)",
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
        else key
    )


def _first_relevant_text(
    rag_results: list[dict] | None,
    max_items: int = 5,
) -> str:
    """
    Join the top retrieved filing chunks for qualitative analysis.
    """

    if not rag_results:
        return ""

    parts = []

    for item in rag_results[:max_items]:

        text = _clean_text(
            item.get(
                "text",
                "",
            )
        )

        if text:
            parts.append(
                text
            )

    return " ".join(
        parts
    )


# =========================================================
# SQL SUMMARIES
# =========================================================

def format_sql_summary(
    sql_output: dict | None,
) -> str:
    """
    Create a readable structured-data summary.

    IMPORTANT:
    This is the only source of numerical facts used by HYBRID answers.
    """

    records = (
        _safe_records(
            sql_output
        )
    )

    if not records:
        return (
            "No structured result was returned."
        )

    # -----------------------------------------------------
    # SINGLE SCALAR
    # -----------------------------------------------------

    if (
        len(records) == 1
        and len(records[0]) == 1
    ):

        row = records[0]

        key = next(
            iter(
                row
            )
        )

        value = row[
            key
        ]

        return (
            f"{_humanize_key(key)}: "
            f"{_format_number(value)}"
        )

    # -----------------------------------------------------
    # SMALL MULTI-ROW RESULT
    # -----------------------------------------------------

    if len(records) <= 5:

        formatted_rows = []

        for row in records:

            row_parts = []

            for key, value in row.items():

                row_parts.append(
                    (
                        f"{_humanize_key(key)} "
                        f"{_format_number(value)}"
                    )
                )

            formatted_rows.append(
                ", ".join(
                    row_parts
                )
            )

        return "; ".join(
            formatted_rows
        )

    return (
        f"{len(records)} structured rows were returned."
    )


# =========================================================
# RAG SUMMARIES
# =========================================================

def format_rag_summary(
    rag_results: list[dict] | None,
    max_items: int = 5,
) -> str:
    """
    Convert document evidence into compact trace text.

    This is for provenance / debugging.
    It is not used as a source of numerical facts in HYBRID answers.
    """

    if not rag_results:
        return (
            "No document evidence was retrieved."
        )

    sections = []

    for index, item in enumerate(
        rag_results[:max_items],
        start=1,
    ):

        form = (
            item.get(
                "form",
                "Unknown filing",
            )
        )

        report_date = (
            item.get(
                "report_date",
                "Unknown date",
            )
        )

        section = (
            item.get(
                "section",
                "Unknown section",
            )
        )

        text = _clean_text(
            item.get(
                "text",
                "",
            )
        )

        sections.append(
            (
                f"Evidence {index} | "
                f"{form} | "
                f"{report_date} | "
                f"{section}\n"
                f"{text[:1200]}"
            )
        )

    return "\n\n".join(
        sections
    )


# =========================================================
# SOURCE LIST
# =========================================================

def build_source_list(
    rag_results: list[dict] | None,
) -> list[dict]:
    """
    Build a de-duplicated source list for the UI.
    """

    if not rag_results:
        return []

    sources = []
    seen = set()

    for item in rag_results:

        source_url = (
            item.get(
                "source_url",
                "",
            )
        )

        form = (
            item.get(
                "form",
                "",
            )
        )

        report_date = (
            item.get(
                "report_date",
                "",
            )
        )

        section = (
            item.get(
                "section",
                "",
            )
        )

        key = (
            source_url,
            form,
            report_date,
            section,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "form":
                    form,

                "report_date":
                    report_date,

                "section":
                    section,

                "chunk_id":
                    item.get(
                        "chunk_id",
                        "",
                    ),

                "source_url":
                    source_url,
            }
        )

    return sources


# =========================================================
# QUALITATIVE MANAGEMENT DRIVER EXTRACTION
# =========================================================

def _contains_any(
    text: str,
    phrases: list[str],
) -> bool:
    """
    Case-insensitive phrase presence helper.
    """

    lower = text.lower()

    return any(
        phrase.lower() in lower
        for phrase in phrases
    )


def _management_driver_summary(
    question: str,
    rag_results: list[dict] | None,
) -> str:
    """
    Summarize management commentary qualitatively.

    CRITICAL DESIGN RULE:
    - This function NEVER extracts percentages, dollar amounts,
      counts, or other numerical facts from filing text.
    - HYBRID numerical facts must come from SQL only.
    """

    evidence = (
        _first_relevant_text(
            rag_results
        )
    )

    if not evidence:

        return (
            "No supporting management commentary "
            "was retrieved."
        )

    question_lower = (
        question.lower()
    )

    drivers = []

    # -----------------------------------------------------
    # FUEL / ENERGY
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "aircraft fuel",
            "jet fuel",
            "fuel expense",
            "fuel and related taxes",
        ],
    ):

        if _contains_any(
            evidence,
            [
                "higher",
                "increase",
                "increased",
                "rose",
                "rising",
            ],
        ):

            drivers.append(
                "higher fuel costs and jet-fuel pricing"
            )

        else:

            drivers.append(
                "fuel costs and jet-fuel pricing"
            )

    # -----------------------------------------------------
    # LABOR
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "salaries and related costs",
            "salaries",
            "wages",
            "labor costs",
            "employee costs",
        ],
    ):

        drivers.append(
            "higher salaries and related labor costs"
        )

    # -----------------------------------------------------
    # MAINTENANCE
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "maintenance materials and repairs",
            "maintenance expense",
            "aircraft maintenance",
        ],
    ):

        drivers.append(
            "maintenance-related expense pressure"
        )

    # -----------------------------------------------------
    # CAPACITY / OPERATIONS
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "capacity",
            "available seat miles",
            "asm",
        ],
    ):

        drivers.append(
            "changes in capacity and operating activity"
        )

    # -----------------------------------------------------
    # DEMAND / PRICING / PREMIUM
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "premium",
            "demand",
            "pricing",
            "yield",
            "passenger revenue",
        ],
    ):

        drivers.append(
            "demand and pricing conditions"
        )

    # -----------------------------------------------------
    # OTHER OPERATING COSTS
    # -----------------------------------------------------

    if _contains_any(
        evidence,
        [
            "other operating expense",
            "other operating expenses",
            "landing fees",
            "rentals",
            "selling expenses",
            "contracted services",
        ],
    ):

        drivers.append(
            "other operating expenses"
        )

    # -----------------------------------------------------
    # LIQUIDITY / DEBT
    # -----------------------------------------------------

    if (
        "liquidity" in question_lower
        or "debt" in question_lower
        or "cash" in question_lower
    ):

        if _contains_any(
            evidence,
            [
                "liquidity",
                "cash",
                "debt",
                "borrowings",
                "financing",
            ],
        ):

            drivers.append(
                "liquidity, financing and debt-management considerations"
            )

    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    if (
        "risk" in question_lower
        or "risks" in question_lower
    ):

        drivers.append(
            "the business risks described in the filing"
        )

    # -----------------------------------------------------
    # DE-DUPLICATE
    # -----------------------------------------------------

    unique_drivers = []

    seen = set()

    for driver in drivers:

        if driver in seen:
            continue

        seen.add(
            driver
        )

        unique_drivers.append(
            driver
        )

    # -----------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------

    if not unique_drivers:

        return (
            "Management commentary provides qualitative "
            "context for the structured result."
        )

    if len(unique_drivers) == 1:

        return (
            "Management commentary points to "
            f"{unique_drivers[0]}."
        )

    if len(unique_drivers) == 2:

        return (
            "Management commentary points to "
            f"{unique_drivers[0]} and "
            f"{unique_drivers[1]}."
        )

    return (
        "Management commentary points to "
        + ", ".join(
            unique_drivers[:-1]
        )
        + ", and "
        + unique_drivers[-1]
        + "."
    )


# =========================================================
# SQL-ONLY ANSWERS
# =========================================================

def _answer_single_metric_sql(
    sql_output: dict | None,
) -> str | None:
    """
    Produce a readable answer for a single SQL value.
    """

    records = (
        _safe_records(
            sql_output
        )
    )

    if (
        len(records) != 1
        or len(records[0]) != 1
    ):
        return None

    row = records[0]

    key = next(
        iter(
            row
        )
    )

    value = row[
        key
    ]

    friendly_names = {
        "cancellation_rate_pct":
            "Delta's cancellation rate",

        "on_time_arrival_pct":
            "Delta's on-time arrival rate",

        "scheduled_flights":
            "Delta's scheduled flight count",

        "margin_change_pp":
            "Delta's operating-margin change",

        "revenue_change_pct":
            "Delta's revenue change",

        "revenue_change_m":
            "Delta's revenue change",

        "operating_margin_pct":
            "Delta's operating margin",

        "operating_income_m":
            "Delta's operating income",

        "net_income_m":
            "Delta's net income",

        "debt_m":
            "Delta's debt",

        "cash_m":
            "Delta's cash",
    }

    label = (
        friendly_names.get(
            key,
            _humanize_key(
                key
            ),
        )
    )

    suffix = ""

    if key.endswith(
        "_pct"
    ):
        suffix = "%"

    elif key.endswith(
        "_pp"
    ):
        suffix = (
            " percentage points"
        )

    elif key.endswith(
        "_m"
    ):
        suffix = (
            " million"
        )

    return (
        f"{label} was "
        f"{_format_number(value)}"
        f"{suffix}."
    )


# =========================================================
# RAG-ONLY ANSWERS
# =========================================================

def _answer_rag_only(
    question: str,
    rag_results: list[dict] | None,
) -> str:
    """
    Produce a concise filing-based answer.

    NOTE:
    RAG-only questions may legitimately contain numerical
    information because the filing itself is the requested source.
    HYBRID answers do NOT use this function for numerical facts.
    """

    if not rag_results:

        return (
            "I could not find sufficiently relevant "
            "document evidence to answer the question."
        )

    evidence_text = (
        _first_relevant_text(
            rag_results
        )
    )

    if not evidence_text:

        return (
            "Relevant filing evidence was found, "
            "but no usable text was available."
        )

    cleaned = re.sub(
        (
            r"^ITEM\s+\d+\.?\s+"
            r"MANAGEMENT'?S?\s+DISCUSSION.*?"
            r"OPERATIONS\s*"
        ),
        "",
        evidence_text,
        flags=re.IGNORECASE,
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    question_terms = {
        token
        for token in re.findall(
            r"[a-zA-Z]{4,}",
            question.lower(),
        )
        if token not in {
            "what",
            "did",
            "delta",
            "management",
            "about",
            "does",
            "say",
            "said",
            "according",
            "provide",
        }
    }

    skip_phrases = [
        "should be read in conjunction",
        "included elsewhere",
        "condensed consolidated financial statements",
        "the table below",
        "the following table",
    ]

    candidates = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(
            sentence
        ) < 35:
            continue

        sentence_lower = (
            sentence.lower()
        )

        if any(
            phrase in sentence_lower
            for phrase in skip_phrases
        ):
            continue

        overlap = sum(
            1
            for term in question_terms
            if term in sentence_lower
        )

        if overlap <= 0:
            continue

        candidates.append(
            (
                overlap,
                sentence,
            )
        )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    useful = [
        sentence
        for _, sentence in candidates[:3]
    ]

    if not useful:

        useful = [
            sentence.strip()
            for sentence in sentences
            if len(
                sentence.strip()
            ) >= 35
        ][
            :2
        ]

    if not useful:

        return (
            "The retrieved filing contains relevant "
            "management commentary, but it could not be "
            "summarized cleanly."
        )

    return (
        "According to Delta's filing, "
        + " ".join(
            useful
        )
    )


# =========================================================
# HYBRID ANSWERS
# =========================================================

def _answer_operating_margin_change(
    question: str,
    sql_output: dict | None,
    rag_results: list[dict] | None,
) -> str | None:
    """
    Operating-margin HYBRID answer.

    CRITICAL RULE:
    The numerical margin change comes only from SQL.
    No filing percentage is parsed or attached to another metric.
    """

    records = (
        _safe_records(
            sql_output
        )
    )

    if not records:
        return None

    row = records[0]

    if (
        "margin_change_pp"
        not in row
    ):
        return None

    try:
        change = float(
            row[
                "margin_change_pp"
            ]
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    direction = (
        "declined"
        if change < 0
        else "increased"
    )

    qualitative_context = (
        _management_driver_summary(
            question=question,
            rag_results=rag_results,
        )
    )

    return (
        f"Delta's operating margin {direction} by "
        f"{abs(change):.2f} percentage points "
        f"from Q2 2025 to Q2 2026. "
        f"{qualitative_context}"
    )


def _answer_general_hybrid(
    question: str,
    sql_output: dict | None,
    rag_results: list[dict] | None,
) -> str:
    """
    General HYBRID answer.

    Numerical findings come from SQL.
    RAG contributes qualitative management context only.
    """

    records = (
        _safe_records(
            sql_output
        )
    )

    qualitative_context = (
        _management_driver_summary(
            question=question,
            rag_results=rag_results,
        )
    )

    if not records:

        return qualitative_context

    single_metric = (
        _answer_single_metric_sql(
            sql_output
        )
    )

    if single_metric:

        return (
            f"{single_metric} "
            f"{qualitative_context}"
        )

    return (
        "The structured analysis produced the numerical "
        "result shown in the table or chart. "
        f"{qualitative_context}"
    )


# =========================================================
# FINAL ANSWER GENERATOR
# =========================================================

def generate_rule_based_answer(
    question: str,
    route: str,
    sql_output: dict | None,
    rag_results: list[dict] | None,
) -> str:
    """
    Generate analyst-style answers.

    Architectural rule:
    - SQL owns numerical facts for SQL and HYBRID routes.
    - RAG owns management commentary / explanations.
    - HYBRID answers never scrape percentages from filing text.
    """

    route = (
        route.upper()
    )

    question_lower = (
        question.lower()
    )

    # -----------------------------------------------------
    # SQL
    # -----------------------------------------------------

    if route == "SQL":

        answer = (
            _answer_single_metric_sql(
                sql_output
            )
        )

        if answer:
            return answer

        records = (
            _safe_records(
                sql_output
            )
        )

        if not records:

            return (
                "The structured analysis completed, "
                "but no matching result was returned."
            )

        return (
            "The structured analysis returned the result "
            "shown in the chart or table."
        )

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    if route == "RAG":

        return (
            _answer_rag_only(
                question=question,
                rag_results=rag_results,
            )
        )

    # -----------------------------------------------------
    # HYBRID — OPERATING MARGIN
    # -----------------------------------------------------

    if (
        route == "HYBRID"
        and "operating margin" in question_lower
    ):

        answer = (
            _answer_operating_margin_change(
                question=question,
                sql_output=sql_output,
                rag_results=rag_results,
            )
        )

        if answer:
            return answer

    # -----------------------------------------------------
    # HYBRID — GENERAL
    # -----------------------------------------------------

    if route == "HYBRID":

        return (
            _answer_general_hybrid(
                question=question,
                sql_output=sql_output,
                rag_results=rag_results,
            )
        )

    return (
        "The analysis completed, but a final answer "
        "could not be generated for this route."
    )


# =========================================================
# FINAL RESPONSE OBJECT
# =========================================================

def build_final_response(
    question: str,
    route: str,
    sql_output: dict | None,
    rag_results: list[dict] | None,
) -> dict[str, Any]:
    """
    Build the final response object used by the CLI
    and Streamlit application.
    """

    answer = (
        generate_rule_based_answer(
            question=question,
            route=route,
            sql_output=sql_output,
            rag_results=rag_results,
        )
    )

    return {
        "question":
            question,

        "route":
            route,

        "answer":
            answer,

        "sources":
            build_source_list(
                rag_results
            ),

        "sql_summary":
            format_sql_summary(
                sql_output
            ),

        "rag_summary":
            format_rag_summary(
                rag_results
            ),
    }
