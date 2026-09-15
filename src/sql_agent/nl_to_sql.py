from pathlib import Path
import argparse
import csv
import json
import os
import re
import time

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.sql_agent.deterministic_sql import (
    get_deterministic_sql,
)

from src.sql_agent.safe_sql_runtime import (
    execute_safe_sql,
    load_schema_catalog,
)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sql_agent"
    / "nl_to_sql_hybrid_log.csv"
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv(
    PROJECT_ROOT
    / ".env"
)


OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:8b",
)


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


OLLAMA_CHAT_URL = (
    f"{OLLAMA_BASE_URL}/api/chat"
)


OLLAMA_TIMEOUT_SECONDS = int(
    os.getenv(
        "OLLAMA_TIMEOUT_SECONDS",
        "180",
    )
)


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

class SQLPlan(BaseModel):

    sql: str = Field(
        description=(
            "One executable DuckDB "
            "SELECT query."
        )
    )


    explanation: str = Field(
        description=(
            "Short explanation of "
            "what the query calculates."
        )
    )


    assumptions: list[str] = Field(
        default_factory=list
    )


class BusinessSemanticError(
    Exception
):
    pass


# =========================================================
# SCHEMA
# =========================================================

SCHEMA_CATALOG = (
    load_schema_catalog()
)


TABLE_DESCRIPTIONS = {

    "fact_flights":
        (
            "One row per scheduled "
            "Delta-marketed flight."
        ),

    "dim_airport":
        "Airport dimension.",

    "dim_route":
        "Directional route dimension.",

    "financial_quarterly":
        (
            "Quarterly Delta financial data "
            "from 2020-Q1 through 2026-Q2."
        ),

    "vw_monthly_kpis":
        "Monthly operational KPIs.",

    "vw_yearly_kpis":
        "Yearly operational KPIs.",

    "vw_airport_performance":
        "Airport performance view.",

    "vw_route_performance":
        "Route performance view.",

    "vw_operator_performance":
        "Operating-carrier performance.",

    "vw_delay_causes":
        "Delay-cause metrics.",

    "vw_financial_performance":
        "Quarterly financial KPI view.",
}


# =========================================================
# BUILD SCHEMA PROMPT
# =========================================================

def build_schema_prompt():

    sections = []


    for (
        table_name,
        details,
    ) in SCHEMA_CATALOG.items():

        description = (
            TABLE_DESCRIPTIONS.get(
                table_name,
                "",
            )
        )


        columns = []


        for column in details.get(
            "columns",
            [],
        ):

            columns.append(
                f"{column['name']} "
                f"({column['type']})"
            )


        sections.append(
            f"TABLE: {table_name}\n"
            f"DESCRIPTION: {description}\n"
            f"COLUMNS:\n"
            f"{', '.join(columns)}"
        )


    return "\n\n".join(
        sections
    )


SCHEMA_PROMPT = (
    build_schema_prompt()
)


# =========================================================
# LLM FALLBACK PROMPT
# =========================================================

SYSTEM_PROMPT = f"""
You generate DuckDB SELECT queries for a
Delta Air Lines business analytics system.

A deterministic analytics engine handles
common business calculations before you are called.

You are the FALLBACK for analytical questions
that do not match a trusted deterministic pattern.

DATABASE SCHEMA
================

{SCHEMA_PROMPT}


IMPORTANT RULES
===============

1. fact_flights already contains only
   Delta-marketed scheduled flights.

2. Do not add:
   Marketing_Airline_Network = 'Delta'

3. Cancellation rate:

   100.0 * AVG(
       CASE
           WHEN is_cancelled
           THEN 1
           ELSE 0
       END
   )

4. On-time arrival KPI:

   100.0 * AVG(
       CASE
           WHEN is_on_time_arrival
           THEN 1
           ELSE 0
       END
   )

5. Do not use:

   AVG(CAST(is_on_time_arrival AS INTEGER))

   because NULL handling can change the denominator.

6. Airport means Origin unless destination
   is explicitly requested.

7. Airport rankings normally require at least
   5,000 departures.

8. Route rankings normally require at least
   500 flights.

9. operator_type exact values include:

   'Delta Mainline'
   'Partner'

10. Never use 'Mainline' instead of
    'Delta Mainline'.

11. Use Operating_Airline for actual
    operating-carrier analysis.

12. Use the existing year column when possible.

13. First half / H1 means January-June.

14. When comparing periods, calculate each
    period using its own numerator and denominator.

15. Do not divide multiple period-specific
    numerators by one combined denominator.

16. Financial columns ending in _m represent
    USD millions.

17. Prefer revenue_m, operating_income_m,
    net_income_m and total_debt_m rather than
    raw-dollar versions for human-facing analysis.

18. quarter is numeric.

    Q1 = 1
    Q2 = 2
    Q3 = 3
    Q4 = 4

19. Do not assume period = 'Q2'.

20. Operational 2026 data ends June 30, 2026.

21. Financial 2026 data ends Q2.

22. Never invent tables or columns.

23. SELECT only.

24. Never use CREATE, DELETE, UPDATE, DROP,
    ALTER, INSERT, COPY, ATTACH or PRAGMA.

25. Never access external files or URLs.

26. Keep SQL simple and auditable.

27. Do not output Markdown code fences.

Return JSON:

{{
    "sql": "SELECT ...",
    "explanation": "short explanation",
    "assumptions": []
}}
"""


# =========================================================
# OLLAMA HEALTH
# =========================================================

def is_ollama_available():
    """
    Return True when the configured Ollama server is reachable.

    This keeps local development unchanged while allowing hosted environments
    such as Streamlit Community Cloud to fail fast instead of waiting for a
    long HTTP timeout against localhost.
    """

    try:

        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=3,
        )

        response.raise_for_status()

        return True

    except requests.RequestException:

        return False


def check_ollama():
    """
    Explicit health check retained for CLI/debugging use.
    """

    if not is_ollama_available():

        raise RuntimeError(
            "\nOllama is not reachable.\n"
            f"Expected URL: "
            f"{OLLAMA_BASE_URL}\n"
        )


# =========================================================
# NORMALIZE MODEL JSON
# =========================================================

def normalize_model_output(
    parsed,
):

    if (
        "sql"
        not in parsed
        and "query"
        in parsed
    ):

        parsed[
            "sql"
        ] = parsed[
            "query"
        ]


    if (
        "explanation"
        not in parsed
        and "description"
        in parsed
    ):

        parsed[
            "explanation"
        ] = parsed[
            "description"
        ]


    if (
        "explanation"
        not in parsed
        and "reasoning"
        in parsed
    ):

        parsed[
            "explanation"
        ] = parsed[
            "reasoning"
        ]


    if (
        "assumptions"
        not in parsed
    ):

        parsed[
            "assumptions"
        ] = []


    return parsed


# =========================================================
# JSON EXTRACTION
# =========================================================

def extract_json(
    text,
):

    text = (
        text.strip()
        .replace(
            "```json",
            "",
        )
        .replace(
            "```",
            "",
        )
    )


    first = (
        text.find(
            "{"
        )
    )


    last = (
        text.rfind(
            "}"
        )
    )


    if (
        first == -1
        or last == -1
    ):

        raise ValueError(
            "No JSON object found "
            "in model response."
        )


    return text[
        first:
        last + 1
    ]


# =========================================================
# BASIC BUSINESS VALIDATION
# =========================================================

def validate_business_semantics(
    question,
    sql,
):

    q = (
        question.lower()
    )


    s = re.sub(
        r"\s+",
        " ",
        sql.lower(),
    )


    errors = []


    # -----------------------------------------------------
    # REDUNDANT DELTA FILTER
    # -----------------------------------------------------

    if re.search(
        r"marketing_airline_network"
        r"\s*=\s*['\"]delta['\"]",
        s,
    ):

        errors.append(
            "fact_flights is already "
            "Delta-marketed."
        )


    # -----------------------------------------------------
    # MAINLINE VALUE
    # -----------------------------------------------------

    if (
        "mainline"
        in q
        and "operator_type"
        in s
        and re.search(
            r"['\"]mainline['\"]",
            s,
        )
    ):

        errors.append(
            "Use exact operator_type value "
            "'Delta Mainline'."
        )


    # -----------------------------------------------------
    # STANDARD ON-TIME KPI
    # -----------------------------------------------------

    if (
        (
            "on-time"
            in q
            or "on time"
            in q
        )
        and "fact_flights"
        in s
    ):

        if (
            "is_on_time_arrival"
            not in s
        ):

            errors.append(
                "On-time arrival analysis "
                "must use is_on_time_arrival."
            )


        if re.search(
            r"avg\s*\(\s*cast\s*\(\s*"
            r"is_on_time_arrival",
            s,
        ):

            errors.append(
                "Do not use AVG(CAST("
                "is_on_time_arrival AS INTEGER)). "
                "Use CASE WHEN ... THEN 1 ELSE 0 "
                "so scheduled-flight denominator "
                "is preserved."
            )


    # -----------------------------------------------------
    # FINANCIAL MILLIONS
    # -----------------------------------------------------

    if (
        "revenue"
        in q
        and "financial_quarterly"
        in s
    ):

        if re.search(
            r"\brevenue\b",
            s,
        ) and (
            "revenue_m"
            not in s
        ):

            errors.append(
                "Use revenue_m for human-facing "
                "financial analysis."
            )


    # -----------------------------------------------------
    # SAFER AIRPORT KEY
    # -----------------------------------------------------

    if (
        "airport"
        in q
        and "origin"
        in q
        and "fact_flights"
        in s
        and "originairportid"
        in s
    ):

        errors.append(
            "Use Origin airport code for this "
            "analysis rather than OriginAirportID."
        )


    if errors:

        raise BusinessSemanticError(
            "Business-semantic "
            "validation failed:\n"
            +
            "\n".join(
                f"- {error}"
                for error
                in errors
            )
        )


    return True


# =========================================================
# GENERATE FALLBACK SQL
# =========================================================

def generate_sql_plan(
    question,
    previous_sql=None,
    validation_error=None,
):

    user_message = (
        f"USER QUESTION:\n"
        f"{question}"
    )


    if validation_error:

        user_message += (
            "\n\nPREVIOUS ATTEMPT FAILED.\n"
            f"ERROR:\n"
            f"{validation_error}"
        )


    if previous_sql:

        user_message += (
            "\n\nPREVIOUS SQL:\n"
            f"{previous_sql}\n\n"
            "Generate corrected SQL."
        )


    payload = {

        "model":
            OLLAMA_MODEL,

        "stream":
            False,

        "keep_alive":
            "30m",

        "format":
            SQLPlan.model_json_schema(),

        "options": {
            "temperature":
                0,
        },

        "messages": [

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    user_message,
            },
        ],
    }


    response = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=(
            OLLAMA_TIMEOUT_SECONDS
        ),
    )


    response.raise_for_status()


    raw_content = (
        response.json()
        .get(
            "message",
            {},
        )
        .get(
            "content",
            "",
        )
    )


    if not raw_content:

        raise RuntimeError(
            "Ollama returned "
            "an empty response."
        )


    parsed = json.loads(
        extract_json(
            raw_content
        )
    )


    parsed = normalize_model_output(
        parsed
    )


    return SQLPlan.model_validate(
        parsed
    )


# =========================================================
# LOGGING
# =========================================================

def write_log(
    question,
    source,
    pattern,
    attempt,
    sql,
    success,
    error,
    elapsed_ms,
):

    record = {

        "question":
            question,

        "source":
            source,

        "pattern":
            pattern,

        "model":
            (
                OLLAMA_MODEL
                if source
                == "llm_fallback"
                else ""
            ),

        "attempt":
            attempt,

        "success":
            success,

        "elapsed_ms":
            elapsed_ms,

        "error":
            error,

        "sql":
            sql,
    }


    LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    exists = (
        LOG_PATH.exists()
    )


    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=(
                    record.keys()
                ),
            )
        )


        if not exists:

            writer.writeheader()


        writer.writerow(
            record
        )


# =========================================================
# EXECUTE A PLAN
# =========================================================

def execute_plan(
    question,
    plan,
    source,
    pattern,
    attempt=1,
):

    sql = (
        plan.sql
        .strip()
        .rstrip(
            ";"
        )
    )


    print(
        "\nGenerated SQL:"
    )


    print(
        sql
    )


    print(
        "\nPlan source:"
    )


    print(
        source
    )


    if pattern:

        print(
            "\nMatched pattern:"
        )

        print(
            pattern
        )


    print(
        "\nExplanation:"
    )


    print(
        plan.explanation
    )


    if plan.assumptions:

        print(
            "\nAssumptions:"
        )


        for assumption in (
            plan.assumptions
        ):

            print(
                f"  - {assumption}"
            )


    validate_business_semantics(
        question,
        sql,
    )


    result = execute_safe_sql(
        sql
    )


    return (
        sql,
        result,
    )


# =========================================================
# FULL HYBRID PIPELINE
# =========================================================

def answer_sql_question(
    question,
    max_attempts=3,
):

    pipeline_start = (
        time.perf_counter()
    )


    # =====================================================
    # 1. DETERMINISTIC ANALYTICS FIRST
    # =====================================================

    deterministic_plan = (
        get_deterministic_sql(
            question
        )
    )


    if (
        deterministic_plan
        is not None
    ):

        print(
            "\n"
            + "=" * 80
        )


        print(
            "DETERMINISTIC ANALYTICS MATCH"
        )


        print(
            "=" * 80
        )


        plan = SQLPlan(

            sql=(
                deterministic_plan.sql
            ),

            explanation=(
                deterministic_plan
                .explanation
            ),

            assumptions=(
                deterministic_plan
                .assumptions
            ),
        )


        try:

            (
                sql,
                result,
            ) = execute_plan(

                question=question,

                plan=plan,

                source=(
                    "deterministic_template"
                ),

                pattern=(
                    deterministic_plan
                    .pattern
                ),

                attempt=1,
            )


        except Exception as exc:

            elapsed_ms = (
                (
                    time.perf_counter()
                    - pipeline_start
                )
                * 1000
            )


            write_log(
                question=question,
                source=(
                    "deterministic_template"
                ),
                pattern=(
                    deterministic_plan
                    .pattern
                ),
                attempt=1,
                sql=(
                    deterministic_plan
                    .sql
                ),
                success=False,
                error=str(
                    exc
                ),
                elapsed_ms=round(
                    elapsed_ms,
                    2,
                ),
            )


            raise


        elapsed_ms = (
            (
                time.perf_counter()
                - pipeline_start
            )
            * 1000
        )


        if result[
            "success"
        ]:

            print(
                "\nBusiness semantic "
                "validation: PASS"
            )


            print(
                "SQL safety validation: PASS"
            )


            write_log(
                question=question,
                source=(
                    "deterministic_template"
                ),
                pattern=(
                    deterministic_plan
                    .pattern
                ),
                attempt=1,
                sql=sql,
                success=True,
                error=None,
                elapsed_ms=round(
                    elapsed_ms,
                    2,
                ),
            )


            return {

                "question":
                    question,

                "model":
                    "deterministic",

                "source":
                    (
                        "deterministic_template"
                    ),

                "pattern":
                    (
                        deterministic_plan
                        .pattern
                    ),

                "attempts":
                    1,

                "plan":
                    plan,

                "result":
                    result,
            }


        write_log(
            question=question,
            source=(
                "deterministic_template"
            ),
            pattern=(
                deterministic_plan
                .pattern
            ),
            attempt=1,
            sql=sql,
            success=False,
            error=result.get(
                "error"
            ),
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
        )


        raise RuntimeError(
            "Trusted deterministic SQL "
            "failed safety/database validation:\n"
            f"{result.get('error')}"
        )


    # =====================================================
    # 2. LLM FALLBACK
    # =====================================================

    print(
        "\n"
        + "=" * 80
    )


    print(
        "NO DETERMINISTIC TEMPLATE"
    )


    # On the local machine, Ollama remains available exactly as before.
    # On Streamlit Cloud (or any environment without Ollama), fail fast with
    # a clear message instead of attempting localhost calls for several
    # minutes and surfacing a connection traceback.
    if not is_ollama_available():

        elapsed_ms = (
            (
                time.perf_counter()
                - pipeline_start
            )
            * 1000
        )

        message = (
            "This question is outside the validated deterministic SQL "
            "patterns available in the hosted demo. "
            "Try one of the preset questions or ask about supported "
            "operational or financial KPIs. "
            "The local development version can use the Ollama/Qwen fallback "
            "for additional flexible NL-to-SQL questions."
        )

        write_log(
            question=question,
            source="llm_fallback_unavailable",
            pattern="",
            attempt=0,
            sql="",
            success=False,
            error=message,
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
        )

        raise RuntimeError(
            message
        )


    print(
        "USING LOCAL LLM FALLBACK"
    )


    print(
        "=" * 80
    )


    previous_sql = None

    validation_error = None


    for attempt in range(
        1,
        max_attempts + 1,
    ):

        print(
            "\n"
            + "=" * 80
        )


        print(
            f"LLM SQL GENERATION ATTEMPT "
            f"{attempt}/{max_attempts}"
        )


        print(
            "=" * 80
        )


        start = (
            time.perf_counter()
        )


        try:

            plan = (
                generate_sql_plan(
                    question=(
                        question
                    ),
                    previous_sql=(
                        previous_sql
                    ),
                    validation_error=(
                        validation_error
                    ),
                )
            )


            sql = (
                plan.sql
                .strip()
                .rstrip(
                    ";"
                )
            )


            validate_business_semantics(
                question,
                sql,
            )


        except Exception as exc:

            validation_error = (
                str(
                    exc
                )
            )


            elapsed_ms = (
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            )


            write_log(
                question=question,
                source=(
                    "llm_fallback"
                ),
                pattern="",
                attempt=attempt,
                sql=(
                    previous_sql
                    or ""
                ),
                success=False,
                error=(
                    validation_error
                ),
                elapsed_ms=round(
                    elapsed_ms,
                    2,
                ),
            )


            print(
                "\nGeneration/semantic "
                "validation failure:"
            )


            print(
                validation_error
            )


            if (
                attempt
                == max_attempts
            ):

                raise


            continue


        print(
            "\nGenerated SQL:"
        )


        print(
            sql
        )


        print(
            "\nPlan source:"
        )


        print(
            "llm_fallback"
        )


        print(
            "\nExplanation:"
        )


        print(
            plan.explanation
        )


        result = execute_safe_sql(
            sql
        )


        elapsed_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )


        if result[
            "success"
        ]:

            print(
                "\nBusiness semantic "
                "validation: PASS"
            )


            print(
                "SQL safety validation: PASS"
            )


            write_log(
                question=question,
                source=(
                    "llm_fallback"
                ),
                pattern="",
                attempt=attempt,
                sql=sql,
                success=True,
                error=None,
                elapsed_ms=round(
                    elapsed_ms,
                    2,
                ),
            )


            return {

                "question":
                    question,

                "model":
                    OLLAMA_MODEL,

                "source":
                    "llm_fallback",

                "pattern":
                    None,

                "attempts":
                    attempt,

                "plan":
                    plan,

                "result":
                    result,
            }


        previous_sql = (
            sql
        )


        validation_error = (
            result[
                "error"
            ]
        )


        write_log(
            question=question,
            source=(
                "llm_fallback"
            ),
            pattern="",
            attempt=attempt,
            sql=sql,
            success=False,
            error=(
                validation_error
            ),
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
        )


        print(
            "\nSQL validation rejected:"
        )


        print(
            validation_error
        )


    raise RuntimeError(
        "The SQL agent failed after "
        f"{max_attempts} attempts."
    )


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid deterministic + local LLM "
            "NL-to-SQL analytics engine."
        )
    )


    parser.add_argument(
        "question",
        type=str,
    )


    args = (
        parser.parse_args()
    )


    print("=" * 80)
    print("DELTA BUSINESS ANALYST")
    print("HYBRID NL → SQL ENGINE")
    print("=" * 80)


    print(
        f"\nQuestion:\n"
        f"{args.question}"
    )


    output = (
        answer_sql_question(
            args.question
        )
    )


    result = (
        output[
            "result"
        ]
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "EXECUTION RESULT"
    )


    print(
        "=" * 80
    )


    print(
        f"\nSource: "
        f"{output.get('source')}"
    )


    print(
        f"Pattern: "
        f"{output.get('pattern')}"
    )


    print(
        f"Attempts: "
        f"{output['attempts']}"
    )


    print(
        f"Tables used: "
        f"{result['tables']}"
    )


    print(
        f"Rows returned: "
        f"{result['row_count']}"
    )


    print(
        f"Database execution: "
        f"{result['elapsed_ms']:.2f} ms"
    )


    print(
        "\nResult:"
    )


    print(
        result[
            "data"
        ].to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "HYBRID NL → SQL COMPLETE"
    )


    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()