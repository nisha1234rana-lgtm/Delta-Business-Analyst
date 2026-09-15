from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re
import time

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


# =========================================================
# CONFIG
# =========================================================

MAX_RESULT_ROWS = 200


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "database"
    / "delta_analytics.duckdb"
)


SQL_AGENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sql_agent"
)


SCHEMA_PATH = (
    SQL_AGENT_DIR
    / "schema_catalog.json"
)


AUDIT_PATH = (
    SQL_AGENT_DIR
    / "sql_runtime_audit.csv"
)


# =========================================================
# ALLOWED RELATIONS
# =========================================================

ALLOWED_RELATIONS = {
    "fact_flights",
    "dim_airport",
    "dim_route",
    "financial_quarterly",
    "vw_monthly_kpis",
    "vw_yearly_kpis",
    "vw_airport_performance",
    "vw_route_performance",
    "vw_operator_performance",
    "vw_delay_causes",
    "vw_financial_performance",
}


# =========================================================
# FORBIDDEN OPERATIONS
# =========================================================

FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "REPLACE",
    "TRUNCATE",
    "MERGE",
    "COPY",
    "EXPORT",
    "IMPORT",
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "CALL",
    "PRAGMA",
    "SET",
    "VACUUM",
]


FORBIDDEN_EXTERNAL_PATTERNS = [
    r"\bread_csv\b",
    r"\bread_csv_auto\b",
    r"\bread_parquet\b",
    r"\bread_json\b",
    r"\bread_json_auto\b",
    r"\bread_text\b",
    r"\bread_blob\b",
    r"\bparquet_scan\b",
    r"\bglob\b",
    r"\bhttpfs\b",
    r"\bsqlite_scan\b",
    r"\bpostgres_scan\b",
    r"\bmysql_scan\b",
    r"\bdelta_scan\b",
    r"\biceberg_scan\b",
]


FORBIDDEN_AST_NODES = {
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Alter",
    "Create",
    "Merge",
    "Copy",
    "Command",
    "Transaction",
    "Commit",
    "Rollback",
}


# =========================================================
# CUSTOM ERROR
# =========================================================

class SafeSQLValidationError(
    Exception
):
    pass


# =========================================================
# VALIDATE FILES
# =========================================================

if not DB_PATH.exists():

    raise FileNotFoundError(
        f"DuckDB warehouse not found:\n"
        f"{DB_PATH}"
    )


if not SCHEMA_PATH.exists():

    raise FileNotFoundError(
        f"Schema catalog not found:\n"
        f"{SCHEMA_PATH}"
    )


# =========================================================
# LOAD SCHEMA CATALOG
# =========================================================

def load_schema_catalog():

    with open(
        SCHEMA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


# =========================================================
# NORMALIZE SQL
# =========================================================

def normalize_sql(
    sql: str,
):

    if not isinstance(
        sql,
        str,
    ):

        raise SafeSQLValidationError(
            "SQL must be a string."
        )


    sql = sql.strip()


    # Remove accidental Markdown fences.
    sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql,
    )


    while sql.endswith(
        ";"
    ):

        sql = (
            sql[:-1]
            .strip()
        )


    if not sql:

        raise SafeSQLValidationError(
            "SQL query is empty."
        )


    return sql


# =========================================================
# FORBIDDEN TEXT VALIDATION
# =========================================================

def validate_forbidden_text(
    sql: str,
):

    upper_sql = (
        sql.upper()
    )


    for keyword in (
        FORBIDDEN_KEYWORDS
    ):

        pattern = (
            rf"\b"
            rf"{re.escape(keyword)}"
            rf"\b"
        )


        if re.search(
            pattern,
            upper_sql,
        ):

            raise SafeSQLValidationError(
                "Forbidden SQL operation "
                f"detected: {keyword}"
            )


    for pattern in (
        FORBIDDEN_EXTERNAL_PATTERNS
    ):

        if re.search(
            pattern,
            sql,
            flags=re.IGNORECASE,
        ):

            raise SafeSQLValidationError(
                "External file or database "
                "access is not permitted."
            )


# =========================================================
# PARSE
# =========================================================

def parse_sql(
    sql: str,
):

    try:

        statements = (
            sqlglot.parse(
                sql,
                read="duckdb",
            )
        )

    except Exception as exc:

        raise SafeSQLValidationError(
            f"SQL parsing failed: {exc}"
        )


    if len(
        statements
    ) != 1:

        raise SafeSQLValidationError(
            "Only one SQL statement "
            "is permitted."
        )


    expression = (
        statements[0]
    )


    if (
        expression.find(
            exp.Select
        )
        is None
    ):

        raise SafeSQLValidationError(
            "Only SELECT queries "
            "are permitted."
        )


    return expression


# =========================================================
# AST SAFETY
# =========================================================

def validate_ast(
    expression,
):

    for node in (
        expression.walk()
    ):

        node_name = (
            node
            .__class__
            .__name__
        )


        if (
            node_name
            in FORBIDDEN_AST_NODES
        ):

            raise SafeSQLValidationError(
                "Forbidden SQL node "
                f"detected: {node_name}"
            )


# =========================================================
# TABLE EXTRACTION
# =========================================================

def extract_tables(
    expression,
):

    cte_names = {

        cte.alias_or_name.lower()

        for cte
        in expression.find_all(
            exp.CTE
        )

        if cte.alias_or_name
    }


    tables = set()


    for table in (
        expression.find_all(
            exp.Table
        )
    ):

        table_name = (
            table.name.lower()
        )


        if (
            table_name
            in cte_names
        ):

            continue


        tables.add(
            table_name
        )


    return sorted(
        tables
    )


# =========================================================
# TABLE WHITELIST
# =========================================================

def validate_tables(
    tables,
):

    unauthorized = [

        table

        for table in tables

        if table
        not in ALLOWED_RELATIONS
    ]


    if unauthorized:

        raise SafeSQLValidationError(
            "Query references "
            "unauthorized relation(s): "
            f"{unauthorized}"
        )


# =========================================================
# DUCKDB BINDER CHECK
# =========================================================

def validate_with_duckdb(
    sql: str,
):

    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True,
    )


    try:

        connection.execute(
            f"EXPLAIN {sql}"
        ).fetchall()

    except Exception as exc:

        raise SafeSQLValidationError(
            "DuckDB validation failed: "
            f"{exc}"
        )

    finally:

        connection.close()


# =========================================================
# COMPLETE VALIDATION
# =========================================================

def validate_sql(
    sql: str,
):

    normalized = normalize_sql(
        sql
    )


    validate_forbidden_text(
        normalized
    )


    expression = parse_sql(
        normalized
    )


    validate_ast(
        expression
    )


    tables = extract_tables(
        expression
    )


    validate_tables(
        tables
    )


    validate_with_duckdb(
        normalized
    )


    return {
        "valid":
            True,

        "sql":
            normalized,

        "tables":
            tables,
    }


# =========================================================
# AUDIT LOG
# =========================================================

def write_audit(
    sql,
    status,
    tables=None,
    rows=None,
    elapsed_ms=None,
    error=None,
):

    SQL_AGENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    record = {
        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            status,

        "tables":
            ", ".join(
                tables or []
            ),

        "rows":
            rows,

        "elapsed_ms":
            elapsed_ms,

        "error":
            error,

        "sql":
            sql,
    }


    file_exists = (
        AUDIT_PATH.exists()
    )


    with open(
        AUDIT_PATH,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=record.keys(),
        )


        if not file_exists:

            writer.writeheader()


        writer.writerow(
            record
        )


# =========================================================
# SAFE EXECUTION
# =========================================================

def execute_safe_sql(
    sql: str,
    max_rows: int = MAX_RESULT_ROWS,
):

    start = (
        time.perf_counter()
    )


    try:

        validation = validate_sql(
            sql
        )


        normalized_sql = (
            validation["sql"]
        )

        tables = (
            validation["tables"]
        )


        protected_sql = f"""
            SELECT *
            FROM (
                {normalized_sql}
            ) AS safe_query
            LIMIT {int(max_rows)}
        """


        connection = duckdb.connect(
            str(DB_PATH),
            read_only=True,
        )


        try:

            result = (
                connection
                .execute(
                    protected_sql
                )
                .df()
            )

        finally:

            connection.close()


        elapsed_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )


        write_audit(
            sql=normalized_sql,
            status="SUCCESS",
            tables=tables,
            rows=len(result),
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
        )


        return {
            "success":
                True,

            "sql":
                normalized_sql,

            "tables":
                tables,

            "row_count":
                len(result),

            "elapsed_ms":
                round(
                    elapsed_ms,
                    2,
                ),

            "data":
                result,
        }


    except Exception as exc:

        elapsed_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000
        )


        write_audit(
            sql=str(sql),
            status="REJECTED",
            elapsed_ms=round(
                elapsed_ms,
                2,
            ),
            error=str(exc),
        )


        return {
            "success":
                False,

            "sql":
                str(sql),

            "tables":
                [],

            "row_count":
                0,

            "elapsed_ms":
                round(
                    elapsed_ms,
                    2,
                ),

            "error":
                str(exc),

            "data":
                None,
        }