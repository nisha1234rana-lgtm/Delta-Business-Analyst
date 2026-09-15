from pathlib import Path

import duckdb
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINANCIAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial"
)

CSV_PATH = (
    FINANCIAL_DIR
    / "delta_financial_quarterly_2020_2026.csv"
)

PARQUET_PATH = (
    FINANCIAL_DIR
    / "delta_financial_quarterly_2020_2026.parquet"
)

QUALITY_PATH = (
    FINANCIAL_DIR
    / "financial_quality_report.csv"
)

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "database"
    / "delta_analytics.duckdb"
)


# =========================================================
# LOAD
# =========================================================

print("=" * 76)
print("DELTA BUSINESS ANALYST")
print("FINALIZE FINANCIAL DATA")
print("=" * 76)

df = pd.read_parquet(
    PARQUET_PATH
)

print(
    f"\nLoaded periods: {len(df)}"
)

print(
    f"Range: "
    f"{df['period'].iloc[0]} "
    f"→ {df['period'].iloc[-1]}"
)


# =========================================================
# FIX OPERATING EXPENSE
# =========================================================

missing_before = (
    df["operating_expense"]
    .isna()
    .sum()
)

print(
    f"\nMissing operating expense "
    f"before repair: {missing_before}"
)


df["operating_expense_source"] = (
    "SEC XBRL"
)


missing_mask = (
    df["operating_expense"]
    .isna()
)


df.loc[
    missing_mask,
    "operating_expense"
] = (
    df.loc[
        missing_mask,
        "revenue"
    ]
    -
    df.loc[
        missing_mask,
        "operating_income"
    ]
)


df.loc[
    missing_mask,
    "operating_expense_source"
] = (
    "Derived: revenue - operating income"
)


df["operating_expense_m"] = (
    df["operating_expense"]
    / 1_000_000
)


# =========================================================
# ACCOUNTING IDENTITY VALIDATION
# =========================================================

df[
    "operating_identity_difference"
] = (
    df["revenue"]
    - df["operating_expense"]
    - df["operating_income"]
)


df[
    "operating_identity_valid"
] = (
    df[
        "operating_identity_difference"
    ].abs()
    < 1
)


# =========================================================
# CORE DATA QUALITY
# =========================================================

core_metrics = [
    "revenue",
    "operating_income",
    "operating_expense",
    "net_income",
    "fuel_cost",
    "cash",
    "assets",
    "total_debt",
]


quality_rows = []


for column in core_metrics:

    quality_rows.append(
        {
            "metric": column,
            "missing_values": int(
                df[column]
                .isna()
                .sum()
            ),
            "available_values": int(
                df[column]
                .notna()
                .sum()
            ),
            "periods": len(df),
        }
    )


quality_df = pd.DataFrame(
    quality_rows
)


# =========================================================
# VALIDATE SPECIFIC RECENT QUARTER
# =========================================================

q2_2026 = df[
    df["period"] == "2026-Q2"
].iloc[0]


print("\n" + "=" * 76)
print("Q2 2026 VALIDATION")
print("=" * 76)

print(
    f"\nRevenue:            "
    f"${q2_2026['revenue_m']:,.0f}M"
)

print(
    f"Operating expense:  "
    f"${q2_2026['operating_expense_m']:,.0f}M"
)

print(
    f"Operating income:   "
    f"${q2_2026['operating_income_m']:,.0f}M"
)

print(
    f"Net income:         "
    f"${q2_2026['net_income_m']:,.0f}M"
)

print(
    f"Operating margin:   "
    f"{q2_2026['operating_margin_pct']:.2f}%"
)


# =========================================================
# OVERALL VALIDATION
# =========================================================

invalid_identity = (
    (~df["operating_identity_valid"])
    .sum()
)


print("\n" + "=" * 76)
print("DATA QUALITY RESULTS")
print("=" * 76)

print(
    f"\nMissing operating expense "
    f"after repair: "
    f"{df['operating_expense'].isna().sum()}"
)

print(
    f"Accounting identity failures: "
    f"{invalid_identity}"
)

print("\nCore metric completeness:")

print(
    quality_df.to_string(
        index=False
    )
)


# =========================================================
# SAVE FINAL DATA
# =========================================================

df.to_csv(
    CSV_PATH,
    index=False,
)

df.to_parquet(
    PARQUET_PATH,
    index=False,
)

quality_df.to_csv(
    QUALITY_PATH,
    index=False,
)


# =========================================================
# RELOAD DUCKDB
# =========================================================

print(
    "\nUpdating DuckDB "
    "financial table..."
)

con = duckdb.connect(
    str(DB_PATH)
)

con.register(
    "financial_df",
    df,
)


con.execute("""
    CREATE OR REPLACE TABLE
        financial_quarterly
    AS

    SELECT *
    FROM financial_df;
""")


con.execute("""
    CREATE OR REPLACE VIEW
        vw_financial_performance
    AS

    SELECT
        year,
        quarter,
        period,

        revenue_m,
        operating_expense_m,
        operating_income_m,
        net_income_m,
        fuel_cost_m,
        interest_expense_m,

        operating_margin_pct,
        net_margin_pct,
        fuel_cost_pct_revenue,

        revenue_yoy_pct,
        operating_income_yoy_pct,

        cash_m,
        assets_m,
        total_debt_m,

        operating_expense_source

    FROM financial_quarterly;
""")


con.close()


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 76)
print("FINANCIAL DATA FINALIZED")
print("=" * 76)

print(
    f"\nFinal periods: "
    f"{len(df)}"
)

print(
    f"Core financial metrics complete: "
    f"{quality_df['missing_values'].sum() == 0}"
)

print(
    f"\nQuality report:\n"
    f"{QUALITY_PATH}"
)

print(
    f"\nDuckDB updated:\n"
    f"{DB_PATH}"
)