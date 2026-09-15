from pathlib import Path
from datetime import datetime
import json

import duckdb
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec"
    / "delta_companyfacts.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "financial"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUTPUT_DIR
    / "delta_financial_quarterly_2020_2026.csv"
)

PARQUET_PATH = (
    OUTPUT_DIR
    / "delta_financial_quarterly_2020_2026.parquet"
)

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "database"
    / "delta_analytics.duckdb"
)


START_YEAR = 2020
END_YEAR = 2026


# =========================================================
# LOAD SEC COMPANY FACTS
# =========================================================

if not SEC_PATH.exists():
    raise FileNotFoundError(
        f"SEC Company Facts file not found:\n{SEC_PATH}"
    )

with open(
    SEC_PATH,
    "r",
    encoding="utf-8",
) as file:
    company_facts = json.load(file)


US_GAAP = (
    company_facts
    .get("facts", {})
    .get("us-gaap", {})
)


print("=" * 76)
print("DELTA BUSINESS ANALYST")
print("BUILD QUARTERLY FINANCIAL DATASET")
print("=" * 76)

print(
    f"\nCompany: "
    f"{company_facts.get('entityName')}"
)


# =========================================================
# CONCEPT CONFIGURATION
# =========================================================

CONCEPTS = {

    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ],

    "operating_income": [
        "OperatingIncomeLoss",
    ],

    "net_income": [
        "NetIncomeLoss",
    ],

    "operating_expense": [
        "OperatingExpenses",
    ],

    "fuel_cost": [
        "FuelCosts",
    ],

    "interest_expense": [
        "InterestExpense",
    ],

    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
    ],

    "assets": [
        "Assets",
    ],

    "debt_combined": [
        "DebtLongtermAndShorttermCombinedAmount",
    ],

    "debt_current": [
        "LongTermDebtCurrent",
    ],

    "debt_noncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
}


# =========================================================
# HELPERS
# =========================================================

def parse_date(value):
    if not value:
        return None

    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def quarter_from_end_date(end_date):

    if not end_date:
        return None

    month = end_date.month

    if month <= 3:
        return 1

    if month <= 6:
        return 2

    if month <= 9:
        return 3

    return 4


def get_concept_facts(
    concept_name,
    unit="USD",
):

    concept = US_GAAP.get(
        concept_name
    )

    if not concept:
        return []

    units = (
        concept.get("units")
        or {}
    )

    return (
        units.get(unit)
        or []
    )


def clean_facts(
    concept_name,
    unit="USD",
):

    rows = []

    for fact in get_concept_facts(
        concept_name,
        unit,
    ):

        form = fact.get("form")

        if form not in {
            "10-Q",
            "10-K",
        }:
            continue

        end = parse_date(
            fact.get("end")
        )

        start = parse_date(
            fact.get("start")
        )

        if not end:
            continue

        if not (
            START_YEAR
            <= end.year
            <= END_YEAR
        ):
            continue

        filed = parse_date(
            fact.get("filed")
        )

        duration_days = None

        if start:
            duration_days = (
                end - start
            ).days + 1

        rows.append(
            {
                "concept": concept_name,
                "start": start,
                "end": end,
                "filed": filed,
                "form": form,
                "value": fact.get("val"),
                "duration_days": duration_days,
                "fy": fact.get("fy"),
                "fp": fact.get("fp"),
                "frame": fact.get("frame"),
                "accn": fact.get("accn"),
            }
        )

    return rows


# =========================================================
# DURATION FACT EXTRACTION
# =========================================================

def find_direct_quarter(
    concept_name,
    year,
    quarter,
):

    rows = clean_facts(
        concept_name
    )

    candidates = []

    for row in rows:

        end = row["end"]
        duration = row["duration_days"]

        if not duration:
            continue

        if end.year != year:
            continue

        if (
            quarter_from_end_date(end)
            != quarter
        ):
            continue

        # Standalone quarter is normally ~90 days.
        if not (
            70
            <= duration
            <= 110
        ):
            continue

        candidates.append(
            row
        )

    if not candidates:
        return None

    # Later filing wins in case of restatement.
    candidates.sort(
        key=lambda x: (
            x["filed"]
            or datetime.min.date()
        )
    )

    return candidates[-1]["value"]


def find_ytd(
    concept_name,
    year,
    quarter,
):

    rows = clean_facts(
        concept_name
    )

    target_ranges = {
        1: (70, 110),
        2: (160, 205),
        3: (250, 295),
        4: (330, 380),
    }

    low, high = target_ranges[
        quarter
    ]

    candidates = []

    for row in rows:

        end = row["end"]
        duration = row["duration_days"]

        if not duration:
            continue

        if end.year != year:
            continue

        if (
            quarter_from_end_date(end)
            != quarter
        ):
            continue

        if not (
            low
            <= duration
            <= high
        ):
            continue

        candidates.append(
            row
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["filed"]
            or datetime.min.date()
        )
    )

    return candidates[-1]["value"]


def duration_quarter_value(
    concept_names,
    year,
    quarter,
    known_quarters,
):

    for concept_name in concept_names:

        # First try an actual standalone
        # three-month fact.
        direct = find_direct_quarter(
            concept_name,
            year,
            quarter,
        )

        if direct is not None:
            return direct, concept_name


        # Q4 normally must be derived
        # from FY minus Q1-Q3.
        ytd = find_ytd(
            concept_name,
            year,
            quarter,
        )

        if ytd is None:
            continue


        if quarter == 1:
            return ytd, concept_name


        previous_values = [
            known_quarters.get(q)
            for q in range(
                1,
                quarter,
            )
        ]

        if all(
            value is not None
            for value in previous_values
        ):

            discrete = (
                ytd
                - sum(previous_values)
            )

            return (
                discrete,
                concept_name,
            )

    return None, None


# =========================================================
# INSTANT FACT EXTRACTION
# =========================================================

def instant_value(
    concept_names,
    year,
    quarter,
):

    target_month = {
        1: 3,
        2: 6,
        3: 9,
        4: 12,
    }[quarter]

    for concept_name in concept_names:

        rows = clean_facts(
            concept_name
        )

        candidates = []

        for row in rows:

            end = row["end"]

            # Instant facts have no start date.
            if row["start"] is not None:
                continue

            if end.year != year:
                continue

            if end.month != target_month:
                continue

            candidates.append(
                row
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (
                x["filed"]
                or datetime.min.date()
            )
        )

        return (
            candidates[-1]["value"],
            concept_name,
        )

    return None, None


# =========================================================
# BUILD QUARTERLY DATASET
# =========================================================

duration_metrics = [
    "revenue",
    "operating_income",
    "net_income",
    "operating_expense",
    "fuel_cost",
    "interest_expense",
]


rows = []

quarter_cache = {
    metric: {}
    for metric in duration_metrics
}


for year in range(
    START_YEAR,
    END_YEAR + 1,
):

    for quarter in range(
        1,
        5,
    ):

        # We only have filings through Q2 2026.
        if (
            year == 2026
            and quarter > 2
        ):
            continue

        row = {
            "year": year,
            "quarter": quarter,
            "period": (
                f"{year}-Q{quarter}"
            ),
        }


        # ---------------------------------------------
        # DURATION METRICS
        # ---------------------------------------------

        for metric in duration_metrics:

            known = {
                q: quarter_cache[
                    metric
                ].get(
                    (year, q)
                )
                for q in range(
                    1,
                    quarter,
                )
            }

            value, concept = (
                duration_quarter_value(
                    CONCEPTS[metric],
                    year,
                    quarter,
                    known,
                )
            )

            quarter_cache[
                metric
            ][
                (year, quarter)
            ] = value

            row[metric] = value

            row[
                f"{metric}_concept"
            ] = concept


        # ---------------------------------------------
        # BALANCE SHEET
        # ---------------------------------------------

        cash, _ = instant_value(
            CONCEPTS["cash"],
            year,
            quarter,
        )

        assets, _ = instant_value(
            CONCEPTS["assets"],
            year,
            quarter,
        )

        combined_debt, _ = (
            instant_value(
                CONCEPTS[
                    "debt_combined"
                ],
                year,
                quarter,
            )
        )

        current_debt, _ = (
            instant_value(
                CONCEPTS[
                    "debt_current"
                ],
                year,
                quarter,
            )
        )

        noncurrent_debt, _ = (
            instant_value(
                CONCEPTS[
                    "debt_noncurrent"
                ],
                year,
                quarter,
            )
        )


        row["cash"] = cash
        row["assets"] = assets

        row[
            "long_term_debt_current"
        ] = current_debt

        row[
            "long_term_debt_noncurrent"
        ] = noncurrent_debt


        if combined_debt is not None:

            row[
                "total_debt"
            ] = combined_debt

        elif (
            current_debt is not None
            and noncurrent_debt is not None
        ):

            row[
                "total_debt"
            ] = (
                current_debt
                + noncurrent_debt
            )

        else:

            row[
                "total_debt"
            ] = None


        rows.append(
            row
        )


df = pd.DataFrame(
    rows
)


# =========================================================
# CONVERT USD TO MILLIONS
# =========================================================

money_columns = [
    "revenue",
    "operating_income",
    "net_income",
    "operating_expense",
    "fuel_cost",
    "interest_expense",
    "cash",
    "assets",
    "long_term_debt_current",
    "long_term_debt_noncurrent",
    "total_debt",
]


for column in money_columns:

    if column in df.columns:

        df[
            f"{column}_m"
        ] = (
            df[column]
            / 1_000_000
        )


# =========================================================
# DERIVED FINANCIAL KPIs
# =========================================================

df[
    "operating_margin_pct"
] = (
    df["operating_income"]
    / df["revenue"]
    * 100
)

df[
    "net_margin_pct"
] = (
    df["net_income"]
    / df["revenue"]
    * 100
)

df[
    "fuel_cost_pct_revenue"
] = (
    df["fuel_cost"]
    / df["revenue"]
    * 100
)

df[
    "revenue_yoy_pct"
] = (
    df.groupby(
        "quarter"
    )["revenue"]
    .pct_change()
    * 100
)

df[
    "operating_income_yoy_pct"
] = (
    df.groupby(
        "quarter"
    )["operating_income"]
    .pct_change()
    * 100
)


# =========================================================
# ROUND DISPLAY METRICS
# =========================================================

percentage_columns = [
    "operating_margin_pct",
    "net_margin_pct",
    "fuel_cost_pct_revenue",
    "revenue_yoy_pct",
    "operating_income_yoy_pct",
]


for column in percentage_columns:

    df[column] = (
        df[column]
        .round(2)
    )


# =========================================================
# DATA QUALITY
# =========================================================

print("\n" + "=" * 76)
print("DATA QUALITY")
print("=" * 76)


print(
    f"\nQuarterly periods: "
    f"{len(df)}"
)

print(
    f"Period range: "
    f"{df['period'].iloc[0]} "
    f"→ "
    f"{df['period'].iloc[-1]}"
)


print(
    "\nMissing values by core metric:"
)

for column in [
    "revenue",
    "operating_income",
    "net_income",
    "operating_expense",
    "fuel_cost",
    "cash",
    "assets",
    "total_debt",
]:

    print(
        f"  {column:<25} "
        f"{df[column].isna().sum()}"
    )


# =========================================================
# SAVE FILES
# =========================================================

df.to_csv(
    CSV_PATH,
    index=False,
)

df.to_parquet(
    PARQUET_PATH,
    index=False,
)


# =========================================================
# LOAD INTO DUCKDB
# =========================================================

print(
    "\nLoading financial data "
    "into DuckDB..."
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
        operating_income_m,
        net_income_m,
        operating_expense_m,
        fuel_cost_m,
        interest_expense_m,

        operating_margin_pct,
        net_margin_pct,
        fuel_cost_pct_revenue,

        revenue_yoy_pct,
        operating_income_yoy_pct,

        cash_m,
        assets_m,
        total_debt_m

    FROM financial_quarterly;
""")


con.close()


# =========================================================
# DISPLAY RESULTS
# =========================================================

display_columns = [
    "period",
    "revenue_m",
    "operating_income_m",
    "net_income_m",
    "operating_margin_pct",
    "net_margin_pct",
    "fuel_cost_m",
    "cash_m",
    "total_debt_m",
]


print("\n" + "=" * 76)
print("DELTA QUARTERLY FINANCIAL PERFORMANCE")
print("=" * 76)


print(
    df[
        display_columns
    ]
    .to_string(
        index=False,
    )
)


# =========================================================
# RECENT PERFORMANCE
# =========================================================

print("\n" + "=" * 76)
print("RECENT QUARTERS")
print("=" * 76)


print(
    df[
        [
            "period",
            "revenue_m",
            "revenue_yoy_pct",
            "operating_income_m",
            "operating_margin_pct",
            "net_income_m",
            "fuel_cost_m",
            "total_debt_m",
        ]
    ]
    .tail(10)
    .to_string(
        index=False,
    )
)


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 76)
print("FINANCIAL DATASET BUILD COMPLETE")
print("=" * 76)

print(
    f"\nCSV:\n{CSV_PATH}"
)

print(
    f"\nParquet:\n{PARQUET_PATH}"
)

print(
    f"\nDuckDB table:"
    f"\nfinancial_quarterly"
)