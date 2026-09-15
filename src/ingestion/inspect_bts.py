from pathlib import Path

import polars as pl


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BTS_DIR = PROJECT_ROOT / "data" / "raw" / "bts"

CSV_FILES = list(RAW_BTS_DIR.glob("*.csv"))

if not CSV_FILES:
    raise FileNotFoundError(
        f"No BTS CSV files found in {RAW_BTS_DIR}"
    )

# For now we only downloaded one month
CSV_PATH = CSV_FILES[0]


# ---------------------------------------------------------
# LOAD DATA LAZILY
# ---------------------------------------------------------

print("=" * 70)
print("DELTA BUSINESS ANALYST")
print("RAW BTS DATA INSPECTION")
print("=" * 70)

print(f"\nFile:")
print(CSV_PATH.name)

print("\nScanning dataset...")

df = pl.scan_csv(
    CSV_PATH,
    infer_schema_length=10000,
    ignore_errors=True,
)


# ---------------------------------------------------------
# DATASET SIZE
# ---------------------------------------------------------

row_count = (
    df.select(pl.len().alias("rows"))
    .collect()
    .item()
)

columns = df.collect_schema().names()

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(f"\nRows:    {row_count:,}")
print(f"Columns: {len(columns):,}")


# ---------------------------------------------------------
# COLUMN LIST
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RAW COLUMNS")
print("=" * 70)

for number, column in enumerate(columns, start=1):
    print(f"{number:>3}. {column}")


# ---------------------------------------------------------
# FIND AIRLINE-RELATED COLUMNS
# ---------------------------------------------------------

airline_keywords = [
    "airline",
    "carrier",
    "marketing",
    "operating",
]

airline_columns = [
    column
    for column in columns
    if any(
        keyword in column.lower()
        for keyword in airline_keywords
    )
]

print("\n" + "=" * 70)
print("AIRLINE / CARRIER COLUMNS")
print("=" * 70)

for column in airline_columns:
    print(column)


# ---------------------------------------------------------
# REPORTING AIRLINE DISTRIBUTION
# ---------------------------------------------------------

if "Reporting_Airline" in columns:

    print("\n" + "=" * 70)
    print("REPORTING AIRLINES")
    print("=" * 70)

    airline_counts = (
        df.group_by("Reporting_Airline")
        .agg(
            pl.len().alias("flights")
        )
        .sort(
            "flights",
            descending=True,
        )
        .collect()
    )

    print(airline_counts)


# ---------------------------------------------------------
# DELTA MAINLINE
# ---------------------------------------------------------

if "Reporting_Airline" in columns:

    delta_mainline = (
        df.filter(
            pl.col("Reporting_Airline") == "DL"
        )
        .select(
            pl.len().alias("flights")
        )
        .collect()
        .item()
    )

    print("\n" + "=" * 70)
    print("DELTA MAINLINE")
    print("=" * 70)

    print(
        f"\nFlights where Reporting_Airline = DL: "
        f"{delta_mainline:,}"
    )

    delta_share = (
        delta_mainline / row_count * 100
        if row_count
        else 0
    )

    print(
        f"Share of monthly dataset: "
        f"{delta_share:.2f}%"
    )


# ---------------------------------------------------------
# DELTA MARKETING CARRIER
# ---------------------------------------------------------

marketing_column = None

possible_marketing_columns = [
    "IATA_Code_Marketing_Airline",
    "Marketing_Airline_Network",
]

for candidate in possible_marketing_columns:

    if candidate in columns:
        marketing_column = candidate
        break


if marketing_column:

    delta_marketed = (
        df.filter(
            pl.col(marketing_column) == "DL"
        )
        .select(
            pl.len().alias("flights")
        )
        .collect()
        .item()
    )

    print("\n" + "=" * 70)
    print("DELTA-MARKETED NETWORK")
    print("=" * 70)

    print(f"\nColumn used: {marketing_column}")

    print(
        f"Flights marketed by Delta: "
        f"{delta_marketed:,}"
    )

    if "Reporting_Airline" in columns:

        partner_flights = (
            delta_marketed - delta_mainline
        )

        print(
            f"Delta-marketed flights not reported "
            f"as DL mainline: {partner_flights:,}"
        )


# ---------------------------------------------------------
# SAMPLE RECORDS
# ---------------------------------------------------------

important_candidates = [
    "FlightDate",
    "Reporting_Airline",
    "IATA_Code_Marketing_Airline",
    "Tail_Number",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "OriginCityName",
    "Dest",
    "DestCityName",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "Distance",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

sample_columns = [
    column
    for column in important_candidates
    if column in columns
]

print("\n" + "=" * 70)
print("SAMPLE FLIGHT RECORDS")
print("=" * 70)

sample = (
    df.select(sample_columns)
    .head(10)
    .collect()
)

print(sample)


# ---------------------------------------------------------
# DELTA SAMPLE
# ---------------------------------------------------------

if "Reporting_Airline" in columns:

    print("\n" + "=" * 70)
    print("DELTA SAMPLE RECORDS")
    print("=" * 70)

    delta_sample = (
        df.filter(
            pl.col("Reporting_Airline") == "DL"
        )
        .select(sample_columns)
        .head(10)
        .collect()
    )

    print(delta_sample)


print("\n" + "=" * 70)
print("RAW DATA INSPECTION COMPLETE")
print("=" * 70)