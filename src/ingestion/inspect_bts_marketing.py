from pathlib import Path

import polars as pl


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_BTS_DIR = PROJECT_ROOT / "data" / "raw" / "bts"

MARKETING_FILES = list(
    RAW_BTS_DIR.glob(
        "On_Time_Marketing_Carrier_On_Time_Performance_*.csv"
    )
)

if not MARKETING_FILES:
    raise FileNotFoundError(
        "No BTS Marketing Carrier CSV found."
    )

CSV_PATH = MARKETING_FILES[0]


# ---------------------------------------------------------
# LOAD RAW DATA
# ---------------------------------------------------------

print("=" * 70)
print("DELTA BUSINESS ANALYST")
print("MARKETING CARRIER DATA INSPECTION")
print("=" * 70)

print(f"\nFile:")
print(CSV_PATH.name)

df = pl.scan_csv(
    CSV_PATH,
    infer_schema_length=10000,
    ignore_errors=True,
)


# ---------------------------------------------------------
# CLEAN RAW COLUMN NAMES
# ---------------------------------------------------------

raw_columns = df.collect_schema().names()

rename_map = {
    column: column.strip()
    for column in raw_columns
    if column.strip() != column
    and column.strip() != ""
}

if rename_map:

    print("\nCleaning malformed BTS column names...")

    for old, new in rename_map.items():
        print(f"  {repr(old)} -> {repr(new)}")

    df = df.rename(rename_map)


# Remove completely blank columns
columns_after_rename = df.collect_schema().names()

valid_columns = [
    column
    for column in columns_after_rename
    if column.strip() != ""
]

df = df.select(valid_columns)

columns = df.collect_schema().names()


# ---------------------------------------------------------
# DATASET SIZE
# ---------------------------------------------------------

row_count = (
    df.select(
        pl.len().alias("rows")
    )
    .collect()
    .item()
)

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print(f"\nRows:    {row_count:,}")
print(f"Columns after header cleanup: {len(columns):,}")


# ---------------------------------------------------------
# KEY AIRLINE FIELDS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("KEY AIRLINE FIELDS")
print("=" * 70)

key_airline_columns = [
    "Marketing_Airline_Network",
    "Operated_or_Branded_Code_Share_Partners",
    "DOT_ID_Marketing_Airline",
    "IATA_Code_Marketing_Airline",
    "Flight_Number_Marketing_Airline",
    "Originally_Scheduled_Code_Share_Airline",
    "DOT_ID_Originally_Scheduled_Code_Share_Airline",
    "IATA_Code_Originally_Scheduled_Code_Share_Airline",
    "Flight_Num_Originally_Scheduled_Code_Share_Airline",
    "Operating_Airline",
    "DOT_ID_Operating_Airline",
    "IATA_Code_Operating_Airline",
    "Flight_Number_Operating_Airline",
]

for column in key_airline_columns:

    if column in columns:
        print(column)


# ---------------------------------------------------------
# DELTA-MARKETED NETWORK
# ---------------------------------------------------------

delta = df.filter(
    pl.col("Marketing_Airline_Network") == "DL"
)

delta_count = (
    delta
    .select(
        pl.len().alias("flights")
    )
    .collect()
    .item()
)

print("\n" + "=" * 70)
print("DELTA-MARKETED NETWORK")
print("=" * 70)

print(
    f"\nDelta-marketed flights: "
    f"{delta_count:,}"
)

print(
    f"Share of all flights: "
    f"{delta_count / row_count * 100:.2f}%"
)


# ---------------------------------------------------------
# ACTUAL OPERATING CARRIERS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("WHO ACTUALLY OPERATED DELTA-MARKETED FLIGHTS?")
print("=" * 70)

operator_counts = (
    delta
    .group_by(
        "Operating_Airline"
    )
    .agg(
        pl.len().alias("flights")
    )
    .with_columns(
        (
            pl.col("flights")
            / delta_count
            * 100
        )
        .round(2)
        .alias("network_share_pct")
    )
    .sort(
        "flights",
        descending=True,
    )
    .collect()
)

print(operator_counts)


# ---------------------------------------------------------
# MAINLINE VS PARTNER
# ---------------------------------------------------------

delta_mainline = (
    delta
    .filter(
        pl.col("Operating_Airline") == "DL"
    )
    .select(
        pl.len().alias("flights")
    )
    .collect()
    .item()
)

delta_partner = (
    delta_count
    - delta_mainline
)


print("\n" + "=" * 70)
print("DELTA MAINLINE VS PARTNER OPERATIONS")
print("=" * 70)

print(
    f"\nDelta-operated flights: "
    f"{delta_mainline:,}"
)

print(
    f"Partner-operated flights: "
    f"{delta_partner:,}"
)

print(
    f"\nDelta-operated share: "
    f"{delta_mainline / delta_count * 100:.2f}%"
)

print(
    f"Partner-operated share: "
    f"{delta_partner / delta_count * 100:.2f}%"
)


# ---------------------------------------------------------
# DELTA NETWORK COVERAGE
# ---------------------------------------------------------

delta_summary = (
    delta
    .select(
        [
            pl.col("Origin")
            .n_unique()
            .alias(
                "unique_origin_airports"
            ),

            pl.col("Dest")
            .n_unique()
            .alias(
                "unique_destination_airports"
            ),

            pl.struct(
                [
                    "Origin",
                    "Dest",
                ]
            )
            .n_unique()
            .alias(
                "unique_directional_routes"
            ),

            pl.col("Tail_Number")
            .n_unique()
            .alias(
                "unique_aircraft"
            ),
        ]
    )
    .collect()
)

print("\n" + "=" * 70)
print("DELTA NETWORK COVERAGE")
print("=" * 70)

print(delta_summary)


# ---------------------------------------------------------
# TOP ORIGIN AIRPORTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP DELTA ORIGIN AIRPORTS")
print("=" * 70)

top_origins = (
    delta
    .group_by(
        [
            "Origin",
            "OriginCityName",
        ]
    )
    .agg(
        pl.len().alias("flights")
    )
    .sort(
        "flights",
        descending=True,
    )
    .head(15)
    .collect()
)

print(top_origins)


# ---------------------------------------------------------
# TOP ROUTES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP DELTA ROUTES")
print("=" * 70)

top_routes = (
    delta
    .group_by(
        [
            "Origin",
            "Dest",
        ]
    )
    .agg(
        pl.len().alias("flights")
    )
    .sort(
        "flights",
        descending=True,
    )
    .head(15)
    .collect()
)

print(top_routes)


# ---------------------------------------------------------
# OPERATIONAL PERFORMANCE SNAPSHOT
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DELTA OPERATIONAL SNAPSHOT")
print("=" * 70)

operational_snapshot = (
    delta
    .select(
        [
            pl.len().alias(
                "scheduled_flights"
            ),

            pl.col("Cancelled")
            .sum()
            .alias(
                "cancelled_flights"
            ),

            (
                pl.col("Cancelled").mean()
                * 100
            )
            .round(2)
            .alias(
                "cancellation_rate_pct"
            ),

            pl.col("Diverted")
            .sum()
            .alias(
                "diverted_flights"
            ),

            pl.col("DepDelay")
            .mean()
            .round(2)
            .alias(
                "avg_departure_delay_min"
            ),

            pl.col("ArrDelay")
            .mean()
            .round(2)
            .alias(
                "avg_arrival_delay_min"
            ),

            (
                pl.col("ArrDel15")
                .mean()
                * 100
            )
            .round(2)
            .alias(
                "arrival_delay_15plus_pct"
            ),
        ]
    )
    .collect()
)

print(operational_snapshot)


# ---------------------------------------------------------
# SAMPLE DELTA FLIGHTS
# ---------------------------------------------------------

sample_columns = [
    "FlightDate",
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Tail_Number",
    "Flight_Number_Marketing_Airline",
    "Origin",
    "OriginCityName",
    "Dest",
    "DestCityName",
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "ArrDelay",
    "Cancelled",
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
    for column in sample_columns
    if column in columns
]

print("\n" + "=" * 70)
print("DELTA SAMPLE RECORDS")
print("=" * 70)

sample = (
    delta
    .select(
        sample_columns
    )
    .head(15)
    .collect()
)

print(sample)


print("\n" + "=" * 70)
print("MARKETING DATA INSPECTION COMPLETE")
print("=" * 70)