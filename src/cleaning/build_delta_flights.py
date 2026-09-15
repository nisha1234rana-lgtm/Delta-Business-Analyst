from pathlib import Path
import polars as pl


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "delta_flights_2026_06.parquet"
)


# =========================================================
# FIND MARKETING CARRIER FILE
# =========================================================

files = list(
    RAW_DIR.glob(
        "On_Time_Marketing_Carrier_On_Time_Performance_*.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "Marketing Carrier CSV not found."
    )

CSV_PATH = files[0]


print("=" * 70)
print("DELTA BUSINESS ANALYST")
print("BUILD CLEAN DELTA FLIGHT DATASET")
print("=" * 70)

print(f"\nSource:\n{CSV_PATH.name}")


# =========================================================
# LOAD RAW FILE
# =========================================================

df = pl.scan_csv(
    CSV_PATH,
    infer_schema_length=10000,
    ignore_errors=True,
)

raw_columns = df.collect_schema().names()


# =========================================================
# CLEAN COLUMN HEADERS
# =========================================================

rename_map = {
    column: column.strip()
    for column in raw_columns
    if column.strip()
    and column.strip() != column
}

if rename_map:

    print("\nCleaning malformed headers:")

    for old, new in rename_map.items():
        print(
            f"  {repr(old)} -> {repr(new)}"
        )

    df = df.rename(rename_map)


# Remove completely blank columns

clean_columns = [
    column
    for column in df.collect_schema().names()
    if column.strip()
]

df = df.select(clean_columns)


# =========================================================
# FILTER DELTA NETWORK
# =========================================================

df = df.filter(
    pl.col("Marketing_Airline_Network")
    == "DL"
)


# =========================================================
# SELECT ANALYTICAL FIELDS
# =========================================================

wanted_columns = [

    # Time
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",

    # Airline
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Tail_Number",
    "Flight_Number_Marketing_Airline",
    "Flight_Number_Operating_Airline",

    # Origin
    "OriginAirportID",
    "Origin",
    "OriginCityName",
    "OriginState",
    "OriginStateName",

    # Destination
    "DestAirportID",
    "Dest",
    "DestCityName",
    "DestState",
    "DestStateName",

    # Departure
    "CRSDepTime",
    "DepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "DepTimeBlk",
    "TaxiOut",
    "WheelsOff",

    # Arrival
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrDel15",
    "ArrivalDelayGroups",
    "ArrTimeBlk",

    # Flight status
    "Cancelled",
    "CancellationCode",
    "Diverted",

    # Flight duration
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",

    # Distance
    "Distance",
    "DistanceGroup",

    # Delay causes
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

available_columns = (
    df.collect_schema().names()
)

selected_columns = [
    column
    for column in wanted_columns
    if column in available_columns
]

missing_columns = [
    column
    for column in wanted_columns
    if column not in available_columns
]

if missing_columns:

    print("\nWARNING: Missing fields:")

    for column in missing_columns:
        print(f"  - {column}")


df = df.select(selected_columns)


# =========================================================
# STANDARDIZE DATA TYPES
# =========================================================

df = df.with_columns(

    # Date
    pl.col("FlightDate")
    .str.strptime(
        pl.Date,
        "%Y-%m-%d",
        strict=False,
    )
    .alias("flight_date"),

    # Integer identifiers
    pl.col("Year")
    .cast(pl.Int16, strict=False)
    .alias("year"),

    pl.col("Quarter")
    .cast(pl.Int8, strict=False)
    .alias("quarter"),

    pl.col("Month")
    .cast(pl.Int8, strict=False)
    .alias("month"),

    pl.col("DayofMonth")
    .cast(pl.Int8, strict=False)
    .alias("day_of_month"),

    pl.col("DayOfWeek")
    .cast(pl.Int8, strict=False)
    .alias("day_of_week"),
)


# =========================================================
# DERIVED ANALYTICAL FEATURES
# =========================================================

df = df.with_columns(

    # ---------------------------------------------
    # Time identifiers
    # ---------------------------------------------

    pl.concat_str(
        [
            pl.col("Year").cast(pl.String),
            pl.col("Month")
            .cast(pl.String)
            .str.pad_start(2, "0"),
        ],
        separator="-",
    ).alias("year_month"),


    # ---------------------------------------------
    # Route
    # ---------------------------------------------

    pl.concat_str(
        [
            pl.col("Origin"),
            pl.col("Dest"),
        ],
        separator="-",
    ).alias("route"),


    # ---------------------------------------------
    # Mainline / partner
    # ---------------------------------------------

    (
        pl.col("Operating_Airline") == "DL"
    ).alias("is_delta_operated"),

    pl.when(
        pl.col("Operating_Airline") == "DL"
    )
    .then(
        pl.lit("Delta Mainline")
    )
    .otherwise(
        pl.lit("Partner")
    )
    .alias("operator_type"),


    # ---------------------------------------------
    # Operational flags
    # ---------------------------------------------

    (
        pl.col("Cancelled") == 1
    ).alias("is_cancelled"),

    (
        pl.col("Diverted") == 1
    ).alias("is_diverted"),

    (
        pl.col("DepDel15") == 1
    ).alias("is_departure_delayed_15"),

    (
        pl.col("ArrDel15") == 1
    ).alias("is_arrival_delayed_15"),


    # ---------------------------------------------
    # On-time arrival
    # ---------------------------------------------

    pl.when(
        (pl.col("Cancelled") == 0)
        & (pl.col("Diverted") == 0)
        & pl.col("ArrDelay").is_not_null()
    )
    .then(
        pl.col("ArrDelay") <= 14
    )
    .otherwise(None)
    .alias("is_on_time_arrival"),
)


# =========================================================
# REMOVE ORIGINAL DUPLICATE DATE COMPONENTS
# =========================================================

df = df.drop(
    [
        "Year",
        "Quarter",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "FlightDate",
    ]
)


# =========================================================
# COLLECT DATA
# =========================================================

print("\nProcessing Delta flights...")

delta_df = df.collect()


# =========================================================
# DATA QUALITY CHECKS
# =========================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)

print(
    f"\nRows: "
    f"{delta_df.height:,}"
)

print(
    f"Columns: "
    f"{delta_df.width:,}"
)


# Duplicate flight check

duplicate_count = (
    delta_df
    .select(
        [
            "flight_date",
            "Marketing_Airline_Network",
            "Flight_Number_Marketing_Airline",
            "Origin",
            "Dest",
            "CRSDepTime",
        ]
    )
    .is_duplicated()
    .sum()
)

print(
    f"Potential duplicate records: "
    f"{duplicate_count:,}"
)


# Missing key values

key_fields = [
    "flight_date",
    "Operating_Airline",
    "Origin",
    "Dest",
]

print("\nMissing key-field values:")

for column in key_fields:

    missing = (
        delta_df[column]
        .null_count()
    )

    print(
        f"  {column}: "
        f"{missing:,}"
    )


# =========================================================
# SAVE PARQUET
# =========================================================

delta_df.write_parquet(
    OUTPUT_PATH,
    compression="zstd",
    statistics=True,
)


# =========================================================
# OUTPUT SUMMARY
# =========================================================

raw_size_mb = (
    CSV_PATH.stat().st_size
    / (1024 * 1024)
)

parquet_size_mb = (
    OUTPUT_PATH.stat().st_size
    / (1024 * 1024)
)

compression_pct = (
    (
        1
        - parquet_size_mb / raw_size_mb
    )
    * 100
)


print("\n" + "=" * 70)
print("PARQUET OUTPUT")
print("=" * 70)

print(
    f"\nRaw CSV size: "
    f"{raw_size_mb:.2f} MB"
)

print(
    f"Processed Parquet size: "
    f"{parquet_size_mb:.2f} MB"
)

print(
    f"Storage reduction: "
    f"{compression_pct:.2f}%"
)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_PATH}"
)


# =========================================================
# BUSINESS SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("DELTA JUNE 2026 SUMMARY")
print("=" * 70)

summary = (
    delta_df
    .select(
        [
            pl.len()
            .alias(
                "flights"
            ),

            pl.col("Origin")
            .n_unique()
            .alias(
                "origin_airports"
            ),

            pl.col("route")
            .n_unique()
            .alias(
                "routes"
            ),

            (
                pl.col(
                    "is_delta_operated"
                )
                .mean()
                * 100
            )
            .round(2)
            .alias(
                "delta_operated_pct"
            ),

            (
                pl.col(
                    "is_cancelled"
                )
                .mean()
                * 100
            )
            .round(2)
            .alias(
                "cancellation_rate_pct"
            ),

            (
                pl.col(
                    "is_on_time_arrival"
                )
                .mean()
                * 100
            )
            .round(2)
            .alias(
                "on_time_arrival_pct"
            ),

            pl.col("ArrDelay")
            .mean()
            .round(2)
            .alias(
                "avg_arrival_delay_min"
            ),
        ]
    )
)

print(summary)

print("\n" + "=" * 70)
print("CLEANING PIPELINE COMPLETE")
print("=" * 70)