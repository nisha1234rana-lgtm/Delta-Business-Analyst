from pathlib import Path
from datetime import date
import time
import zipfile

import polars as pl
import requests


# =========================================================
# CONFIGURATION
# =========================================================

START_YEAR = 2020
START_MONTH = 1

END_YEAR = 2026
END_MONTH = 6

KEEP_ZIP_FILES = True
KEEP_EXTRACTED_CSV = False

REQUEST_TIMEOUT = 180
MAX_RETRIES = 3


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bts"
    / "marketing"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delta_flights"
    / "monthly"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# BTS DATASET INFORMATION
# =========================================================

DATASET_NAME = (
    "On_Time_Marketing_Carrier_"
    "On_Time_Performance_"
    "Beginning_January_2018"
)

BASE_URL = (
    "https://transtats.bts.gov/PREZIP"
)


# =========================================================
# ANALYTICAL COLUMNS
# =========================================================

WANTED_COLUMNS = [

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

    # Duration
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


REQUIRED_COLUMNS = [
    "Year",
    "Quarter",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "FlightDate",
    "Marketing_Airline_Network",
    "Operating_Airline",
    "Origin",
    "Dest",
    "Cancelled",
    "Diverted",
    "DepDel15",
    "ArrDel15",
    "ArrDelay",
]


# =========================================================
# MONTH GENERATOR
# =========================================================

def month_range(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
):

    year = start_year
    month = start_month

    while (
        year < end_year
        or (
            year == end_year
            and month <= end_month
        )
    ):

        yield year, month

        month += 1

        if month == 13:
            month = 1
            year += 1


# =========================================================
# DOWNLOAD
# =========================================================

def download_zip(
    year: int,
    month: int,
) -> Path:

    filename = (
        f"{DATASET_NAME}_{year}_{month}.zip"
    )

    url = (
        f"{BASE_URL}/{filename}"
    )

    destination = (
        RAW_DIR / filename
    )

    if destination.exists():

        size_mb = (
            destination.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"ZIP already exists "
            f"({size_mb:.1f} MB)"
        )

        return destination


    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    }


    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            print(
                f"Downloading official BTS file..."
            )

            with requests.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers=headers,
            ) as response:

                response.raise_for_status()

                total_bytes = 0

                with open(
                    destination,
                    "wb",
                ) as file:

                    for chunk in (
                        response.iter_content(
                            chunk_size=1024 * 1024
                        )
                    ):

                        if not chunk:
                            continue

                        file.write(chunk)

                        total_bytes += len(chunk)

                        downloaded_mb = (
                            total_bytes
                            / (1024 * 1024)
                        )

                        print(
                            f"\rDownloaded: "
                            f"{downloaded_mb:.1f} MB",
                            end="",
                        )

            print()

            return destination


        except Exception as exc:

            print(
                f"\nDownload attempt "
                f"{attempt} failed:"
            )

            print(exc)

            if destination.exists():
                destination.unlink()

            if attempt == MAX_RETRIES:
                raise

            time.sleep(2)


# =========================================================
# EXTRACT
# =========================================================

def extract_csv(
    zip_path: Path,
) -> Path:

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        csv_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:

            raise RuntimeError(
                "No CSV found in BTS ZIP."
            )

        csv_name = csv_files[0]

        csv_path = (
            RAW_DIR
            / Path(csv_name).name
        )

        if csv_path.exists():

            print(
                "Extracted CSV already exists."
            )

            return csv_path


        print("Extracting CSV...")

        archive.extract(
            csv_name,
            RAW_DIR,
        )


        original_path = (
            RAW_DIR / csv_name
        )


        # Handle nested ZIP paths if BTS ever uses them
        if original_path != csv_path:

            original_path.replace(
                csv_path
            )


    return csv_path


# =========================================================
# CLEAN MONTH
# =========================================================

def process_month(
    csv_path: Path,
    year: int,
    month: int,
) -> dict:

    output_path = (
        PROCESSED_DIR
        / f"delta_flights_{year}_{month:02d}.parquet"
    )


    if output_path.exists():

        existing = pl.read_parquet(
            output_path
        )

        print(
            f"Processed file already exists: "
            f"{existing.height:,} Delta flights"
        )

        return {
            "year": year,
            "month": month,
            "rows": existing.height,
            "status": "existing",
            "output": str(output_path),
        }


    print("Scanning raw CSV...")

    df = pl.scan_csv(
        csv_path,
        infer_schema_length=10000,
        ignore_errors=True,
    )


    # -----------------------------------------------------
    # CLEAN COLUMN HEADERS
    # -----------------------------------------------------

    raw_columns = (
        df.collect_schema().names()
    )

    rename_map = {
        column: column.strip()
        for column in raw_columns
        if column.strip()
        and column.strip() != column
    }

    if rename_map:

        df = df.rename(
            rename_map
        )


    valid_columns = [
        column
        for column in (
            df.collect_schema().names()
        )
        if column.strip()
    ]

    df = df.select(
        valid_columns
    )


    available_columns = (
        df.collect_schema().names()
    )


    # -----------------------------------------------------
    # VALIDATE SCHEMA
    # -----------------------------------------------------

    missing_required = [
        column
        for column in REQUIRED_COLUMNS
        if column not in available_columns
    ]

    if missing_required:

        raise RuntimeError(
            f"Required columns missing: "
            f"{missing_required}"
        )


    # -----------------------------------------------------
    # FILTER DELTA
    # -----------------------------------------------------

    df = df.filter(
        pl.col(
            "Marketing_Airline_Network"
        )
        == "DL"
    )


    # -----------------------------------------------------
    # SELECT FIELDS
    # -----------------------------------------------------

    selected_columns = [
        column
        for column in WANTED_COLUMNS
        if column in available_columns
    ]

    df = df.select(
        selected_columns
    )


    # -----------------------------------------------------
    # STANDARDIZE TYPES
    # -----------------------------------------------------

    df = df.with_columns(

        pl.col("FlightDate")
        .str.strptime(
            pl.Date,
            "%Y-%m-%d",
            strict=False,
        )
        .alias("flight_date"),


        pl.col("Year")
        .cast(
            pl.Int16,
            strict=False,
        )
        .alias("year"),


        pl.col("Quarter")
        .cast(
            pl.Int8,
            strict=False,
        )
        .alias("quarter"),


        pl.col("Month")
        .cast(
            pl.Int8,
            strict=False,
        )
        .alias("month"),


        pl.col("DayofMonth")
        .cast(
            pl.Int8,
            strict=False,
        )
        .alias("day_of_month"),


        pl.col("DayOfWeek")
        .cast(
            pl.Int8,
            strict=False,
        )
        .alias("day_of_week"),
    )


    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    df = df.with_columns(

        pl.concat_str(
            [
                pl.col("Year")
                .cast(pl.String),

                pl.col("Month")
                .cast(pl.String)
                .str.pad_start(
                    2,
                    "0",
                ),
            ],
            separator="-",
        )
        .alias(
            "year_month"
        ),


        pl.concat_str(
            [
                pl.col("Origin"),
                pl.col("Dest"),
            ],
            separator="-",
        )
        .alias(
            "route"
        ),


        (
            pl.col(
                "Operating_Airline"
            )
            == "DL"
        )
        .alias(
            "is_delta_operated"
        ),


        pl.when(
            pl.col(
                "Operating_Airline"
            )
            == "DL"
        )
        .then(
            pl.lit(
                "Delta Mainline"
            )
        )
        .otherwise(
            pl.lit(
                "Partner"
            )
        )
        .alias(
            "operator_type"
        ),


        (
            pl.col("Cancelled")
            == 1
        )
        .alias(
            "is_cancelled"
        ),


        (
            pl.col("Diverted")
            == 1
        )
        .alias(
            "is_diverted"
        ),


        (
            pl.col("DepDel15")
            == 1
        )
        .alias(
            "is_departure_delayed_15"
        ),


        (
            pl.col("ArrDel15")
            == 1
        )
        .alias(
            "is_arrival_delayed_15"
        ),


        pl.when(
            (
                pl.col("Cancelled")
                == 0
            )
            & (
                pl.col("Diverted")
                == 0
            )
            & (
                pl.col("ArrDelay")
                .is_not_null()
            )
        )
        .then(
            pl.col("ArrDelay")
            <= 14
        )
        .otherwise(None)
        .alias(
            "is_on_time_arrival"
        ),
    )


    # -----------------------------------------------------
    # REMOVE DUPLICATE ORIGINAL DATE FIELDS
    # -----------------------------------------------------

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


    print(
        "Building clean Delta dataset..."
    )

    delta_df = df.collect()


    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

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


    missing_key_values = sum(
        delta_df[column]
        .null_count()

        for column in [
            "flight_date",
            "Operating_Airline",
            "Origin",
            "Dest",
        ]
    )


    if duplicate_count > 0:

        print(
            f"WARNING: "
            f"{duplicate_count:,} "
            f"potential duplicates"
        )


    if missing_key_values > 0:

        print(
            f"WARNING: "
            f"{missing_key_values:,} "
            f"missing key values"
        )


    # -----------------------------------------------------
    # SAVE PARQUET
    # -----------------------------------------------------

    delta_df.write_parquet(
        output_path,
        compression="zstd",
        statistics=True,
    )


    parquet_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )


    print(
        f"Delta flights: "
        f"{delta_df.height:,}"
    )

    print(
        f"Parquet size: "
        f"{parquet_mb:.2f} MB"
    )


    return {
        "year": year,
        "month": month,
        "rows": delta_df.height,
        "duplicates": int(
            duplicate_count
        ),
        "missing_keys": int(
            missing_key_values
        ),
        "size_mb": round(
            parquet_mb,
            2,
        ),
        "status": "processed",
        "output": str(
            output_path
        ),
    }


# =========================================================
# CLEAN TEMP FILES
# =========================================================

def cleanup(
    csv_path: Path,
    zip_path: Path,
):

    if (
        not KEEP_EXTRACTED_CSV
        and csv_path.exists()
    ):

        csv_path.unlink()

        print(
            "Removed extracted raw CSV."
        )


    if (
        not KEEP_ZIP_FILES
        and zip_path.exists()
    ):

        zip_path.unlink()

        print(
            "Removed ZIP file."
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)
    print("DELTA BUSINESS ANALYST")
    print("2020-2026 BTS HISTORICAL PIPELINE")
    print("=" * 72)

    print(
        f"\nPeriod: "
        f"{START_YEAR}-{START_MONTH:02d} "
        f"through "
        f"{END_YEAR}-{END_MONTH:02d}"
    )

    months = list(
        month_range(
            START_YEAR,
            START_MONTH,
            END_YEAR,
            END_MONTH,
        )
    )

    print(
        f"Monthly datasets: "
        f"{len(months)}"
    )


    results = []


    for index, (
        year,
        month,
    ) in enumerate(
        months,
        start=1,
    ):

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"[{index}/{len(months)}] "
            f"{year}-{month:02d}"
        )

        print(
            "=" * 72
        )


        output_path = (
            PROCESSED_DIR
            / f"delta_flights_"
              f"{year}_"
              f"{month:02d}.parquet"
        )


        # Completely processed months can be skipped
        # without downloading again.
        if output_path.exists():

            existing = (
                pl.scan_parquet(
                    output_path
                )
                .select(
                    pl.len()
                    .alias("rows")
                )
                .collect()
                .item()
            )

            print(
                f"Already processed: "
                f"{existing:,} rows"
            )

            results.append(
                {
                    "year": year,
                    "month": month,
                    "rows": existing,
                    "status": "existing",
                    "output": str(
                        output_path
                    ),
                }
            )

            continue


        try:

            zip_path = download_zip(
                year,
                month,
            )

            csv_path = extract_csv(
                zip_path
            )

            result = process_month(
                csv_path,
                year,
                month,
            )

            results.append(
                result
            )

            cleanup(
                csv_path,
                zip_path,
            )


        except Exception as exc:

            print(
                f"\nFAILED "
                f"{year}-{month:02d}"
            )

            print(exc)

            results.append(
                {
                    "year": year,
                    "month": month,
                    "rows": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )


    # =====================================================
    # PIPELINE MANIFEST
    # =====================================================

    manifest = pl.DataFrame(
        results
    )


    manifest_path = (
        PROCESSED_DIR.parent
        / "bts_pipeline_manifest.csv"
    )


    manifest.write_csv(
        manifest_path
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    successful = (
        manifest
        .filter(
            pl.col("status")
            != "failed"
        )
    )


    failed = (
        manifest
        .filter(
            pl.col("status")
            == "failed"
        )
    )


    total_rows = (
        successful
        ["rows"]
        .drop_nulls()
        .sum()
    )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "HISTORICAL PIPELINE SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        f"\nMonths requested: "
        f"{len(months)}"
    )

    print(
        f"Months completed: "
        f"{successful.height}"
    )

    print(
        f"Months failed: "
        f"{failed.height}"
    )

    print(
        f"Total Delta flight records: "
        f"{total_rows:,}"
    )

    print(
        f"\nManifest saved to:\n"
        f"{manifest_path}"
    )


    if failed.height > 0:

        print(
            "\nFailed months:"
        )

        print(
            failed.select(
                [
                    "year",
                    "month",
                    "error",
                ]
            )
        )


    print(
        "\n"
        + "=" * 72
    )

    print(
        "BTS HISTORICAL DATASET BUILD COMPLETE"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()