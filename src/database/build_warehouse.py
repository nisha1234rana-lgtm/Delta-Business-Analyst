from pathlib import Path
import duckdb


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARQUET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delta_flights"
    / "monthly"
)

DATABASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "database"
)

DATABASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DB_PATH = (
    DATABASE_DIR
    / "delta_analytics.duckdb"
)

PARQUET_PATTERN = (
    str(PARQUET_DIR / "*.parquet")
    .replace("\\", "/")
)


# =========================================================
# VALIDATE INPUT
# =========================================================

parquet_files = list(
    PARQUET_DIR.glob("*.parquet")
)

if not parquet_files:
    raise FileNotFoundError(
        f"No monthly Parquet files found in:\n"
        f"{PARQUET_DIR}"
    )


print("=" * 72)
print("DELTA BUSINESS ANALYST")
print("DUCKDB ANALYTICAL WAREHOUSE")
print("=" * 72)

print(
    f"\nMonthly Parquet files found: "
    f"{len(parquet_files)}"
)

print(
    f"\nDatabase:\n{DB_PATH}"
)


# =========================================================
# CONNECT
# =========================================================

con = duckdb.connect(
    str(DB_PATH)
)


# =========================================================
# SETTINGS
# =========================================================

con.execute("""
    SET preserve_insertion_order = false;
""")


# =========================================================
# BUILD MAIN FLIGHT TABLE
# =========================================================

print("\nBuilding fact_flights...")

con.execute(
    f"""
    CREATE OR REPLACE TABLE fact_flights AS

    SELECT *

    FROM read_parquet(
        '{PARQUET_PATTERN}',
        union_by_name = true
    );
    """
)


# =========================================================
# ROW COUNT
# =========================================================

flight_count = con.execute(
    """
    SELECT COUNT(*)
    FROM fact_flights;
    """
).fetchone()[0]

print(
    f"Loaded flight records: "
    f"{flight_count:,}"
)


# =========================================================
# CREATE INDEXES
# =========================================================

print("\nCreating indexes...")

con.execute("""
    CREATE INDEX IF NOT EXISTS idx_flight_date
    ON fact_flights(flight_date);
""")

con.execute("""
    CREATE INDEX IF NOT EXISTS idx_origin
    ON fact_flights(Origin);
""")

con.execute("""
    CREATE INDEX IF NOT EXISTS idx_dest
    ON fact_flights(Dest);
""")

con.execute("""
    CREATE INDEX IF NOT EXISTS idx_route
    ON fact_flights(route);
""")

con.execute("""
    CREATE INDEX IF NOT EXISTS idx_operator
    ON fact_flights(Operating_Airline);
""")


# =========================================================
# DIMENSION: AIRPORTS
# =========================================================

print("Building dim_airport...")

con.execute("""
    CREATE OR REPLACE TABLE dim_airport AS

    WITH airports AS (

        SELECT DISTINCT
            Origin AS airport_code,
            OriginCityName AS city_name,
            OriginState AS state_code,
            OriginStateName AS state_name

        FROM fact_flights

        UNION

        SELECT DISTINCT
            Dest AS airport_code,
            DestCityName AS city_name,
            DestState AS state_code,
            DestStateName AS state_name

        FROM fact_flights
    )

    SELECT DISTINCT
        airport_code,
        city_name,
        state_code,
        state_name

    FROM airports

    WHERE airport_code IS NOT NULL;
""")


# =========================================================
# DIMENSION: ROUTES
# =========================================================

print("Building dim_route...")

con.execute("""
    CREATE OR REPLACE TABLE dim_route AS

    SELECT
        route,
        Origin AS origin,
        Dest AS destination,
        MIN(Distance) AS distance_miles

    FROM fact_flights

    GROUP BY
        route,
        Origin,
        Dest;
""")


# =========================================================
# MONTHLY KPI VIEW
# =========================================================

print("Building monthly KPI view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_monthly_kpis AS

    SELECT
        year,
        month,
        year_month,

        COUNT(*) AS scheduled_flights,

        SUM(
            CASE
                WHEN is_cancelled
                THEN 1
                ELSE 0
            END
        ) AS cancelled_flights,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct,

        SUM(
            CASE
                WHEN is_diverted
                THEN 1
                ELSE 0
            END
        ) AS diverted_flights,

        ROUND(
            AVG(DepDelay),
            2
        ) AS avg_departure_delay_min,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_departure_delayed_15
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS departure_delay_15plus_pct,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_arrival_delayed_15
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS arrival_delay_15plus_pct,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_delta_operated
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS delta_operated_pct,

        COUNT(
            DISTINCT Origin
        ) AS origin_airports,

        COUNT(
            DISTINCT route
        ) AS routes

    FROM fact_flights

    GROUP BY
        year,
        month,
        year_month;
""")


# =========================================================
# AIRPORT KPI VIEW
# =========================================================

print("Building airport KPI view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_airport_performance AS

    SELECT
        year,
        month,
        year_month,

        Origin AS airport_code,
        OriginCityName AS city,

        COUNT(*) AS departures,

        SUM(
            CASE
                WHEN is_cancelled
                THEN 1
                ELSE 0
            END
        ) AS cancellations,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct,

        ROUND(
            AVG(DepDelay),
            2
        ) AS avg_departure_delay_min,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct

    FROM fact_flights

    GROUP BY
        year,
        month,
        year_month,
        Origin,
        OriginCityName;
""")


# =========================================================
# ROUTE KPI VIEW
# =========================================================

print("Building route KPI view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_route_performance AS

    SELECT
        year,
        month,
        year_month,

        route,
        Origin AS origin,
        Dest AS destination,

        COUNT(*) AS flights,

        ROUND(
            AVG(Distance),
            0
        ) AS avg_distance_miles,

        ROUND(
            AVG(DepDelay),
            2
        ) AS avg_departure_delay_min,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct

    FROM fact_flights

    GROUP BY
        year,
        month,
        year_month,
        route,
        Origin,
        Dest;
""")


# =========================================================
# OPERATOR PERFORMANCE
# =========================================================

print("Building operator KPI view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_operator_performance AS

    SELECT
        year,
        month,
        year_month,

        Operating_Airline AS operating_airline,
        operator_type,

        COUNT(*) AS flights,

        ROUND(
            100.0
            * COUNT(*)
            / SUM(COUNT(*))
              OVER (
                  PARTITION BY year_month
              ),
            2
        ) AS network_share_pct,

        ROUND(
            AVG(DepDelay),
            2
        ) AS avg_departure_delay_min,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct

    FROM fact_flights

    GROUP BY
        year,
        month,
        year_month,
        Operating_Airline,
        operator_type;
""")


# =========================================================
# DELAY CAUSE VIEW
# =========================================================

print("Building delay-cause view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_delay_causes AS

    SELECT
        year,
        month,
        year_month,

        SUM(
            COALESCE(
                CarrierDelay,
                0
            )
        ) AS carrier_delay_minutes,

        SUM(
            COALESCE(
                WeatherDelay,
                0
            )
        ) AS weather_delay_minutes,

        SUM(
            COALESCE(
                NASDelay,
                0
            )
        ) AS nas_delay_minutes,

        SUM(
            COALESCE(
                SecurityDelay,
                0
            )
        ) AS security_delay_minutes,

        SUM(
            COALESCE(
                LateAircraftDelay,
                0
            )
        ) AS late_aircraft_delay_minutes

    FROM fact_flights

    GROUP BY
        year,
        month,
        year_month;
""")


# =========================================================
# YEARLY KPI VIEW
# =========================================================

print("Building yearly KPI view...")

con.execute("""
    CREATE OR REPLACE VIEW vw_yearly_kpis AS

    SELECT
        year,

        COUNT(*) AS scheduled_flights,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct,

        ROUND(
            AVG(DepDelay),
            2
        ) AS avg_departure_delay_min,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0
            * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct,

        COUNT(
            DISTINCT Origin
        ) AS airports,

        COUNT(
            DISTINCT route
        ) AS routes

    FROM fact_flights

    GROUP BY year;
""")


# =========================================================
# VALIDATION
# =========================================================

print("\nValidating warehouse...")


year_range = con.execute("""
    SELECT
        MIN(flight_date),
        MAX(flight_date)

    FROM fact_flights;
""").fetchone()


airport_count = con.execute("""
    SELECT COUNT(*)
    FROM dim_airport;
""").fetchone()[0]


route_count = con.execute("""
    SELECT COUNT(*)
    FROM dim_route;
""").fetchone()[0]


print(
    f"\nDate range: "
    f"{year_range[0]} → {year_range[1]}"
)

print(
    f"Airports: "
    f"{airport_count:,}"
)

print(
    f"Directional routes: "
    f"{route_count:,}"
)


# =========================================================
# YEARLY PERFORMANCE
# =========================================================

print("\n" + "=" * 72)
print("DELTA NETWORK YEARLY PERFORMANCE")
print("=" * 72)


yearly = con.execute("""
    SELECT *

    FROM vw_yearly_kpis

    ORDER BY year;
""").fetchdf()


print(
    yearly.to_string(
        index=False
    )
)


# =========================================================
# TOP AIRPORTS
# =========================================================

print("\n" + "=" * 72)
print("TOP AIRPORTS BY FLIGHT VOLUME")
print("=" * 72)


top_airports = con.execute("""
    SELECT
        Origin AS airport,
        OriginCityName AS city,
        COUNT(*) AS flights

    FROM fact_flights

    GROUP BY
        Origin,
        OriginCityName

    ORDER BY flights DESC

    LIMIT 15;
""").fetchdf()


print(
    top_airports.to_string(
        index=False
    )
)


# =========================================================
# TABLE / VIEW LIST
# =========================================================

print("\n" + "=" * 72)
print("WAREHOUSE OBJECTS")
print("=" * 72)


objects = con.execute("""
    SELECT
        table_name,
        table_type

    FROM information_schema.tables

    WHERE table_schema = 'main'

    ORDER BY
        table_type,
        table_name;
""").fetchdf()


print(
    objects.to_string(
        index=False
    )
)


# =========================================================
# CLOSE
# =========================================================

con.close()


db_size_mb = (
    DB_PATH.stat().st_size
    / (1024 * 1024)
)


print("\n" + "=" * 72)
print("WAREHOUSE BUILD COMPLETE")
print("=" * 72)

print(
    f"\nDatabase size: "
    f"{db_size_mb:.2f} MB"
)

print(
    f"\nSaved to:\n"
    f"{DB_PATH}"
)