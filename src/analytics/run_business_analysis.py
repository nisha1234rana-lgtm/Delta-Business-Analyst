from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "database"
    / "delta_analytics.duckdb"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "analysis"
)

CHART_DIR = (
    PROJECT_ROOT
    / "assets"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


if not DB_PATH.exists():
    raise FileNotFoundError(
        f"DuckDB warehouse not found:\n{DB_PATH}"
    )


# =========================================================
# CONNECT
# =========================================================

con = duckdb.connect(
    str(DB_PATH),
    read_only=True,
)


print("=" * 76)
print("DELTA BUSINESS ANALYST")
print("MULTI-YEAR BUSINESS ANALYSIS")
print("=" * 76)


# =========================================================
# 1. FULL-YEAR PERFORMANCE
# =========================================================

print("\n[1] Full-year performance 2020-2025")

yearly = con.execute("""
    SELECT
        year,
        scheduled_flights,
        cancellation_rate_pct,
        avg_departure_delay_min,
        avg_arrival_delay_min,
        on_time_arrival_pct,
        airports,
        routes

    FROM vw_yearly_kpis

    WHERE year BETWEEN 2020 AND 2025

    ORDER BY year;
""").fetchdf()

yearly.to_csv(
    OUTPUT_DIR / "yearly_performance_2020_2025.csv",
    index=False,
)

print(yearly.to_string(index=False))


# =========================================================
# 2. MONTHLY PERFORMANCE
# =========================================================

print("\n[2] Monthly KPI history")

monthly = con.execute("""
    SELECT
        year,
        month,
        year_month,
        scheduled_flights,
        cancellation_rate_pct,
        avg_departure_delay_min,
        avg_arrival_delay_min,
        on_time_arrival_pct,
        delta_operated_pct,
        origin_airports,
        routes

    FROM vw_monthly_kpis

    ORDER BY
        year,
        month;
""").fetchdf()

monthly["date"] = pd.to_datetime(
    monthly["year_month"] + "-01"
)

monthly.to_csv(
    OUTPUT_DIR / "monthly_kpis_2020_2026.csv",
    index=False,
)


# =========================================================
# 3. H1 2025 VS H1 2026
# =========================================================

print("\n[3] H1 2025 vs H1 2026")

h1_yoy = con.execute("""
    SELECT
        year,

        COUNT(*) AS scheduled_flights,

        ROUND(
            100.0 * AVG(
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
            100.0 * AVG(
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

    WHERE
        year IN (2025, 2026)
        AND month BETWEEN 1 AND 6

    GROUP BY year

    ORDER BY year;
""").fetchdf()


h1_yoy.to_csv(
    OUTPUT_DIR / "h1_2025_vs_2026.csv",
    index=False,
)

print(h1_yoy.to_string(index=False))


# =========================================================
# 4. AIRPORT DETERIORATION
# =========================================================

print(
    "\n[4] Airports with largest on-time deterioration "
    "from 2024 to 2025"
)

airport_change = con.execute("""
    WITH airport_year AS (

        SELECT
            year,
            Origin AS airport,
            OriginCityName AS city,

            COUNT(*) AS departures,

            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ) AS on_time_pct,

            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ) AS cancellation_pct,

            AVG(DepDelay)
                AS avg_departure_delay

        FROM fact_flights

        WHERE year IN (2024, 2025)

        GROUP BY
            year,
            Origin,
            OriginCityName
    ),

    comparison AS (

        SELECT
            a.airport,
            a.city,

            a.departures
                AS departures_2024,

            b.departures
                AS departures_2025,

            ROUND(
                a.on_time_pct,
                2
            ) AS on_time_2024,

            ROUND(
                b.on_time_pct,
                2
            ) AS on_time_2025,

            ROUND(
                b.on_time_pct
                - a.on_time_pct,
                2
            ) AS on_time_change_pp,

            ROUND(
                a.cancellation_pct,
                2
            ) AS cancellation_2024,

            ROUND(
                b.cancellation_pct,
                2
            ) AS cancellation_2025,

            ROUND(
                b.avg_departure_delay
                - a.avg_departure_delay,
                2
            ) AS departure_delay_change_min

        FROM airport_year a

        JOIN airport_year b

            ON a.airport = b.airport

        WHERE
            a.year = 2024
            AND b.year = 2025
            AND a.departures >= 5000
            AND b.departures >= 5000
    )

    SELECT *

    FROM comparison

    ORDER BY on_time_change_pp ASC;
""").fetchdf()


airport_change.to_csv(
    OUTPUT_DIR / "airport_performance_change_2024_2025.csv",
    index=False,
)

print(
    airport_change
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 5. WORST HIGH-VOLUME ROUTES
# =========================================================

print(
    "\n[5] Worst high-volume routes in 2025"
)

routes_2025 = con.execute("""
    SELECT
        route,
        Origin AS origin,
        Dest AS destination,

        COUNT(*) AS flights,

        ROUND(
            AVG(ArrDelay),
            2
        ) AS avg_arrival_delay_min,

        ROUND(
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct,

        ROUND(
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct

    FROM fact_flights

    WHERE year = 2025

    GROUP BY
        route,
        Origin,
        Dest

    HAVING COUNT(*) >= 500

    ORDER BY
        on_time_arrival_pct ASC,
        flights DESC;
""").fetchdf()


routes_2025.to_csv(
    OUTPUT_DIR / "route_performance_2025.csv",
    index=False,
)

print(
    routes_2025
    .head(15)
    .to_string(index=False)
)


# =========================================================
# 6. OPERATING CARRIER PERFORMANCE
# =========================================================

print(
    "\n[6] Delta network operating-carrier performance, 2025"
)

operators = con.execute("""
    SELECT
        Operating_Airline AS operator,

        COUNT(*) AS flights,

        ROUND(
            100.0
            * COUNT(*)
            / SUM(COUNT(*)) OVER (),
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
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS on_time_arrival_pct,

        ROUND(
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            ),
            2
        ) AS cancellation_rate_pct

    FROM fact_flights

    WHERE year = 2025

    GROUP BY Operating_Airline

    ORDER BY flights DESC;
""").fetchdf()


operators.to_csv(
    OUTPUT_DIR / "operator_performance_2025.csv",
    index=False,
)

print(operators.to_string(index=False))


# =========================================================
# 7. DELAY CAUSES
# =========================================================

print(
    "\n[7] Delay causes by year"
)

delay_causes = con.execute("""
    SELECT
        year,

        ROUND(
            SUM(
                COALESCE(
                    CarrierDelay,
                    0
                )
            ) / 60.0,
            0
        ) AS carrier_delay_hours,

        ROUND(
            SUM(
                COALESCE(
                    WeatherDelay,
                    0
                )
            ) / 60.0,
            0
        ) AS weather_delay_hours,

        ROUND(
            SUM(
                COALESCE(
                    NASDelay,
                    0
                )
            ) / 60.0,
            0
        ) AS nas_delay_hours,

        ROUND(
            SUM(
                COALESCE(
                    SecurityDelay,
                    0
                )
            ) / 60.0,
            0
        ) AS security_delay_hours,

        ROUND(
            SUM(
                COALESCE(
                    LateAircraftDelay,
                    0
                )
            ) / 60.0,
            0
        ) AS late_aircraft_delay_hours

    FROM fact_flights

    WHERE year BETWEEN 2020 AND 2025

    GROUP BY year

    ORDER BY year;
""").fetchdf()


delay_causes.to_csv(
    OUTPUT_DIR / "delay_causes_2020_2025.csv",
    index=False,
)

print(delay_causes.to_string(index=False))


# =========================================================
# CHART 1
# MONTHLY ON-TIME PERFORMANCE
# =========================================================

fig = px.line(
    monthly,
    x="date",
    y="on_time_arrival_pct",
    title=(
        "Delta Network Monthly "
        "On-Time Arrival Performance"
    ),
    labels={
        "date": "Month",
        "on_time_arrival_pct":
            "On-Time Arrival (%)",
    },
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "monthly_on_time_arrival.html"
)


# =========================================================
# CHART 2
# CANCELLATION RATE
# =========================================================

fig = px.line(
    monthly,
    x="date",
    y="cancellation_rate_pct",
    title=(
        "Delta Network Monthly "
        "Cancellation Rate"
    ),
    labels={
        "date": "Month",
        "cancellation_rate_pct":
            "Cancellation Rate (%)",
    },
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "monthly_cancellation_rate.html"
)


# =========================================================
# CHART 3
# YEARLY FLIGHT VOLUME
# =========================================================

fig = px.bar(
    yearly,
    x="year",
    y="scheduled_flights",
    title=(
        "Delta Network Scheduled "
        "Flights by Year"
    ),
    labels={
        "year": "Year",
        "scheduled_flights":
            "Scheduled Flights",
    },
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "yearly_flight_volume.html"
)


# =========================================================
# CHART 4
# AIRPORT DETERIORATION
# =========================================================

worst_airports = (
    airport_change
    .head(15)
    .sort_values(
        "on_time_change_pp"
    )
)

fig = px.bar(
    worst_airports,
    x="on_time_change_pp",
    y="airport",
    orientation="h",
    hover_data=[
        "city",
        "departures_2025",
        "on_time_2024",
        "on_time_2025",
    ],
    title=(
        "Largest Airport On-Time "
        "Performance Declines: "
        "2024 → 2025"
    ),
    labels={
        "airport": "Airport",
        "on_time_change_pp":
            "Change in On-Time Performance "
            "(percentage points)",
    },
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "airport_deterioration_2024_2025.html"
)


# =========================================================
# CHART 5
# OPERATOR PERFORMANCE
# =========================================================

fig = px.scatter(
    operators,
    x="on_time_arrival_pct",
    y="cancellation_rate_pct",
    size="flights",
    text="operator",
    hover_data=[
        "network_share_pct",
        "avg_arrival_delay_min",
    ],
    title=(
        "Delta Network Operating "
        "Carrier Performance — 2025"
    ),
    labels={
        "on_time_arrival_pct":
            "On-Time Arrival (%)",
        "cancellation_rate_pct":
            "Cancellation Rate (%)",
    },
)

fig.update_traces(
    textposition="top center"
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "operator_performance_2025.html"
)


# =========================================================
# CHART 6
# DELAY CAUSES 2025
# =========================================================

delay_2025 = (
    delay_causes[
        delay_causes["year"] == 2025
    ]
    .drop(
        columns=["year"]
    )
    .T
    .reset_index()
)

delay_2025.columns = [
    "delay_type",
    "delay_hours",
]

delay_2025["delay_type"] = (
    delay_2025["delay_type"]
    .str.replace(
        "_delay_hours",
        "",
        regex=False,
    )
    .str.replace(
        "_",
        " ",
        regex=False,
    )
    .str.title()
)


fig = px.bar(
    delay_2025,
    x="delay_type",
    y="delay_hours",
    title=(
        "Delta Network Delay Causes — 2025"
    ),
    labels={
        "delay_type":
            "Delay Category",
        "delay_hours":
            "Total Delay Hours",
    },
)

fig.update_layout(
    template="plotly_white"
)

fig.write_html(
    CHART_DIR
    / "delay_causes_2025.html"
)


# =========================================================
# AUTOMATIC FINDINGS
# =========================================================

print("\n" + "=" * 76)
print("AUTOMATIC BUSINESS FINDINGS")
print("=" * 76)


best_year = yearly.loc[
    yearly[
        "on_time_arrival_pct"
    ].idxmax()
]

worst_year = yearly.loc[
    yearly[
        "on_time_arrival_pct"
    ].idxmin()
]


print(
    f"\nBest full-year on-time performance: "
    f"{int(best_year['year'])} "
    f"({best_year['on_time_arrival_pct']:.2f}%)"
)

print(
    f"Worst full-year on-time performance: "
    f"{int(worst_year['year'])} "
    f"({worst_year['on_time_arrival_pct']:.2f}%)"
)


if len(h1_yoy) == 2:

    y2025 = h1_yoy[
        h1_yoy["year"] == 2025
    ].iloc[0]

    y2026 = h1_yoy[
        h1_yoy["year"] == 2026
    ].iloc[0]

    change = (
        y2026["on_time_arrival_pct"]
        - y2025["on_time_arrival_pct"]
    )

    cancellation_change = (
        y2026["cancellation_rate_pct"]
        - y2025["cancellation_rate_pct"]
    )

    print(
        f"\nH1 2026 vs H1 2025 "
        f"on-time change: "
        f"{change:+.2f} percentage points"
    )

    print(
        f"H1 2026 vs H1 2025 "
        f"cancellation-rate change: "
        f"{cancellation_change:+.2f} percentage points"
    )


if not airport_change.empty:

    worst_airport = (
        airport_change.iloc[0]
    )

    print(
        f"\nLargest 2024→2025 "
        f"airport deterioration: "
        f"{worst_airport['airport']} "
        f"({worst_airport['city']})"
    )

    print(
        f"On-time change: "
        f"{worst_airport['on_time_change_pp']:+.2f} "
        f"percentage points"
    )


if not routes_2025.empty:

    worst_route = (
        routes_2025.iloc[0]
    )

    print(
        f"\nWorst high-volume 2025 route: "
        f"{worst_route['route']}"
    )

    print(
        f"On-time arrival: "
        f"{worst_route['on_time_arrival_pct']:.2f}%"
    )


# =========================================================
# COMPLETE
# =========================================================

con.close()


print("\n" + "=" * 76)
print("BUSINESS ANALYSIS COMPLETE")
print("=" * 76)

print(
    f"\nAnalysis tables:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nInteractive charts:\n"
    f"{CHART_DIR}"
)