from dataclasses import dataclass
import calendar
import re


# =========================================================
# RESULT OBJECT
# =========================================================

@dataclass
class DeterministicSQLPlan:

    sql: str

    explanation: str

    pattern: str

    assumptions: list[str]


# =========================================================
# HELPERS
# =========================================================

def normalize_question(
    question: str,
):

    return re.sub(
        r"\s+",
        " ",
        question.lower().strip(),
    )


def extract_years(
    question: str,
):

    years = re.findall(
        r"\b(20\d{2})\b",
        question,
    )


    return [
        int(year)
        for year
        in dict.fromkeys(
            years
        )
    ]


def extract_top_n(
    question: str,
    default=5,
):

    q = normalize_question(
        question
    )


    word_numbers = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }


    match = re.search(
        r"\btop\s+(\d+)\b",
        q,
    )


    if match:

        return int(
            match.group(1)
        )


    match = re.search(
        r"\b(\d+)\s+(?:origin\s+)?airports?\b",
        q,
    )


    if match:

        return int(
            match.group(1)
        )


    for word, number in (
        word_numbers.items()
    ):

        if re.search(
            rf"\b{word}\s+(?:origin\s+)?airports?\b",
            q,
        ):

            return number


    if (
        "which origin airport"
        in q
        or
        "which airport"
        in q
    ):

        return 1


    return default


def extract_threshold(
    question: str,
    default=5000,
):

    q = normalize_question(
        question
    )


    patterns = [

        r"at least\s+([\d,]+)",

        r"minimum\s+(?:of\s+)?([\d,]+)",

        r"more than\s+([\d,]+)",

        r"over\s+([\d,]+)",
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            q,
        )


        if match:

            return int(
                match.group(1)
                .replace(
                    ",",
                    "",
                )
            )


    return default


def extract_month(
    question: str,
):

    q = normalize_question(
        question
    )


    for month_number in range(
        1,
        13,
    ):

        month_name = (
            calendar.month_name[
                month_number
            ].lower()
        )


        abbreviation = (
            calendar.month_abbr[
                month_number
            ].lower()
        )


        if re.search(
            rf"\b{month_name}\b",
            q,
        ):

            return month_number


        if re.search(
            rf"\b{abbreviation}\b",
            q,
        ):

            return month_number


    return None


# =========================================================
# SQL BUILDERS
# =========================================================

def build_overall_rate_sql(
    year,
    metric,
    month=None,
):

    if metric == "cancellation":

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "cancellation_rate_pct"
        )


    else:

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "on_time_pct"
        )


    where_parts = [
        f"year = {year}"
    ]


    if month is not None:

        where_parts.append(
            "EXTRACT("
            "MONTH FROM flight_date"
            f") = {month}"
        )


    where_sql = (
        "\n    AND ".join(
            where_parts
        )
    )


    return f"""
SELECT
    ROUND(
        {expression},
        2
    ) AS {alias}
FROM fact_flights
WHERE
    {where_sql}
""".strip()


def build_year_comparison_sql(
    years,
    metric,
    h1=False,
):

    year_list = ", ".join(
        str(year)
        for year
        in years
    )


    if metric == "cancellation":

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "cancellation_rate_pct"
        )


    else:

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "on_time_pct"
        )


    h1_filter = ""


    if h1:

        h1_filter = """
    AND EXTRACT(
        MONTH FROM flight_date
    ) <= 6
"""


    return f"""
SELECT
    year,
    ROUND(
        {expression},
        2
    ) AS {alias}
FROM fact_flights
WHERE
    year IN ({year_list})
{h1_filter}
GROUP BY year
ORDER BY year
""".strip()


def build_operator_type_comparison_sql(
    year,
    metric,
):

    if metric == "cancellation":

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "cancellation_rate_pct"
        )


    else:

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "on_time_pct"
        )


    return f"""
SELECT
    operator_type,
    ROUND(
        {expression},
        2
    ) AS {alias}
FROM fact_flights
WHERE
    year = {year}
    AND operator_type IN (
        'Delta Mainline',
        'Partner'
    )
GROUP BY operator_type
ORDER BY operator_type
""".strip()


def build_operator_count_sql(
    year,
    operator_type,
):

    if operator_type == "Delta Mainline":

        alias = (
            "mainline_flights"
        )


    else:

        alias = (
            "partner_flights"
        )


    return f"""
SELECT
    COUNT(*) AS {alias}
FROM fact_flights
WHERE
    year = {year}
    AND operator_type = '{operator_type}'
""".strip()


def build_operating_carrier_rank_sql(
    year,
    direction,
):

    order = (
        "DESC"
        if direction == "highest"
        else "ASC"
    )


    return f"""
SELECT
    Operating_Airline,
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
WHERE
    year = {year}
GROUP BY Operating_Airline
ORDER BY cancellation_rate_pct {order}
LIMIT 1
""".strip()


def build_airport_rank_sql(
    year,
    metric,
    direction,
    top_n,
    threshold,
):

    order = (
        "ASC"
        if direction == "worst"
        else "DESC"
    )


    if metric == "on_time":

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_on_time_arrival
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "on_time_pct"
        )


    else:

        expression = """
            100.0 * AVG(
                CASE
                    WHEN is_cancelled
                    THEN 1
                    ELSE 0
                END
            )
        """

        alias = (
            "cancellation_rate_pct"
        )


        # For cancellation:
        # highest = worst
        # lowest = best.

        if direction == "worst":

            order = "DESC"


        else:

            order = "ASC"


    return f"""
SELECT
    Origin,
    ROUND(
        {expression},
        2
    ) AS {alias}
FROM fact_flights
WHERE
    year = {year}
GROUP BY Origin
HAVING COUNT(*) >= {threshold}
ORDER BY {alias} {order}
LIMIT {top_n}
""".strip()


def build_airport_change_sql(
    start_year,
    end_year,
    threshold,
    top_n,
    direction,
):

    order = (
        "ASC"
        if direction == "deterioration"
        else "DESC"
    )


    return f"""
WITH yearly_airport AS (

    SELECT
        Origin,
        year,
        COUNT(*) AS departures,

        100.0 * AVG(
            CASE
                WHEN is_on_time_arrival
                THEN 1
                ELSE 0
            END
        ) AS on_time_pct

    FROM fact_flights

    WHERE
        year IN (
            {start_year},
            {end_year}
        )

    GROUP BY
        Origin,
        year

    HAVING
        COUNT(*) >= {threshold}
),

airport_comparison AS (

    SELECT
        Origin,

        MAX(
            CASE
                WHEN year = {start_year}
                THEN on_time_pct
            END
        ) AS on_time_{start_year},

        MAX(
            CASE
                WHEN year = {end_year}
                THEN on_time_pct
            END
        ) AS on_time_{end_year},

        COUNT(
            DISTINCT year
        ) AS years_present

    FROM yearly_airport

    GROUP BY Origin
)

SELECT
    Origin,

    ROUND(
        on_time_{start_year},
        2
    ) AS on_time_{start_year},

    ROUND(
        on_time_{end_year},
        2
    ) AS on_time_{end_year},

    ROUND(
        on_time_{end_year}
        -
        on_time_{start_year},
        2
    ) AS change_pp

FROM airport_comparison

WHERE
    years_present = 2

ORDER BY
    change_pp {order}

LIMIT {top_n}
""".strip()


def build_financial_period_compare_sql(
    years,
    quarter,
    metric,
):

    year_list = ", ".join(
        str(year)
        for year
        in years
    )


    metric_map = {

        "revenue":
            "revenue_m",

        "operating_income":
            "operating_income_m",

        "net_income":
            "net_income_m",

        "total_debt":
            "total_debt_m",
    }


    column = metric_map[
        metric
    ]


    return f"""
SELECT
    year,
    quarter,
    {column}
FROM financial_quarterly
WHERE
    year IN ({year_list})
    AND quarter = {quarter}
ORDER BY year
""".strip()


def build_operating_margin_compare_sql(
    years,
    quarter,
):

    year_list = ", ".join(
        str(year)
        for year
        in years
    )


    return f"""
SELECT
    year,
    quarter,

    ROUND(
        100.0
        * operating_income_m
        / revenue_m,
        2
    ) AS operating_margin_pct

FROM financial_quarterly

WHERE
    year IN ({year_list})
    AND quarter = {quarter}

ORDER BY year
""".strip()


def build_revenue_change_sql(
    start_year,
    end_year,
    quarter,
):

    return f"""
SELECT
    MAX(
        CASE
            WHEN year = {end_year}
            THEN revenue_m
        END
    )
    -
    MAX(
        CASE
            WHEN year = {start_year}
            THEN revenue_m
        END
    ) AS revenue_change_m

FROM financial_quarterly

WHERE
    quarter = {quarter}
    AND year IN (
        {start_year},
        {end_year}
    )
""".strip()


def build_margin_change_sql(
    start_year,
    end_year,
    quarter,
):

    return f"""
WITH margins AS (

    SELECT
        year,

        100.0
        * operating_income_m
        / revenue_m
        AS margin_pct

    FROM financial_quarterly

    WHERE
        quarter = {quarter}
        AND year IN (
            {start_year},
            {end_year}
        )
)

SELECT
    ROUND(

        MAX(
            CASE
                WHEN year = {end_year}
                THEN margin_pct
            END
        )

        -

        MAX(
            CASE
                WHEN year = {start_year}
                THEN margin_pct
            END
        ),

        2

    ) AS margin_change_pp

FROM margins
""".strip()


# =========================================================
# MAIN PATTERN ROUTER
# =========================================================

def get_deterministic_sql(
    question: str,
):

    q = normalize_question(
        question
    )


    years = extract_years(
        question
    )


    month = extract_month(
        question
    )


    # =====================================================
    # AIRPORT YEAR-OVER-YEAR CHANGE
    # =====================================================

    if (
        "airport"
        in q
        and len(
            years
        ) >= 2
        and (
            "on-time"
            in q
            or "on time"
            in q
        )
        and any(
            term in q
            for term in [
                "decline",
                "deterioration",
                "improve",
                "improvement",
            ]
        )
    ):

        start_year = years[0]

        end_year = years[1]

        threshold = (
            extract_threshold(
                question,
                default=5000,
            )
        )


        top_n = (
            extract_top_n(
                question,
                default=1,
            )
        )


        direction = (
            "deterioration"
            if any(
                term in q
                for term in [
                    "decline",
                    "deterioration",
                ]
            )
            else "improvement"
        )


        return DeterministicSQLPlan(

            sql=build_airport_change_sql(
                start_year=start_year,
                end_year=end_year,
                threshold=threshold,
                top_n=top_n,
                direction=direction,
            ),

            explanation=(
                "Uses a deterministic year-over-year "
                "airport comparison with the volume "
                "threshold enforced separately in each year."
            ),

            pattern=(
                "airport_year_over_year_change"
            ),

            assumptions=[
                (
                    "Airport means origin airport "
                    "unless destination is requested."
                )
            ],
        )


    # =====================================================
    # AIRPORT RANKINGS
    # =====================================================

    if (
        "airport"
        in q
        and len(
            years
        ) == 1
        and any(
            term in q
            for term in [
                "worst",
                "best",
                "highest",
                "lowest",
            ]
        )
    ):

        year = years[0]

        threshold = (
            extract_threshold(
                question,
                default=5000,
            )
        )

        top_n = (
            extract_top_n(
                question,
                default=5,
            )
        )


        if (
            "on-time"
            in q
            or "on time"
            in q
        ):

            metric = "on_time"

            direction = (
                "worst"
                if (
                    "worst"
                    in q
                    or "lowest"
                    in q
                )
                else "best"
            )


        elif (
            "cancellation"
            in q
        ):

            metric = (
                "cancellation"
            )

            direction = (
                "worst"
                if (
                    "highest"
                    in q
                    or "worst"
                    in q
                )
                else "best"
            )


        else:

            metric = None


        if metric:

            return DeterministicSQLPlan(

                sql=build_airport_rank_sql(
                    year=year,
                    metric=metric,
                    direction=direction,
                    top_n=top_n,
                    threshold=threshold,
                ),

                explanation=(
                    "Uses a deterministic airport "
                    "ranking with a minimum-volume filter."
                ),

                pattern=(
                    "airport_ranking"
                ),

                assumptions=[],
            )


    # =====================================================
    # H1 CANCELLATION COMPARISON
    # =====================================================

    if (
        len(
            years
        ) >= 2
        and (
            "first half"
            in q
            or "h1"
            in q
        )
        and "cancellation"
        in q
    ):

        return DeterministicSQLPlan(

            sql=build_year_comparison_sql(
                years=years[:2],
                metric="cancellation",
                h1=True,
            ),

            explanation=(
                "Calculates H1 cancellation rates "
                "independently for each year."
            ),

            pattern=(
                "h1_cancellation_comparison"
            ),

            assumptions=[
                "H1 means January through June."
            ],
        )


    # =====================================================
    # H1 ON-TIME COMPARISON
    # =====================================================

    if (
        len(
            years
        ) >= 2
        and (
            "first half"
            in q
            or "h1"
            in q
        )
        and (
            "on-time"
            in q
            or "on time"
            in q
        )
    ):

        return DeterministicSQLPlan(

            sql=build_year_comparison_sql(
                years=years[:2],
                metric="on_time",
                h1=True,
            ),

            explanation=(
                "Calculates H1 on-time arrival "
                "rates independently for each year."
            ),

            pattern=(
                "h1_on_time_comparison"
            ),

            assumptions=[
                "H1 means January through June."
            ],
        )


    # =====================================================
    # YEARLY CANCELLATION COMPARISON
    # =====================================================

    if (
        len(
            years
        ) >= 2
        and "cancellation rate"
        in q
    ):

        return DeterministicSQLPlan(

            sql=build_year_comparison_sql(
                years=years[:2],
                metric="cancellation",
            ),

            explanation=(
                "Calculates cancellation rate "
                "independently for each requested year."
            ),

            pattern=(
                "yearly_cancellation_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # YEARLY ON-TIME COMPARISON
    # =====================================================

    if (
        len(
            years
        ) >= 2
        and (
            "on-time arrival"
            in q
            or "on time arrival"
            in q
        )
        and "airport"
        not in q
    ):

        return DeterministicSQLPlan(

            sql=build_year_comparison_sql(
                years=years[:2],
                metric="on_time",
            ),

            explanation=(
                "Calculates overall on-time "
                "arrival rate independently "
                "for each requested year."
            ),

            pattern=(
                "yearly_on_time_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # MAINLINE VS PARTNER CANCELLATION
    # =====================================================

    if (
        "mainline"
        in q
        and "partner"
        in q
        and "cancellation"
        in q
        and len(
            years
        ) >= 1
    ):

        return DeterministicSQLPlan(

            sql=build_operator_type_comparison_sql(
                year=years[0],
                metric="cancellation",
            ),

            explanation=(
                "Compares cancellation rates using "
                "the exact operator_type values "
                "Delta Mainline and Partner."
            ),

            pattern=(
                "operator_type_cancellation_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # MAINLINE VS PARTNER ON-TIME
    # =====================================================

    if (
        "mainline"
        in q
        and "partner"
        in q
        and (
            "on-time"
            in q
            or "on time"
            in q
        )
        and len(
            years
        ) >= 1
    ):

        return DeterministicSQLPlan(

            sql=build_operator_type_comparison_sql(
                year=years[0],
                metric="on_time",
            ),

            explanation=(
                "Compares on-time arrival rates "
                "for Delta Mainline and Partner flights."
            ),

            pattern=(
                "operator_type_on_time_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # MAINLINE FLIGHT COUNT
    # =====================================================

    if (
        "mainline"
        in q
        and "how many"
        in q
        and len(
            years
        ) >= 1
    ):

        return DeterministicSQLPlan(

            sql=build_operator_count_sql(
                year=years[0],
                operator_type=(
                    "Delta Mainline"
                ),
            ),

            explanation=(
                "Counts flights whose exact "
                "operator_type is Delta Mainline."
            ),

            pattern=(
                "mainline_flight_count"
            ),

            assumptions=[],
        )


    # =====================================================
    # PARTNER FLIGHT COUNT
    # =====================================================

    if (
        "partner"
        in q
        and "how many"
        in q
        and len(
            years
        ) >= 1
    ):

        return DeterministicSQLPlan(

            sql=build_operator_count_sql(
                year=years[0],
                operator_type="Partner",
            ),

            explanation=(
                "Counts partner-operated "
                "Delta-marketed flights."
            ),

            pattern=(
                "partner_flight_count"
            ),

            assumptions=[],
        )


    # =====================================================
    # OPERATING CARRIER CANCELLATION RANKING
    # =====================================================

    if (
        "operating carrier"
        in q
        and "cancellation"
        in q
        and len(
            years
        ) >= 1
        and (
            "highest"
            in q
            or "lowest"
            in q
        )
    ):

        direction = (
            "highest"
            if "highest"
            in q
            else "lowest"
        )


        return DeterministicSQLPlan(

            sql=build_operating_carrier_rank_sql(
                year=years[0],
                direction=direction,
            ),

            explanation=(
                "Ranks actual operating carriers "
                "by cancellation rate."
            ),

            pattern=(
                "operating_carrier_cancellation_rank"
            ),

            assumptions=[],
        )


    # =====================================================
    # REVENUE CHANGE
    # =====================================================

    if (
        "revenue"
        in q
        and "change"
        in q
        and len(
            years
        ) >= 2
        and "q2"
        in q
    ):

        return DeterministicSQLPlan(

            sql=build_revenue_change_sql(
                start_year=years[0],
                end_year=years[1],
                quarter=2,
            ),

            explanation=(
                "Calculates the Q2 revenue "
                "change using USD millions."
            ),

            pattern=(
                "revenue_change"
            ),

            assumptions=[
                (
                    "Revenue change is later "
                    "period minus earlier period."
                )
            ],
        )


    # =====================================================
    # OPERATING MARGIN CHANGE
    # =====================================================

    if (
        "operating margin"
        in q
        and "change"
        in q
        and len(
            years
        ) >= 2
        and "q2"
        in q
    ):

        return DeterministicSQLPlan(

            sql=build_margin_change_sql(
                start_year=years[0],
                end_year=years[1],
                quarter=2,
            ),

            explanation=(
                "Calculates the Q2 operating-margin "
                "change in percentage points."
            ),

            pattern=(
                "operating_margin_change"
            ),

            assumptions=[],
        )


    # =====================================================
    # Q2 OPERATING MARGIN COMPARISON
    # =====================================================

    if (
        "operating margin"
        in q
        and "q2"
        in q
        and len(
            years
        ) >= 2
    ):

        return DeterministicSQLPlan(

            sql=build_operating_margin_compare_sql(
                years=years[:2],
                quarter=2,
            ),

            explanation=(
                "Compares Q2 operating margins "
                "using operating income divided "
                "by revenue."
            ),

            pattern=(
                "operating_margin_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # Q2 REVENUE COMPARISON
    # =====================================================

    if (
        "revenue"
        in q
        and "q2"
        in q
        and len(
            years
        ) >= 2
    ):

        return DeterministicSQLPlan(

            sql=build_financial_period_compare_sql(
                years=years[:2],
                quarter=2,
                metric="revenue",
            ),

            explanation=(
                "Returns Q2 revenue in USD millions "
                "for each requested year."
            ),

            pattern=(
                "revenue_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # Q2 OPERATING INCOME COMPARISON
    # =====================================================

    if (
        "operating income"
        in q
        and "q2"
        in q
        and len(
            years
        ) >= 2
    ):

        return DeterministicSQLPlan(

            sql=build_financial_period_compare_sql(
                years=years[:2],
                quarter=2,
                metric="operating_income",
            ),

            explanation=(
                "Returns Q2 operating income "
                "in USD millions."
            ),

            pattern=(
                "operating_income_comparison"
            ),

            assumptions=[],
        )


    # =====================================================
    # Q2 NET INCOME
    # =====================================================

    if (
        "net income"
        in q
        and "q2"
        in q
        and len(
            years
        ) == 1
    ):

        return DeterministicSQLPlan(

            sql=build_financial_period_compare_sql(
                years=years,
                quarter=2,
                metric="net_income",
            ),

            explanation=(
                "Returns Q2 net income "
                "in USD millions."
            ),

            pattern=(
                "net_income_q2"
            ),

            assumptions=[],
        )


    # =====================================================
    # Q2 TOTAL DEBT
    # =====================================================

    if (
        "debt"
        in q
        and "q2"
        in q
        and len(
            years
        ) == 1
    ):

        return DeterministicSQLPlan(

            sql=build_financial_period_compare_sql(
                years=years,
                quarter=2,
                metric="total_debt",
            ),

            explanation=(
                "Returns Q2 total debt "
                "in USD millions."
            ),

            pattern=(
                "total_debt_q2"
            ),

            assumptions=[],
        )


    # =====================================================
    # OVERALL ON-TIME RATE
    # =====================================================

    if (
        (
            "overall on-time"
            in q
            or "overall on time"
            in q
        )
        and len(
            years
        ) == 1
    ):

        return DeterministicSQLPlan(

            sql=build_overall_rate_sql(
                year=years[0],
                metric="on_time",
                month=month,
            ),

            explanation=(
                "Calculates overall on-time "
                "arrival rate using the scheduled "
                "flight population."
            ),

            pattern=(
                "overall_on_time_rate"
            ),

            assumptions=[],
        )


    # =====================================================
    # OVERALL CANCELLATION RATE
    # =====================================================

    if (
        "cancellation rate"
        in q
        and len(
            years
        ) == 1
        and "airport"
        not in q
        and "carrier"
        not in q
        and "mainline"
        not in q
        and "partner"
        not in q
    ):

        return DeterministicSQLPlan(

            sql=build_overall_rate_sql(
                year=years[0],
                metric="cancellation",
                month=month,
            ),

            explanation=(
                "Calculates the overall "
                "cancellation rate."
            ),

            pattern=(
                "overall_cancellation_rate"
            ),

            assumptions=[],
        )


    # =====================================================
    # NO TRUSTED TEMPLATE
    # =====================================================

    return None