from pathlib import Path
from collections import Counter
import json
import math

import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


INPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "sql_benchmark_results_v2_raw.csv"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "sql_benchmark_results_v2_audited.csv"
)


SUMMARY_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "sql_benchmark_summary_v2_audited.json"
)


# =========================================================
# SETTINGS
# =========================================================

NUMERIC_TOLERANCE = 0.011


DECLINE_TERMS = (
    "decline",
    "deterioration",
    "decrease",
    "drop",
)


# =========================================================
# HELPERS
# =========================================================

def is_nan(
    value,
):

    try:

        return math.isnan(
            float(value)
        )

    except (
        TypeError,
        ValueError,
    ):

        return False


def is_number(
    value,
):

    if isinstance(
        value,
        bool,
    ):

        return False


    return isinstance(
        value,
        (
            int,
            float,
        ),
    )


def normalize_text(
    value,
):

    return (
        str(value)
        .strip()
        .upper()
    )


def is_year_value(
    value,
):

    if not is_number(
        value
    ):

        return False


    numeric = float(
        value
    )


    return (
        numeric.is_integer()
        and
        2000
        <= numeric
        <= 2100
    )


# =========================================================
# PARSE RESULT JSON
# =========================================================

def parse_result_json(
    text,
):

    if (
        text is None
        or
        pd.isna(text)
        or
        str(text).strip()
        == ""
    ):

        return []


    try:

        parsed = json.loads(
            text
        )


    except Exception:

        return []


    if isinstance(
        parsed,
        list,
    ):

        return parsed


    if isinstance(
        parsed,
        dict,
    ):

        return [
            parsed
        ]


    return []


# =========================================================
# EXTRACT RESULT INFORMATION
# =========================================================

def extract_result_values(
    records,
):

    texts = []

    numbers = []

    column_names = []


    for record in records:

        if not isinstance(
            record,
            dict,
        ):

            continue


        for column in record.keys():

            column_names.append(
                normalize_text(
                    column
                )
            )


        for value in record.values():

            if value is None:

                continue


            if is_number(
                value
            ):

                if is_nan(
                    value
                ):

                    continue


                numbers.append(
                    float(value)
                )


            else:

                text = normalize_text(
                    value
                )


                if text not in {
                    "",
                    "NAN",
                    "NONE",
                    "NULL",
                }:

                    texts.append(
                        text
                    )


    return {
        "texts":
            texts,

        "numbers":
            numbers,

        "columns":
            column_names,
    }


# =========================================================
# YEAR LABEL CHECK
# =========================================================

def year_is_represented(
    year_value,
    actual_data,
):

    year_text = str(
        int(
            year_value
        )
    )


    # Check actual numeric cells.

    for number in actual_data[
        "numbers"
    ]:

        if abs(
            number
            - year_value
        ) <= NUMERIC_TOLERANCE:

            return True


    # Also allow the year to be encoded
    # in a descriptive column name.
    #
    # Example:
    #
    # on_time_pct_2025_h1
    # on_time_pct_2026_h1

    for column in actual_data[
        "columns"
    ]:

        if year_text in column:

            return True


    return False


# =========================================================
# NUMERIC MATCHING
# =========================================================

def numbers_match(
    expected_value,
    actual_value,
    allow_absolute=False,
):

    expected_value = float(
        expected_value
    )


    actual_value = float(
        actual_value
    )


    if allow_absolute:

        difference = abs(
            abs(expected_value)
            -
            abs(actual_value)
        )


    else:

        difference = abs(
            expected_value
            -
            actual_value
        )


    return (
        difference
        <= NUMERIC_TOLERANCE
    )


# =========================================================
# SEMANTIC RESULT COMPARISON
# =========================================================

def semantic_compare(
    question,
    expected_records,
    actual_records,
):

    if not actual_records:

        return (
            False,
            [
                "Actual result is empty."
            ],
        )


    expected_data = (
        extract_result_values(
            expected_records
        )
    )


    actual_data = (
        extract_result_values(
            actual_records
        )
    )


    problems = []


    question_lower = (
        question.lower()
    )


    allow_absolute_change = any(
        term in question_lower
        for term in DECLINE_TERMS
    )


    # =====================================================
    # TEXT VALUES
    # =====================================================
    #
    # This validates entities such as:
    #
    # LGA
    # DCA
    # EWR
    # Partner
    # Delta Mainline
    # OO
    #
    # =====================================================

    expected_text_counter = Counter(
        expected_data[
            "texts"
        ]
    )


    actual_text_counter = Counter(
        actual_data[
            "texts"
        ]
    )


    for (
        expected_text,
        expected_count,
    ) in expected_text_counter.items():

        actual_count = (
            actual_text_counter[
                expected_text
            ]
        )


        if (
            actual_count
            < expected_count
        ):

            problems.append(
                (
                    "Missing expected text value: "
                    f"{expected_text}"
                )
            )


    # =====================================================
    # NUMERIC VALUES
    # =====================================================

    unused_actual_numbers = list(
        actual_data[
            "numbers"
        ]
    )


    for expected_number in (
        expected_data[
            "numbers"
        ]
    ):

        # -------------------------------------------------
        # YEAR LABELS
        # -------------------------------------------------

        if is_year_value(
            expected_number
        ):

            if year_is_represented(
                expected_number,
                actual_data,
            ):

                continue


            problems.append(
                (
                    "Missing expected year/context: "
                    f"{int(expected_number)}"
                )
            )

            continue


        # -------------------------------------------------
        # FIND NUMERIC MATCH
        # -------------------------------------------------

        matched_index = None


        for index, actual_number in enumerate(
            unused_actual_numbers
        ):

            # Do not use year values as answers
            # for ordinary metrics.

            if is_year_value(
                actual_number
            ):

                continue


            if numbers_match(
                expected_value=(
                    expected_number
                ),
                actual_value=(
                    actual_number
                ),
                allow_absolute=(
                    allow_absolute_change
                ),
            ):

                matched_index = (
                    index
                )

                break


        if matched_index is None:

            problems.append(
                (
                    "Missing expected numeric value: "
                    f"{expected_number}"
                )
            )


        else:

            unused_actual_numbers.pop(
                matched_index
            )


    return (
        len(
            problems
        )
        == 0,
        problems,
    )


# =========================================================
# AUDIT ONE BENCHMARK ROW
# =========================================================

def audit_row(
    row,
):

    original_status = (
        str(
            row.get(
                "status",
                "",
            )
        )
        .strip()
        .upper()
    )


    question = str(
        row.get(
            "question",
            "",
        )
    )


    execution_success = str(
        row.get(
            "execution_success",
            "",
        )
    ).strip().lower()


    execution_success = (
        execution_success
        == "true"
    )


    expected_records = (
        parse_result_json(
            row.get(
                "expected_result",
                "",
            )
        )
    )


    actual_records = (
        parse_result_json(
            row.get(
                "actual_result",
                "",
            )
        )
    )


    if not execution_success:

        audited_correct = False

        audited_status = (
            "FAIL"
        )

        audit_problems = [
            (
                "Query did not successfully "
                "complete execution."
            )
        ]


    else:

        (
            audited_correct,
            audit_problems,
        ) = semantic_compare(
            question=question,
            expected_records=(
                expected_records
            ),
            actual_records=(
                actual_records
            ),
        )


        audited_status = (
            "PASS"
            if audited_correct
            else "WRONG"
        )


    changed = (
        audited_status
        != original_status
    )


    if changed:

        audit_note = (
            "Original strict scorer status "
            f"{original_status} changed to "
            f"{audited_status} after semantic audit."
        )


    elif audited_correct:

        audit_note = (
            "Original result confirmed."
        )


    else:

        audit_note = (
            "; ".join(
                audit_problems
            )
        )


    row[
        "original_status"
    ] = (
        original_status
    )


    row[
        "audited_status"
    ] = (
        audited_status
    )


    row[
        "audited_correct"
    ] = (
        audited_correct
    )


    row[
        "changed_by_audit"
    ] = (
        changed
    )


    row[
        "audit_note"
    ] = (
        audit_note
    )


    return row


# =========================================================
# BUILD SUMMARY
# =========================================================

def build_summary(
    dataframe,
):

    total = len(
        dataframe
    )


    correct = int(
        dataframe[
            "audited_correct"
        ].sum()
    )


    accuracy = (
        100.0
        * correct
        / total
        if total
        else 0.0
    )


    execution_successes = int(
        dataframe[
            "execution_success"
        ]
        .astype(
            str
        )
        .str.lower()
        .eq(
            "true"
        )
        .sum()
    )


    execution_rate = (
        100.0
        * execution_successes
        / total
        if total
        else 0.0
    )


    difficulty_accuracy = {}


    for difficulty in [
        "easy",
        "medium",
        "hard",
    ]:

        subset = (
            dataframe[
                dataframe[
                    "difficulty"
                ]
                .str.lower()
                == difficulty
            ]
        )


        if len(
            subset
        ):

            value = (
                100.0
                * subset[
                    "audited_correct"
                ].sum()
                / len(
                    subset
                )
            )


            difficulty_accuracy[
                difficulty
            ] = round(
                float(
                    value
                ),
                2,
            )


    category_accuracy = {}


    for category in sorted(
        dataframe[
            "category"
        ].unique()
    ):

        subset = (
            dataframe[
                dataframe[
                    "category"
                ]
                == category
            ]
        )


        value = (
            100.0
            * subset[
                "audited_correct"
            ].sum()
            / len(
                subset
            )
        )


        category_accuracy[
            category
        ] = round(
            float(
                value
            ),
            2,
        )


    changed_rows = (
        dataframe[
            dataframe[
                "changed_by_audit"
            ]
            == True
        ]
    )


    summary = {

        "benchmark_version":
            "v2_audited",

        "total_questions":
            total,

        "execution_success_rate_pct":
            round(
                execution_rate,
                2,
            ),

        "correct_answers":
            correct,

        "answer_accuracy_pct":
            round(
                accuracy,
                2,
            ),

        "difficulty_accuracy_pct":
            difficulty_accuracy,

        "category_accuracy_pct":
            category_accuracy,

        "answers_changed_by_audit":
            len(
                changed_rows
            ),

        "changed_ids":
            changed_rows[
                "id"
            ].tolist(),
    }


    return summary


# =========================================================
# PRINT REPORT
# =========================================================

def print_report(
    dataframe,
    summary,
):

    print("=" * 80)
    print("DELTA BUSINESS ANALYST")
    print("V2 BENCHMARK SEMANTIC AUDIT")
    print("=" * 80)


    print(
        f"\nQuestions audited: "
        f"{summary['total_questions']}"
    )


    print(
        "Execution success: "
        f"{summary['execution_success_rate_pct']:.2f}%"
    )


    print(
        "Audited answer accuracy: "
        f"{summary['answer_accuracy_pct']:.2f}%"
    )


    print(
        "\nAccuracy by difficulty:"
    )


    for (
        difficulty,
        accuracy,
    ) in summary[
        "difficulty_accuracy_pct"
    ].items():

        print(
            f"  {difficulty:<8} "
            f"{accuracy:>6.2f}%"
        )


    print(
        "\nStatus changes after audit:"
    )


    changed = (
        dataframe[
            dataframe[
                "changed_by_audit"
            ]
            == True
        ]
    )


    if len(
        changed
    ) == 0:

        print(
            "  None"
        )


    else:

        for _, row in (
            changed.iterrows()
        ):

            print(
                f"  {row['id']}: "
                f"{row['original_status']} "
                f"â†’ {row['audited_status']}"
            )


    print(
        "\nRemaining failures:"
    )


    failed = (
        dataframe[
            dataframe[
                "audited_correct"
            ]
            == False
        ]
    )


    for _, row in (
        failed.iterrows()
    ):

        print(
            f"  {row['id']:<7} "
            f"{row['category']:<24} "
            f"{row['audited_status']}"
        )


    print(
        "\nSaved audited results:"
    )

    print(
        OUTPUT_PATH
    )


    print(
        "\nSaved audited summary:"
    )

    print(
        SUMMARY_PATH
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            "V1 baseline results file "
            "was not found:\n"
            f"{INPUT_PATH}"
        )


    dataframe = pd.read_csv(
        INPUT_PATH
    )


    audited_rows = []


    for record in (
        dataframe
        .to_dict(
            orient="records"
        )
    ):

        audited_rows.append(
            audit_row(
                record
            )
        )


    audited_dataframe = (
        pd.DataFrame(
            audited_rows
        )
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    audited_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )


    summary = (
        build_summary(
            audited_dataframe
        )
    )


    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    print_report(
        audited_dataframe,
        summary,
    )


if __name__ == "__main__":

    main()
