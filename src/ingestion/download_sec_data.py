from pathlib import Path
import json
import time

import requests


# =========================================================
# CONFIG
# =========================================================

CIK = "0000027904"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec"
)

SEC_RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

COMPANY_FACTS_PATH = (
    SEC_RAW_DIR
    / "delta_companyfacts.json"
)

SUBMISSIONS_PATH = (
    SEC_RAW_DIR
    / "delta_submissions.json"
)


HEADERS = {
    "User-Agent": (
        "DeltaBusinessAnalyst/1.0 "
        "(academic analytics project)"
    ),
    "Accept-Encoding": "gzip, deflate",
}


COMPANY_FACTS_URL = (
    f"https://data.sec.gov/api/xbrl/"
    f"companyfacts/CIK{CIK}.json"
)

SUBMISSIONS_URL = (
    f"https://data.sec.gov/submissions/"
    f"CIK{CIK}.json"
)


# =========================================================
# DOWNLOAD OR LOAD JSON
# =========================================================

def get_json(
    url: str,
    destination: Path,
):

    if destination.exists():

        print(
            f"\nFile already exists: "
            f"{destination.name}"
        )

        print("Loading existing file...")

        with open(
            destination,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)


    print("\nDownloading:")
    print(url)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    with open(
        destination,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )

    size_mb = (
        destination.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Saved: {destination.name}"
    )

    print(
        f"Size: {size_mb:.2f} MB"
    )

    return data


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 72)
    print("DELTA BUSINESS ANALYST")
    print("SEC FINANCIAL DATA INGESTION")
    print("=" * 72)


    # -----------------------------------------------------
    # COMPANY FACTS
    # -----------------------------------------------------

    company_facts = get_json(
        COMPANY_FACTS_URL,
        COMPANY_FACTS_PATH,
    )

    time.sleep(0.2)


    # -----------------------------------------------------
    # SUBMISSIONS
    # -----------------------------------------------------

    submissions = get_json(
        SUBMISSIONS_URL,
        SUBMISSIONS_PATH,
    )


    # -----------------------------------------------------
    # COMPANY VALIDATION
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("COMPANY")
    print("=" * 72)

    print(
        f"\nEntity name: "
        f"{company_facts.get('entityName')}"
    )

    print(
        f"CIK: "
        f"{company_facts.get('cik')}"
    )


    # -----------------------------------------------------
    # TAXONOMIES
    # -----------------------------------------------------

    facts = (
        company_facts.get(
            "facts"
        )
        or {}
    )

    print("\nAvailable taxonomies:")

    for taxonomy in facts:
        print(
            f"  - {taxonomy}"
        )


    # -----------------------------------------------------
    # US GAAP
    # -----------------------------------------------------

    us_gaap = (
        facts.get(
            "us-gaap"
        )
        or {}
    )

    print(
        f"\nUS-GAAP concepts available: "
        f"{len(us_gaap):,}"
    )


    # -----------------------------------------------------
    # SEARCH IMPORTANT CONCEPTS
    # -----------------------------------------------------

    keywords = [
        "Revenue",
        "OperatingIncome",
        "IncomeLossFromContinuingOperations",
        "NetIncome",
        "OperatingExpense",
        "Fuel",
        "Cash",
        "Debt",
        "InterestExpense",
        "Assets",
        "Liabilities",
        "EarningsPerShare",
    ]


    print("\n" + "=" * 72)
    print("POTENTIALLY USEFUL FINANCIAL CONCEPTS")
    print("=" * 72)


    matched = []


    for concept_name, details in us_gaap.items():

        details = details or {}

        label = (
            details.get("label")
            or ""
        )

        description = (
            details.get("description")
            or ""
        )

        combined = (
            f"{concept_name} "
            f"{label} "
            f"{description}"
        ).lower()


        if any(
            keyword.lower() in combined
            for keyword in keywords
        ):

            matched.append(
                {
                    "concept": concept_name,
                    "label": label,
                }
            )


    matched = sorted(
        matched,
        key=lambda x: x["concept"],
    )


    for item in matched:

        concept = item["concept"]
        label = item["label"]

        print(
            f"{concept:<60} "
            f"{label}"
        )


    print(
        f"\nMatched concepts: "
        f"{len(matched)}"
    )


    # -----------------------------------------------------
    # CHECK HIGH-VALUE CONCEPTS DIRECTLY
    # -----------------------------------------------------

    candidate_concepts = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "OperatingExpenses",
        "CashAndCashEquivalentsAtCarryingValue",
        "Assets",
        "Liabilities",
        "LongTermDebtCurrent",
        "LongTermDebtNoncurrent",
        "InterestExpenseNonOperating",
        "EarningsPerShareDiluted",
    ]


    print("\n" + "=" * 72)
    print("HIGH-VALUE CONCEPT CHECK")
    print("=" * 72)


    for concept in candidate_concepts:

        if concept in us_gaap:

            details = (
                us_gaap.get(concept)
                or {}
            )

            label = (
                details.get("label")
                or ""
            )

            units = (
                details.get("units")
                or {}
            )

            unit_names = list(
                units.keys()
            )

            print(
                f"\nFOUND: {concept}"
            )

            print(
                f"  Label: {label}"
            )

            print(
                f"  Units: {unit_names}"
            )

        else:

            print(
                f"\nNOT FOUND: {concept}"
            )


    # -----------------------------------------------------
    # RECENT FILINGS
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("DELTA 10-K / 10-Q FILINGS SINCE 2020")
    print("=" * 72)


    recent = (
        submissions
        .get("filings", {})
        .get("recent", {})
        or {}
    )


    forms = (
        recent.get("form")
        or []
    )

    filing_dates = (
        recent.get("filingDate")
        or []
    )

    report_dates = (
        recent.get("reportDate")
        or []
    )

    accession_numbers = (
        recent.get("accessionNumber")
        or []
    )

    primary_documents = (
        recent.get("primaryDocument")
        or []
    )


    filing_count = 0


    for (
        form,
        filing_date,
        report_date,
        accession,
        document,
    ) in zip(
        forms,
        filing_dates,
        report_dates,
        accession_numbers,
        primary_documents,
    ):

        if form not in {
            "10-K",
            "10-Q",
        }:
            continue


        if (
            not report_date
            or report_date < "2020-01-01"
        ):
            continue


        filing_count += 1


        print(
            f"\n{form}"
        )

        print(
            f"  Reporting period: "
            f"{report_date}"
        )

        print(
            f"  Filed: "
            f"{filing_date}"
        )

        print(
            f"  Accession: "
            f"{accession}"
        )

        print(
            f"  Document: "
            f"{document}"
        )


    print(
        f"\n10-K / 10-Q filings found: "
        f"{filing_count}"
    )


    # -----------------------------------------------------
    # COMPLETE
    # -----------------------------------------------------

    print("\n" + "=" * 72)
    print("SEC INGESTION COMPLETE")
    print("=" * 72)

    print(
        f"\nCompany facts:\n"
        f"{COMPANY_FACTS_PATH}"
    )

    print(
        f"\nSubmission history:\n"
        f"{SUBMISSIONS_PATH}"
    )


if __name__ == "__main__":
    main()