from pathlib import Path
import json
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =========================================================
# CONFIG
# =========================================================

START_DATE = "2020-01-01"

CIK_PADDED = "0000027904"
CIK_ARCHIVE = "27904"

# IMPORTANT:
# SEC requires a declared automated-tool identity
# with a real contact email.
CONTACT_EMAIL = "nisha1234rana@gmail.com"

REQUEST_DELAY = 1.0
TIMEOUT = 90
MAX_RETRIES = 4


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUBMISSIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sec"
    / "delta_submissions.json"
)

DOCUMENT_ROOT = (
    PROJECT_ROOT
    / "documents"
    / "sec_filings"
)

RAW_HTML_DIR = (
    DOCUMENT_ROOT
    / "raw_html"
)

CLEAN_TEXT_DIR = (
    DOCUMENT_ROOT
    / "clean_text"
)

MANIFEST_PATH = (
    DOCUMENT_ROOT
    / "filing_manifest.csv"
)

RAW_HTML_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLEAN_TEXT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# VALIDATE CONTACT EMAIL
# =========================================================

if (
    "YOUR_EMAIL" in CONTACT_EMAIL
    or "@" not in CONTACT_EMAIL
):
    raise ValueError(
        "\nSet CONTACT_EMAIL near the top of this script "
        "to your real email address before running.\n"
        "SEC requires automated tools to identify themselves."
    )


# =========================================================
# SEC HTTP SESSION
# =========================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "DeltaBusinessAnalyst/1.0 "
            f"{CONTACT_EMAIL}"
        ),
        "Accept-Encoding": "gzip, deflate",
        "Accept": (
            "text/html,"
            "application/xhtml+xml,"
            "application/xml;q=0.9,"
            "*/*;q=0.8"
        ),
        "Host": "www.sec.gov",
        "Connection": "keep-alive",
    }
)


# =========================================================
# LOAD SUBMISSIONS
# =========================================================

if not SUBMISSIONS_PATH.exists():
    raise FileNotFoundError(
        f"SEC submission file not found:\n"
        f"{SUBMISSIONS_PATH}"
    )

with open(
    SUBMISSIONS_PATH,
    "r",
    encoding="utf-8",
) as file:
    submissions = json.load(file)


# =========================================================
# CLEAN HTML
# =========================================================

def html_to_text(html: str) -> str:

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        separator="\n"
    )

    text = text.replace(
        "\xa0",
        " "
    )

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if not line:
            continue

        lines.append(
            line
        )

    text = "\n".join(
        lines
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# SEC DOWNLOAD FUNCTION
# =========================================================

def download_sec_html(
    url: str,
) -> str:

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            print(
                f"Request attempt "
                f"{attempt}/{MAX_RETRIES}"
            )

            response = SESSION.get(
                url,
                timeout=TIMEOUT,
            )

            print(
                f"HTTP status: "
                f"{response.status_code}"
            )

            if response.status_code == 403:

                last_error = RuntimeError(
                    "SEC returned HTTP 403."
                )

                if attempt < MAX_RETRIES:

                    wait_seconds = (
                        attempt * 5
                    )

                    print(
                        f"SEC temporarily rejected "
                        f"the request. Waiting "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue

            response.raise_for_status()

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    ""
                )
                .lower()
            )

            if (
                "text" not in content_type
                and "html" not in content_type
                and "xml" not in content_type
            ):
                print(
                    "Warning: unexpected "
                    f"content type: "
                    f"{content_type}"
                )

            return response.text


        except requests.RequestException as exc:

            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = (
                attempt * 5
            )

            print(
                f"Request failed: {exc}"
            )

            print(
                f"Waiting "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )


    raise RuntimeError(
        "\nUnable to download SEC filing.\n"
        f"URL: {url}\n"
        f"Last error: {last_error}\n\n"
        "If SEC continues returning 403, "
        "stop running the script for several minutes "
        "before trying again."
    )


# =========================================================
# BUILD FILING LIST
# =========================================================

recent = (
    submissions
    .get("filings", {})
    .get("recent", {})
)

forms = recent.get(
    "form",
    []
)

filing_dates = recent.get(
    "filingDate",
    []
)

report_dates = recent.get(
    "reportDate",
    []
)

accessions = recent.get(
    "accessionNumber",
    []
)

primary_documents = recent.get(
    "primaryDocument",
    []
)


filings = []


for (
    form,
    filing_date,
    report_date,
    accession,
    primary_document,
) in zip(
    forms,
    filing_dates,
    report_dates,
    accessions,
    primary_documents,
):

    if form not in {
        "10-K",
        "10-Q",
    }:
        continue

    if not report_date:
        continue

    if report_date < START_DATE:
        continue

    filings.append(
        {
            "form": form,
            "filing_date": filing_date,
            "report_date": report_date,
            "accession": accession,
            "primary_document": primary_document,
        }
    )


filings = sorted(
    filings,
    key=lambda x: x["report_date"],
)


# =========================================================
# START
# =========================================================

print("=" * 76)
print("DELTA BUSINESS ANALYST")
print("SEC FILING DOCUMENT INGESTION")
print("=" * 76)

print(
    f"\nDeclared SEC contact: "
    f"{CONTACT_EMAIL}"
)

print(
    f"Filings found: "
    f"{len(filings)}"
)

print(
    f"Period: "
    f"{filings[0]['report_date']} "
    f"→ "
    f"{filings[-1]['report_date']}"
)


manifest_rows = []


# =========================================================
# PROCESS FILINGS
# =========================================================

for index, filing in enumerate(
    filings,
    start=1,
):

    form = filing["form"]

    report_date = (
        filing["report_date"]
    )

    filing_date = (
        filing["filing_date"]
    )

    accession = (
        filing["accession"]
    )

    document = (
        filing["primary_document"]
    )


    accession_clean = (
        accession.replace(
            "-",
            "",
        )
    )


    source_url = (
        "https://www.sec.gov/"
        "Archives/edgar/data/"
        f"{CIK_ARCHIVE}/"
        f"{accession_clean}/"
        f"{document}"
    )


    safe_form = (
        form.replace(
            "-",
            ""
        )
    )


    base_name = (
        f"{report_date}_"
        f"{safe_form}_"
        f"{accession_clean}"
    )


    html_path = (
        RAW_HTML_DIR
        / f"{base_name}.html"
    )

    text_path = (
        CLEAN_TEXT_DIR
        / f"{base_name}.txt"
    )


    print(
        "\n"
        + "=" * 76
    )

    print(
        f"[{index}/{len(filings)}] "
        f"{form} | {report_date}"
    )

    print(
        "=" * 76
    )


    # -----------------------------------------------------
    # HTML
    # -----------------------------------------------------

    if html_path.exists():

        print(
            "Raw HTML already exists."
        )

        html = (
            html_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    else:

        print(
            "Downloading SEC filing..."
        )

        print(
            source_url
        )

        html = download_sec_html(
            source_url
        )

        html_path.write_text(
            html,
            encoding="utf-8",
        )

        time.sleep(
            REQUEST_DELAY
        )


    # -----------------------------------------------------
    # CLEAN TEXT
    # -----------------------------------------------------

    if text_path.exists():

        print(
            "Clean text already exists."
        )

        clean_text = (
            text_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    else:

        print(
            "Extracting filing text..."
        )

        clean_text = html_to_text(
            html
        )

        text_path.write_text(
            clean_text,
            encoding="utf-8",
        )


    # -----------------------------------------------------
    # STATS
    # -----------------------------------------------------

    word_count = len(
        clean_text.split()
    )

    character_count = len(
        clean_text
    )

    html_size_mb = (
        html_path.stat().st_size
        / (1024 * 1024)
    )

    text_size_mb = (
        text_path.stat().st_size
        / (1024 * 1024)
    )


    print(
        f"Words: "
        f"{word_count:,}"
    )

    print(
        f"Clean text size: "
        f"{text_size_mb:.2f} MB"
    )


    manifest_rows.append(
        {
            "company":
                "Delta Air Lines, Inc.",

            "cik":
                CIK_PADDED,

            "form":
                form,

            "report_date":
                report_date,

            "filing_date":
                filing_date,

            "accession":
                accession,

            "primary_document":
                document,

            "source_url":
                source_url,

            "raw_html_path":
                str(html_path),

            "clean_text_path":
                str(text_path),

            "word_count":
                word_count,

            "character_count":
                character_count,

            "html_size_mb":
                round(
                    html_size_mb,
                    3,
                ),

            "text_size_mb":
                round(
                    text_size_mb,
                    3,
                ),
        }
    )


# =========================================================
# SAVE MANIFEST
# =========================================================

manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
)


# =========================================================
# CORPUS SUMMARY
# =========================================================

print(
    "\n"
    + "=" * 76
)

print(
    "DOCUMENT CORPUS SUMMARY"
)

print(
    "=" * 76
)


print(
    f"\nDocuments: "
    f"{len(manifest):,}"
)

print(
    f"10-K filings: "
    f"{(manifest['form'] == '10-K').sum():,}"
)

print(
    f"10-Q filings: "
    f"{(manifest['form'] == '10-Q').sum():,}"
)

print(
    f"Total words: "
    f"{manifest['word_count'].sum():,}"
)

print(
    f"Average words/document: "
    f"{manifest['word_count'].mean():,.0f}"
)


print(
    "\nLargest documents:"
)

largest = (
    manifest[
        [
            "report_date",
            "form",
            "word_count",
            "text_size_mb",
        ]
    ]
    .sort_values(
        "word_count",
        ascending=False,
    )
    .head(10)
)

print(
    largest.to_string(
        index=False
    )
)


# =========================================================
# LATEST DOCUMENT SANITY CHECK
# =========================================================

latest = manifest.iloc[-1]

latest_text = Path(
    latest[
        "clean_text_path"
    ]
).read_text(
    encoding="utf-8",
    errors="ignore",
)


checks = {

    "Delta Air Lines":
        "DELTA AIR LINES"
        in latest_text.upper(),

    "Management Discussion":
        "MANAGEMENT"
        in latest_text.upper(),

    "Financial Statements":
        "FINANCIAL STATEMENTS"
        in latest_text.upper(),

    "Risk Factors":
        "RISK FACTORS"
        in latest_text.upper(),
}


print(
    "\nLatest-document sanity checks:"
)

for check, passed in (
    checks.items()
):

    print(
        f"  {check:<25} "
        f"{'PASS' if passed else 'CHECK'}"
    )


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n"
    + "=" * 76
)

print(
    "SEC DOCUMENT INGESTION COMPLETE"
)

print(
    "=" * 76
)


print(
    f"\nManifest:\n"
    f"{MANIFEST_PATH}"
)

print(
    f"\nRaw HTML:\n"
    f"{RAW_HTML_DIR}"
)

print(
    f"\nClean text:\n"
    f"{CLEAN_TEXT_DIR}"
)