from pathlib import Path
import hashlib
import re

import pandas as pd


# =========================================================
# CONFIG
# =========================================================

TARGET_CHUNK_WORDS = 650
MIN_CHUNK_WORDS = 180
OVERLAP_WORDS = 90


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEC_ROOT = (
    PROJECT_ROOT
    / "documents"
    / "sec_filings"
)

MANIFEST_PATH = (
    SEC_ROOT
    / "filing_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHUNKS_PARQUET = (
    OUTPUT_DIR
    / "sec_chunks.parquet"
)

CHUNKS_CSV = (
    OUTPUT_DIR
    / "sec_chunks_metadata.csv"
)


# =========================================================
# VALIDATE INPUT
# =========================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"SEC filing manifest not found:\n"
        f"{MANIFEST_PATH}"
    )

manifest = pd.read_csv(
    MANIFEST_PATH
)


print("=" * 78)
print("DELTA BUSINESS ANALYST")
print("SECTION-AWARE SEC DOCUMENT CHUNKING")
print("=" * 78)

print(
    f"\nDocuments in manifest: "
    f"{len(manifest)}"
)


# =========================================================
# SECTION MAPS
# =========================================================

TEN_K_SECTIONS = {

    "1":
        "Business",

    "1A":
        "Risk Factors",

    "1B":
        "Unresolved Staff Comments",

    "1C":
        "Cybersecurity",

    "2":
        "Properties",

    "3":
        "Legal Proceedings",

    "4":
        "Mine Safety Disclosures",

    "5":
        "Market for Registrant Securities",

    "6":
        "Reserved",

    "7":
        "Management Discussion and Analysis",

    "7A":
        "Quantitative and Qualitative Disclosures About Market Risk",

    "8":
        "Financial Statements and Supplementary Data",

    "9":
        "Changes in and Disagreements with Accountants",

    "9A":
        "Controls and Procedures",

    "9B":
        "Other Information",

    "9C":
        "Foreign Jurisdictions Preventing Inspections",

    "10":
        "Directors Executive Officers and Corporate Governance",

    "11":
        "Executive Compensation",

    "12":
        "Security Ownership",

    "13":
        "Certain Relationships and Related Transactions",

    "14":
        "Principal Accountant Fees and Services",

    "15":
        "Exhibits and Financial Statement Schedules",

    "16":
        "Form 10-K Summary",
}


TEN_Q_SECTIONS = {

    "1":
        "Financial Statements",

    "2":
        "Management Discussion and Analysis",

    "3":
        "Quantitative and Qualitative Disclosures About Market Risk",

    "4":
        "Controls and Procedures",

    "1A":
        "Risk Factors",
}


# =========================================================
# TEXT CLEANING
# =========================================================

def normalize_text(
    text: str,
) -> str:

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# SECTION HEADING DETECTION
# =========================================================

ITEM_PATTERN = re.compile(
    r"""
    ^\s*
    ITEM
    \s+
    (
        1A
        |1B
        |1C
        |7A
        |9A
        |9B
        |9C
        |1[0-6]
        |[1-9]
    )
    [\.\:\-\s]*
    (.*?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def section_name_from_item(
    form: str,
    item: str,
    heading_text: str,
):

    item = (
        item
        .upper()
        .strip()
    )

    heading_text = (
        heading_text
        .strip()
        .rstrip(".:")
    )

    if form == "10-K":

        mapped = (
            TEN_K_SECTIONS
            .get(item)
        )

    else:

        mapped = (
            TEN_Q_SECTIONS
            .get(item)
        )


    if mapped:
        return mapped

    if heading_text:
        return heading_text

    return f"Item {item}"


# =========================================================
# SPLIT DOCUMENT INTO SECTIONS
# =========================================================

def detect_sections(
    text: str,
    form: str,
):

    lines = text.splitlines()

    sections = []

    current_section = (
        "Document Introduction"
    )

    current_item = None
    current_lines = []


    for line in lines:

        stripped = (
            line.strip()
        )

        match = (
            ITEM_PATTERN.match(
                stripped
            )
        )


        if match:

            # Save previous section
            if current_lines:

                body = "\n".join(
                    current_lines
                ).strip()

                if body:

                    sections.append(
                        {
                            "section":
                                current_section,

                            "item":
                                current_item,

                            "text":
                                body,
                        }
                    )


            item = (
                match.group(1)
                .upper()
            )

            heading_text = (
                match.group(2)
                or ""
            )


            current_section = (
                section_name_from_item(
                    form,
                    item,
                    heading_text,
                )
            )

            current_item = (
                f"Item {item}"
            )

            current_lines = [
                stripped
            ]

        else:

            current_lines.append(
                line
            )


    if current_lines:

        body = "\n".join(
            current_lines
        ).strip()

        if body:

            sections.append(
                {
                    "section":
                        current_section,

                    "item":
                        current_item,

                    "text":
                        body,
                }
            )


    return sections


# =========================================================
# PARAGRAPH SPLITTING
# =========================================================

def split_paragraphs(
    text: str,
):

    raw_paragraphs = re.split(
        r"\n{2,}",
        text,
    )

    paragraphs = []

    for paragraph in raw_paragraphs:

        paragraph = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if not paragraph:
            continue

        paragraphs.append(
            paragraph
        )

    return paragraphs


# =========================================================
# WORD-LEVEL FALLBACK SPLIT
# =========================================================

def split_long_paragraph(
    paragraph: str,
):

    words = paragraph.split()

    pieces = []

    start = 0

    while start < len(words):

        end = min(
            start + TARGET_CHUNK_WORDS,
            len(words),
        )

        piece = " ".join(
            words[start:end]
        )

        pieces.append(
            piece
        )

        if end == len(words):
            break

        start = max(
            end - OVERLAP_WORDS,
            start + 1,
        )

    return pieces


# =========================================================
# SECTION-AWARE CHUNKING
# =========================================================

def chunk_section(
    text: str,
):

    paragraphs = split_paragraphs(
        text
    )

    expanded = []

    for paragraph in paragraphs:

        word_count = len(
            paragraph.split()
        )

        if word_count > (
            TARGET_CHUNK_WORDS
            * 1.5
        ):

            expanded.extend(
                split_long_paragraph(
                    paragraph
                )
            )

        else:

            expanded.append(
                paragraph
            )


    chunks = []

    current = []
    current_words = 0


    for paragraph in expanded:

        paragraph_words = len(
            paragraph.split()
        )


        if (
            current
            and (
                current_words
                + paragraph_words
                > TARGET_CHUNK_WORDS
            )
        ):

            chunk_text = "\n\n".join(
                current
            )

            chunks.append(
                chunk_text
            )


            # Build overlap from previous chunk
            overlap_words = (
                chunk_text
                .split()[
                    -OVERLAP_WORDS:
                ]
            )

            overlap_text = " ".join(
                overlap_words
            )

            current = [
                overlap_text
            ]

            current_words = len(
                overlap_words
            )


        current.append(
            paragraph
        )

        current_words += (
            paragraph_words
        )


    if current:

        chunk_text = "\n\n".join(
            current
        )

        if (
            len(
                chunk_text.split()
            )
            >= MIN_CHUNK_WORDS
            or not chunks
        ):

            chunks.append(
                chunk_text
            )

        else:

            # Append very small ending
            # to previous chunk.
            chunks[-1] = (
                chunks[-1]
                + "\n\n"
                + chunk_text
            )


    return chunks


# =========================================================
# CHUNK ID
# =========================================================

def make_chunk_id(
    accession: str,
    section: str,
    chunk_number: int,
):

    raw = (
        f"{accession}|"
        f"{section}|"
        f"{chunk_number}"
    )

    digest = (
        hashlib.sha1(
            raw.encode(
                "utf-8"
            )
        )
        .hexdigest()[:12]
    )

    return (
        f"sec_{digest}"
    )


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

chunk_rows = []

document_stats = []


for document_index, row in (
    manifest.iterrows()
):

    form = row["form"]

    report_date = str(
        row["report_date"]
    )

    filing_date = str(
        row["filing_date"]
    )

    accession = str(
        row["accession"]
    )

    source_url = str(
        row["source_url"]
    )

    text_path = Path(
        row["clean_text_path"]
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        f"[{document_index + 1}/"
        f"{len(manifest)}] "
        f"{form} | "
        f"{report_date}"
    )

    print(
        "=" * 78
    )


    if not text_path.exists():

        print(
            "WARNING: text file missing."
        )

        continue


    text = text_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    text = normalize_text(
        text
    )


    sections = detect_sections(
        text,
        form,
    )


    document_chunk_count = 0


    for section_index, section in enumerate(
        sections,
        start=1,
    ):

        section_name = (
            section["section"]
        )

        item = (
            section["item"]
        )

        section_text = (
            section["text"]
        )


        chunks = chunk_section(
            section_text
        )


        for chunk_number, chunk_text in enumerate(
            chunks,
            start=1,
        ):

            word_count = len(
                chunk_text.split()
            )

            if word_count < 40:
                continue


            chunk_id = make_chunk_id(
                accession,
                section_name,
                chunk_number,
            )


            # Context header will later help
            # both embeddings and the LLM.
            retrieval_text = (
                f"Company: Delta Air Lines, Inc.\n"
                f"Form: {form}\n"
                f"Reporting period: {report_date}\n"
                f"Section: {section_name}\n\n"
                f"{chunk_text}"
            )


            chunk_rows.append(
                {
                    "chunk_id":
                        chunk_id,

                    "company":
                        "Delta Air Lines, Inc.",

                    "form":
                        form,

                    "report_date":
                        report_date,

                    "filing_date":
                        filing_date,

                    "accession":
                        accession,

                    "source_url":
                        source_url,

                    "item":
                        item,

                    "section":
                        section_name,

                    "section_index":
                        section_index,

                    "chunk_number":
                        chunk_number,

                    "word_count":
                        word_count,

                    "text":
                        chunk_text,

                    "retrieval_text":
                        retrieval_text,
                }
            )


            document_chunk_count += 1


    document_stats.append(
        {
            "report_date":
                report_date,

            "form":
                form,

            "sections":
                len(sections),

            "chunks":
                document_chunk_count,
        }
    )


    print(
        f"Sections detected: "
        f"{len(sections)}"
    )

    print(
        f"Chunks created: "
        f"{document_chunk_count}"
    )


# =========================================================
# CREATE DATAFRAME
# =========================================================

chunks_df = pd.DataFrame(
    chunk_rows
)

stats_df = pd.DataFrame(
    document_stats
)


if chunks_df.empty:
    raise RuntimeError(
        "No document chunks were created."
    )


# =========================================================
# SAVE
# =========================================================

chunks_df.to_parquet(
    CHUNKS_PARQUET,
    index=False,
)

chunks_df[
    [
        "chunk_id",
        "company",
        "form",
        "report_date",
        "filing_date",
        "accession",
        "item",
        "section",
        "section_index",
        "chunk_number",
        "word_count",
        "source_url",
    ]
].to_csv(
    CHUNKS_CSV,
    index=False,
)


# =========================================================
# VALIDATION
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "RAG CHUNKING SUMMARY"
)

print(
    "=" * 78
)


print(
    f"\nDocuments processed: "
    f"{len(stats_df):,}"
)

print(
    f"Total chunks: "
    f"{len(chunks_df):,}"
)

print(
    f"Average words/chunk: "
    f"{chunks_df['word_count'].mean():,.0f}"
)

print(
    f"Median words/chunk: "
    f"{chunks_df['word_count'].median():,.0f}"
)

print(
    f"Smallest chunk: "
    f"{chunks_df['word_count'].min():,} words"
)

print(
    f"Largest chunk: "
    f"{chunks_df['word_count'].max():,} words"
)


print(
    "\nChunks by filing type:"
)

print(
    chunks_df[
        "form"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nMost common detected sections:"
)

section_counts = (
    chunks_df[
        "section"
    ]
    .value_counts()
    .head(15)
)

print(
    section_counts.to_string()
)


print(
    "\nChunks per document:"
)

print(
    stats_df.to_string(
        index=False
    )
)


# =========================================================
# SAMPLE CHUNKS
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "SAMPLE RAG CHUNKS"
)

print(
    "=" * 78
)


sample_sections = [
    "Management Discussion and Analysis",
    "Risk Factors",
    "Financial Statements",
]


for target_section in sample_sections:

    sample = chunks_df[
        chunks_df[
            "section"
        ]
        == target_section
    ]

    if sample.empty:
        continue


    example = sample.iloc[-1]

    print(
        f"\nSECTION: "
        f"{example['section']}"
    )

    print(
        f"FORM: "
        f"{example['form']}"
    )

    print(
        f"PERIOD: "
        f"{example['report_date']}"
    )

    print(
        f"WORDS: "
        f"{example['word_count']}"
    )

    print(
        "\n"
        + example[
            "text"
        ][:700]
        + "..."
    )


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "SECTION-AWARE CHUNKING COMPLETE"
)

print(
    "=" * 78
)

print(
    f"\nChunk data:\n"
    f"{CHUNKS_PARQUET}"
)

print(
    f"\nMetadata:\n"
    f"{CHUNKS_CSV}"
)