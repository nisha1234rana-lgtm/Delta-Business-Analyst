from pathlib import Path
import re

import pandas as pd
from rank_bm25 import BM25Okapi


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "sec_chunks.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUTPUT_DIR
    / "bm25_baseline_results.csv"
)


# =========================================================
# VALIDATE INPUT
# =========================================================

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"RAG chunks not found:\n{CHUNKS_PATH}"
    )


# =========================================================
# TOKENIZATION
# =========================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "was",
    "were",
    "be",
    "been",
    "are",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "they",
    "their",
    "what",
    "why",
    "how",
    "did",
    "does",
}


def tokenize(text: str):

    if not isinstance(
        text,
        str,
    ):
        return []

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return [
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and len(token) > 1
        )
    ]


# =========================================================
# LOAD CHUNKS
# =========================================================

print("=" * 78)
print("DELTA BUSINESS ANALYST")
print("BM25 BASELINE RETRIEVAL ENGINE")
print("=" * 78)

chunks = pd.read_parquet(
    CHUNKS_PATH
)

chunks[
    "report_date"
] = (
    chunks[
        "report_date"
    ]
    .astype(str)
)

chunks[
    "year"
] = (
    chunks[
        "report_date"
    ]
    .str[:4]
    .astype(int)
)


def get_quarter(
    report_date: str,
):

    month = int(
        report_date[
            5:7
        ]
    )

    if month <= 3:
        return 1

    if month <= 6:
        return 2

    if month <= 9:
        return 3

    return 4


chunks[
    "quarter"
] = (
    chunks[
        "report_date"
    ]
    .apply(
        get_quarter
    )
)


print(
    f"\nChunks loaded: "
    f"{len(chunks):,}"
)

print(
    f"Documents: "
    f"{chunks['accession'].nunique():,}"
)

print(
    f"Sections: "
    f"{chunks['section'].nunique():,}"
)


# =========================================================
# BUILD BM25 INDEX
# =========================================================

print(
    "\nTokenizing corpus..."
)

tokenized_corpus = [
    tokenize(text)
    for text in chunks[
        "retrieval_text"
    ]
]


print(
    "Building BM25 index..."
)

bm25 = BM25Okapi(
    tokenized_corpus
)


print(
    "BM25 index ready."
)


# =========================================================
# SEARCH
# =========================================================

def search(
    query: str,
    top_k: int = 5,
    form: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    report_date: str | None = None,
    section_contains: str | None = None,
):

    query_tokens = tokenize(
        query
    )

    scores = bm25.get_scores(
        query_tokens
    )

    results = chunks.copy()

    results[
        "bm25_score"
    ] = scores


    # -----------------------------------------------------
    # METADATA FILTERING
    # -----------------------------------------------------

    if form:

        results = results[
            results[
                "form"
            ]
            == form
        ]


    if year is not None:

        results = results[
            results[
                "year"
            ]
            == year
        ]


    if quarter is not None:

        results = results[
            results[
                "quarter"
            ]
            == quarter
        ]


    if report_date:

        results = results[
            results[
                "report_date"
            ]
            == report_date
        ]


    if section_contains:

        results = results[
            results[
                "section"
            ]
            .str.contains(
                section_contains,
                case=False,
                na=False,
            )
        ]


    results = (
        results
        .sort_values(
            "bm25_score",
            ascending=False,
        )
        .head(
            top_k
        )
        .copy()
    )


    return results


# =========================================================
# DISPLAY SEARCH
# =========================================================

def print_results(
    query: str,
    results: pd.DataFrame,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 78
    )


    if results.empty:

        print(
            "\nNo results found."
        )

        return


    for rank, (
        _,
        row,
    ) in enumerate(
        results.iterrows(),
        start=1,
    ):

        text = str(
            row["text"]
        )

        snippet = re.sub(
            r"\s+",
            " ",
            text,
        )[:650]


        print(
            f"\n#{rank}"
        )

        print(
            f"Score: "
            f"{row['bm25_score']:.3f}"
        )

        print(
            f"Form: "
            f"{row['form']}"
        )

        print(
            f"Period: "
            f"{row['report_date']}"
        )

        print(
            f"Section: "
            f"{row['section']}"
        )

        print(
            f"Chunk: "
            f"{row['chunk_id']}"
        )

        print(
            "\n"
            f"{snippet}..."
        )


# =========================================================
# BASELINE TEST QUESTIONS
# =========================================================

TESTS = [

    {
        "query":
            "What did management say about premium travel demand?",

        "top_k":
            5,
    },

    {
        "query":
            "What affected fuel expense and fuel costs?",

        "top_k":
            5,
    },

    {
        "query":
            "What did Delta say about liquidity cash and debt?",

        "top_k":
            5,

        "section_contains":
            "Management",
    },

    {
        "query":
            "What were the main business risks discussed by Delta?",

        "top_k":
            5,

        "section_contains":
            "Risk",
    },

    {
        "query":
            "What did management say about revenue performance in the second quarter of 2026?",

        "top_k":
            5,

        "year":
            2026,

        "quarter":
            2,

        "section_contains":
            "Management",
    },

]


# =========================================================
# RUN TESTS
# =========================================================

saved_results = []


for test_id, test in enumerate(
    TESTS,
    start=1,
):

    query = test[
        "query"
    ]

    results = search(
        query=query,
        top_k=test.get(
            "top_k",
            5,
        ),
        form=test.get(
            "form"
        ),
        year=test.get(
            "year"
        ),
        quarter=test.get(
            "quarter"
        ),
        report_date=test.get(
            "report_date"
        ),
        section_contains=test.get(
            "section_contains"
        ),
    )


    print_results(
        query,
        results,
    )


    for rank, (
        _,
        row,
    ) in enumerate(
        results.iterrows(),
        start=1,
    ):

        saved_results.append(
            {
                "test_id":
                    test_id,

                "query":
                    query,

                "rank":
                    rank,

                "bm25_score":
                    row[
                        "bm25_score"
                    ],

                "chunk_id":
                    row[
                        "chunk_id"
                    ],

                "form":
                    row[
                        "form"
                    ],

                "report_date":
                    row[
                        "report_date"
                    ],

                "section":
                    row[
                        "section"
                    ],

                "source_url":
                    row[
                        "source_url"
                    ],

                "text":
                    row[
                        "text"
                    ],
            }
        )


# =========================================================
# SAVE BASELINE RESULTS
# =========================================================

results_df = pd.DataFrame(
    saved_results
)

results_df.to_csv(
    RESULT_PATH,
    index=False,
)


# =========================================================
# QUICK INDEX STATISTICS
# =========================================================

print(
    "\n"
    + "=" * 78
)

print(
    "BM25 BASELINE SUMMARY"
)

print(
    "=" * 78
)


print(
    f"\nIndexed chunks: "
    f"{len(chunks):,}"
)

print(
    f"Test questions: "
    f"{len(TESTS)}"
)

print(
    f"Retrieved results: "
    f"{len(results_df):,}"
)

print(
    f"\nSaved results:\n"
    f"{RESULT_PATH}"
)


print(
    "\n"
    + "=" * 78
)

print(
    "BM25 RETRIEVAL COMPLETE"
)

print(
    "=" * 78
)