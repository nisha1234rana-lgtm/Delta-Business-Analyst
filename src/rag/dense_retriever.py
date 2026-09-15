from pathlib import Path
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAG_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
)

CHUNKS_PATH = (
    RAG_DIR
    / "sec_chunks.parquet"
)

EMBEDDING_DIR = (
    RAG_DIR
    / "embeddings"
)

RETRIEVAL_DIR = (
    RAG_DIR
    / "retrieval"
)

EMBEDDING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RETRIEVAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDINGS_PATH = (
    EMBEDDING_DIR
    / "sec_chunk_embeddings.npy"
)

RESULT_PATH = (
    RETRIEVAL_DIR
    / "dense_baseline_results.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 78)
print("DELTA BUSINESS ANALYST")
print("DENSE SEMANTIC RETRIEVAL ENGINE")
print("=" * 78)


if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        f"SEC chunks not found:\n{CHUNKS_PATH}"
    )


chunks = pd.read_parquet(
    CHUNKS_PATH
)


chunks["report_date"] = (
    chunks["report_date"]
    .astype(str)
)


chunks["year"] = (
    chunks["report_date"]
    .str[:4]
    .astype(int)
)


def get_quarter(
    report_date: str,
):

    month = int(
        report_date[5:7]
    )

    if month <= 3:
        return 1

    if month <= 6:
        return 2

    if month <= 9:
        return 3

    return 4


chunks["quarter"] = (
    chunks["report_date"]
    .apply(get_quarter)
)


print(
    f"\nChunks loaded: "
    f"{len(chunks):,}"
)


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print(
    f"\nLoading embedding model:\n"
    f"{MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Embedding model ready."
)


# =========================================================
# BUILD OR LOAD EMBEDDINGS
# =========================================================

if EMBEDDINGS_PATH.exists():

    print(
        "\nExisting embeddings found."
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )


    if len(embeddings) != len(chunks):

        print(
            "Embedding count does not match "
            "chunk count. Rebuilding..."
        )

        embeddings = model.encode(
            chunks[
                "retrieval_text"
            ].tolist(),

            batch_size=32,

            show_progress_bar=True,

            normalize_embeddings=True,
        )

        np.save(
            EMBEDDINGS_PATH,
            embeddings,
        )

else:

    print(
        "\nGenerating document embeddings..."
    )

    embeddings = model.encode(
        chunks[
            "retrieval_text"
        ].tolist(),

        batch_size=32,

        show_progress_bar=True,

        normalize_embeddings=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )


print(
    f"\nEmbedding matrix: "
    f"{embeddings.shape}"
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

    # -----------------------------------------------------
    # QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]


    # Since embeddings are normalized,
    # dot product = cosine similarity.
    scores = (
        embeddings
        @ query_embedding
    )


    results = chunks.copy()

    results[
        "dense_score"
    ] = scores


    # -----------------------------------------------------
    # METADATA FILTERS
    # -----------------------------------------------------

    if form:

        results = results[
            results["form"] == form
        ]


    if year is not None:

        results = results[
            results["year"] == year
        ]


    if quarter is not None:

        results = results[
            results["quarter"] == quarter
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
            "dense_score",
            ascending=False,
        )
        .head(top_k)
        .copy()
    )


    return results


# =========================================================
# DISPLAY
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


    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        text = re.sub(
            r"\s+",
            " ",
            str(row["text"]),
        )

        snippet = text[:650]


        print(
            f"\n#{rank}"
        )

        print(
            f"Similarity: "
            f"{row['dense_score']:.4f}"
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
            f"\n{snippet}..."
        )


# =========================================================
# SAME QUESTIONS AS BM25
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
            TOP_K,
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


    for rank, (_, row) in enumerate(
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

                "dense_score":
                    row[
                        "dense_score"
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
# SAVE RESULTS
# =========================================================

results_df = pd.DataFrame(
    saved_results
)

results_df.to_csv(
    RESULT_PATH,
    index=False,
)


# =========================================================
# SUMMARY
# =========================================================

embedding_size_mb = (
    EMBEDDINGS_PATH.stat().st_size
    / (1024 * 1024)
)


print(
    "\n"
    + "=" * 78
)

print(
    "DENSE RETRIEVAL SUMMARY"
)

print(
    "=" * 78
)


print(
    f"\nModel: "
    f"{MODEL_NAME}"
)

print(
    f"Indexed chunks: "
    f"{len(chunks):,}"
)

print(
    f"Embedding dimensions: "
    f"{embeddings.shape[1]:,}"
)

print(
    f"Embedding file size: "
    f"{embedding_size_mb:.2f} MB"
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
    f"\nSaved embeddings:\n"
    f"{EMBEDDINGS_PATH}"
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
    "DENSE RETRIEVAL COMPLETE"
)

print(
    "=" * 78
)