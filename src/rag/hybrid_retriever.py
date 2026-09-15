from pathlib import Path
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BM25_CANDIDATES = 30
DENSE_CANDIDATES = 30

RRF_K = 60

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

EMBEDDINGS_PATH = (
    RAG_DIR
    / "embeddings"
    / "sec_chunk_embeddings.npy"
)

OUTPUT_DIR = (
    RAG_DIR
    / "retrieval"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_PATH = (
    OUTPUT_DIR
    / "hybrid_results.csv"
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

    if not isinstance(text, str):
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
# LOAD DATA
# =========================================================

print("=" * 78)
print("DELTA BUSINESS ANALYST")
print("HYBRID RETRIEVAL ENGINE")
print("=" * 78)


chunks = pd.read_parquet(
    CHUNKS_PATH
)

embeddings = np.load(
    EMBEDDINGS_PATH
)


if len(chunks) != len(embeddings):

    raise RuntimeError(
        "Chunk count and embedding count do not match."
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
# BM25
# =========================================================

print(
    "\nBuilding BM25 index..."
)

tokenized_corpus = [
    tokenize(text)
    for text in chunks[
        "retrieval_text"
    ]
]

bm25 = BM25Okapi(
    tokenized_corpus
)


# =========================================================
# DENSE MODEL
# =========================================================

print(
    "Loading dense embedding model..."
)

model = SentenceTransformer(
    MODEL_NAME
)

print(
    "Hybrid search components ready."
)


# =========================================================
# METADATA FILTER
# =========================================================

def metadata_mask(
    form=None,
    year=None,
    quarter=None,
    report_date=None,
    section_contains=None,
):

    mask = pd.Series(
        True,
        index=chunks.index,
    )

    if form:

        mask &= (
            chunks["form"]
            == form
        )

    if year is not None:

        mask &= (
            chunks["year"]
            == year
        )

    if quarter is not None:

        mask &= (
            chunks["quarter"]
            == quarter
        )

    if report_date:

        mask &= (
            chunks["report_date"]
            == report_date
        )

    if section_contains:

        mask &= (
            chunks["section"]
            .str.contains(
                section_contains,
                case=False,
                na=False,
            )
        )

    return mask


# =========================================================
# QUERY INTENT / SECTION BOOST
# =========================================================

def section_boost(
    query: str,
    section: str,
):

    query_lower = query.lower()
    section_lower = str(
        section
    ).lower()

    boost = 0.0


    # Management / operating performance
    management_terms = [
        "management",
        "revenue",
        "expense",
        "cost",
        "fuel",
        "margin",
        "profit",
        "demand",
        "travel",
        "liquidity",
        "cash",
        "debt",
        "performance",
    ]

    if any(
        term in query_lower
        for term in management_terms
    ):

        if "management" in section_lower:
            boost += 0.015


    # Risk questions
    risk_terms = [
        "risk",
        "risks",
        "threat",
        "uncertainty",
    ]

    if any(
        term in query_lower
        for term in risk_terms
    ):

        if "risk" in section_lower:
            boost += 0.025


    # Financial statement questions
    financial_terms = [
        "balance sheet",
        "financial statement",
        "assets",
        "liabilities",
    ]

    if any(
        term in query_lower
        for term in financial_terms
    ):

        if "financial" in section_lower:
            boost += 0.015


    return boost


# =========================================================
# HYBRID SEARCH
# =========================================================

def search(
    query: str,
    top_k: int = TOP_K,
    form=None,
    year=None,
    quarter=None,
    report_date=None,
    section_contains=None,
):

    mask = metadata_mask(
        form=form,
        year=year,
        quarter=quarter,
        report_date=report_date,
        section_contains=section_contains,
    )


    candidate_indices = (
        chunks.index[
            mask
        ]
        .tolist()
    )


    if not candidate_indices:

        return pd.DataFrame()


    # =====================================================
    # BM25 SEARCH
    # =====================================================

    query_tokens = tokenize(
        query
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )


    bm25_ranked = sorted(
        candidate_indices,
        key=lambda idx: (
            bm25_scores[idx]
        ),
        reverse=True,
    )[
        :BM25_CANDIDATES
    ]


    # =====================================================
    # DENSE SEARCH
    # =====================================================

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]


    dense_scores = (
        embeddings
        @ query_embedding
    )


    dense_ranked = sorted(
        candidate_indices,
        key=lambda idx: (
            dense_scores[idx]
        ),
        reverse=True,
    )[
        :DENSE_CANDIDATES
    ]


    # =====================================================
    # RECIPROCAL RANK FUSION
    # =====================================================

    fusion_scores = {}


    for rank, idx in enumerate(
        bm25_ranked,
        start=1,
    ):

        fusion_scores[
            idx
        ] = (
            fusion_scores.get(
                idx,
                0.0,
            )
            +
            1.0 / (
                RRF_K + rank
            )
        )


    for rank, idx in enumerate(
        dense_ranked,
        start=1,
    ):

        fusion_scores[
            idx
        ] = (
            fusion_scores.get(
                idx,
                0.0,
            )
            +
            1.0 / (
                RRF_K + rank
            )
        )


    # =====================================================
    # SECTION RELEVANCE BOOST
    # =====================================================

    for idx in fusion_scores:

        fusion_scores[
            idx
        ] += section_boost(
            query,
            chunks.loc[
                idx,
                "section"
            ],
        )


    # =====================================================
    # FINAL RANKING
    # =====================================================

    final_indices = sorted(
        fusion_scores.keys(),
        key=lambda idx: (
            fusion_scores[idx]
        ),
        reverse=True,
    )[
        :top_k
    ]


    results = (
        chunks
        .loc[
            final_indices
        ]
        .copy()
    )


    results[
        "hybrid_score"
    ] = [
        fusion_scores[idx]
        for idx in final_indices
    ]


    results[
        "bm25_score"
    ] = [
        bm25_scores[idx]
        for idx in final_indices
    ]


    results[
        "dense_score"
    ] = [
        dense_scores[idx]
        for idx in final_indices
    ]


    results[
        "bm25_rank"
    ] = [
        (
            bm25_ranked.index(idx)
            + 1
        )
        if idx in bm25_ranked
        else None

        for idx in final_indices
    ]


    results[
        "dense_rank"
    ] = [
        (
            dense_ranked.index(idx)
            + 1
        )
        if idx in dense_ranked
        else None

        for idx in final_indices
    ]


    return results


# =========================================================
# DISPLAY
# =========================================================

def print_results(
    query,
    results,
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
            "\nNo results."
        )

        return


    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        text = re.sub(
            r"\s+",
            " ",
            str(
                row["text"]
            ),
        )


        print(
            f"\n#{rank}"
        )

        print(
            f"Hybrid score: "
            f"{row['hybrid_score']:.5f}"
        )

        print(
            f"BM25 rank: "
            f"{row['bm25_rank']}"
        )

        print(
            f"Dense rank: "
            f"{row['dense_rank']}"
        )

        print(
            f"Dense similarity: "
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
            "\n"
            f"{text[:650]}..."
        )


# =========================================================
# SAME TEST SUITE
# =========================================================

TESTS = [

    {
        "query":
            "What did management say about premium travel demand?",
    },

    {
        "query":
            "What affected fuel expense and fuel costs?",
    },

    {
        "query":
            "What did Delta say about liquidity cash and debt?",

        "section_contains":
            "Management",
    },

    {
        "query":
            "What were the main business risks discussed by Delta?",

        "section_contains":
            "Risk",
    },

    {
        "query":
            "What did management say about revenue performance in the second quarter of 2026?",

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

        top_k=5,

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

                "hybrid_score":
                    row[
                        "hybrid_score"
                    ],

                "bm25_score":
                    row[
                        "bm25_score"
                    ],

                "dense_score":
                    row[
                        "dense_score"
                    ],

                "bm25_rank":
                    row[
                        "bm25_rank"
                    ],

                "dense_rank":
                    row[
                        "dense_rank"
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
# SAVE
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

print(
    "\n"
    + "=" * 78
)

print(
    "HYBRID RETRIEVAL SUMMARY"
)

print(
    "=" * 78
)


print(
    f"\nIndexed chunks: "
    f"{len(chunks):,}"
)

print(
    f"BM25 candidate pool: "
    f"{BM25_CANDIDATES}"
)

print(
    f"Dense candidate pool: "
    f"{DENSE_CANDIDATES}"
)

print(
    f"Fusion method: "
    f"Reciprocal Rank Fusion"
)

print(
    f"RRF constant: "
    f"{RRF_K}"
)

print(
    f"Test questions: "
    f"{len(TESTS)}"
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
    "HYBRID RETRIEVAL COMPLETE"
)

print(
    "=" * 78
)