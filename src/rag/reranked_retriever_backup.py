from pathlib import Path
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================================
# CONFIG
# =========================================================

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

BM25_CANDIDATES = 40
DENSE_CANDIDATES = 40

HYBRID_CANDIDATES = 20

FINAL_TOP_K = 5

RRF_K = 60


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
    / "reranked_results.csv"
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


def tokenize(text):

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
# LOAD DATA
# =========================================================

print("=" * 80)
print("DELTA BUSINESS ANALYST")
print("HYBRID + CROSS-ENCODER RERANKING")
print("=" * 80)


chunks = pd.read_parquet(
    CHUNKS_PATH
)

embeddings = np.load(
    EMBEDDINGS_PATH
)


if len(chunks) != len(
    embeddings
):

    raise RuntimeError(
        "Chunk and embedding counts "
        "do not match."
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
    report_date,
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
    .apply(
        get_quarter
    )
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
# LOAD MODELS
# =========================================================

print(
    "\nLoading embedding model..."
)

embedding_model = (
    SentenceTransformer(
        EMBEDDING_MODEL
    )
)


print(
    "Loading cross-encoder reranker..."
)

reranker = CrossEncoder(
    RERANKER_MODEL
)


print(
    "\nRetrieval models ready."
)


# =========================================================
# METADATA FILTERING
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
# SECTION BOOST
# =========================================================

def section_boost(
    query,
    section,
):

    query_lower = (
        query.lower()
    )

    section_lower = (
        str(section)
        .lower()
    )

    boost = 0.0


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

        if (
            "management"
            in section_lower
        ):

            boost += 0.01


    if (
        "risk"
        in query_lower
        or "risks"
        in query_lower
    ):

        if (
            "risk"
            in section_lower
        ):

            boost += 0.02


    return boost


# =========================================================
# HYBRID CANDIDATE GENERATION
# =========================================================

def hybrid_candidates(
    query,
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


    valid_indices = (
        chunks.index[
            mask
        ].tolist()
    )


    if not valid_indices:
        return pd.DataFrame()


    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    query_tokens = tokenize(
        query
    )

    bm25_scores = (
        bm25.get_scores(
            query_tokens
        )
    )


    bm25_ranked = sorted(
        valid_indices,
        key=lambda idx:
            bm25_scores[idx],
        reverse=True,
    )[
        :BM25_CANDIDATES
    ]


    # -----------------------------------------------------
    # DENSE
    # -----------------------------------------------------

    query_embedding = (
        embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
    )


    dense_scores = (
        embeddings
        @ query_embedding
    )


    dense_ranked = sorted(
        valid_indices,
        key=lambda idx:
            dense_scores[idx],
        reverse=True,
    )[
        :DENSE_CANDIDATES
    ]


    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    fusion = {}


    for rank, idx in enumerate(
        bm25_ranked,
        start=1,
    ):

        fusion[idx] = (
            fusion.get(
                idx,
                0.0,
            )
            +
            1 / (
                RRF_K
                + rank
            )
        )


    for rank, idx in enumerate(
        dense_ranked,
        start=1,
    ):

        fusion[idx] = (
            fusion.get(
                idx,
                0.0,
            )
            +
            1 / (
                RRF_K
                + rank
            )
        )


    for idx in fusion:

        fusion[idx] += (
            section_boost(
                query,
                chunks.loc[
                    idx,
                    "section"
                ],
            )
        )


    ranked_indices = sorted(
        fusion,
        key=lambda idx:
            fusion[idx],
        reverse=True,
    )[
        :HYBRID_CANDIDATES
    ]


    candidates = (
        chunks
        .loc[
            ranked_indices
        ]
        .copy()
    )


    candidates[
        "hybrid_score"
    ] = [
        fusion[idx]
        for idx in ranked_indices
    ]


    candidates[
        "bm25_score"
    ] = [
        bm25_scores[idx]
        for idx in ranked_indices
    ]


    candidates[
        "dense_score"
    ] = [
        dense_scores[idx]
        for idx in ranked_indices
    ]


    candidates[
        "hybrid_rank"
    ] = list(
        range(
            1,
            len(candidates) + 1,
        )
    )


    return candidates


# =========================================================
# CROSS-ENCODER RERANKING
# =========================================================

def search(
    query,
    top_k=FINAL_TOP_K,
    form=None,
    year=None,
    quarter=None,
    report_date=None,
    section_contains=None,
):

    candidates = (
        hybrid_candidates(
            query=query,
            form=form,
            year=year,
            quarter=quarter,
            report_date=report_date,
            section_contains=section_contains,
        )
    )


    if candidates.empty:
        return candidates


    # -----------------------------------------------------
    # QUERY / DOCUMENT PAIRS
    # -----------------------------------------------------

    pairs = [
        [
            query,
            text,
        ]

        for text in candidates[
            "retrieval_text"
        ].tolist()
    ]


    # -----------------------------------------------------
    # CROSS-ENCODER
    # -----------------------------------------------------

    reranker_scores = (
        reranker.predict(
            pairs,
            show_progress_bar=False,
        )
    )


    candidates[
        "reranker_score"
    ] = reranker_scores


    # -----------------------------------------------------
    # FINAL ORDER
    # -----------------------------------------------------

    candidates = (
        candidates
        .sort_values(
            "reranker_score",
            ascending=False,
        )
        .head(top_k)
        .copy()
    )


    candidates[
        "final_rank"
    ] = list(
        range(
            1,
            len(candidates) + 1,
        )
    )


    return candidates


# =========================================================
# DISPLAY
# =========================================================

def print_results(
    query,
    results,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 80
    )


    if results.empty:

        print(
            "\nNo results."
        )

        return


    for _, row in (
        results.iterrows()
    ):

        text = re.sub(
            r"\s+",
            " ",
            str(
                row["text"]
            ),
        )


        print(
            f"\n#{int(row['final_rank'])}"
        )

        print(
            f"Reranker score: "
            f"{row['reranker_score']:.4f}"
        )

        print(
            f"Previous hybrid rank: "
            f"{int(row['hybrid_rank'])}"
        )

        print(
            f"Hybrid score: "
            f"{row['hybrid_score']:.5f}"
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
            f"{text[:700]}..."
        )


# =========================================================
# TEST SUITE
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

saved_rows = []


for test_id, test in enumerate(
    TESTS,
    start=1,
):

    query = (
        test["query"]
    )


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


    for _, row in (
        results.iterrows()
    ):

        saved_rows.append(
            {
                "test_id":
                    test_id,

                "query":
                    query,

                "final_rank":
                    row[
                        "final_rank"
                    ],

                "reranker_score":
                    row[
                        "reranker_score"
                    ],

                "hybrid_rank":
                    row[
                        "hybrid_rank"
                    ],

                "hybrid_score":
                    row[
                        "hybrid_score"
                    ],

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
# SAVE
# =========================================================

results_df = pd.DataFrame(
    saved_rows
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
    + "=" * 80
)

print(
    "RERANKING SUMMARY"
)

print(
    "=" * 80
)


print(
    f"\nEmbedding model:"
    f"\n{EMBEDDING_MODEL}"
)

print(
    f"\nReranker:"
    f"\n{RERANKER_MODEL}"
)

print(
    f"\nIndexed chunks: "
    f"{len(chunks):,}"
)

print(
    f"Hybrid candidates/query: "
    f"{HYBRID_CANDIDATES}"
)

print(
    f"Final results/query: "
    f"{FINAL_TOP_K}"
)

print(
    f"Test questions: "
    f"{len(TESTS)}"
)

print(
    f"\nResults saved:\n"
    f"{RESULT_PATH}"
)


print(
    "\n"
    + "=" * 80
)

print(
    "RERANKED RETRIEVAL COMPLETE"
)

print(
    "=" * 80
)