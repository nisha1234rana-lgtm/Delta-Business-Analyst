from pathlib import Path
import re

import pandas as pd
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ROUTES = [
    "SQL",
    "RAG",
    "HYBRID",
]


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "routing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "router_test_results.csv"
)


# =========================================================
# VOCABULARY
# =========================================================

STRUCTURED_TERMS = [

    # Operations
    "flight",
    "flights",
    "airport",
    "airports",
    "route",
    "routes",
    "delay",
    "delays",
    "departure delay",
    "arrival delay",
    "on-time",
    "on time",
    "cancellation",
    "cancellations",
    "cancellation rate",
    "diversion",
    "diverted",
    "operator",
    "operating carrier",
    "carrier",
    "mainline",
    "partner airline",
    "regional carrier",
    "operational performance",

    # Finance
    "revenue",
    "operating revenue",
    "operating income",
    "net income",
    "operating margin",
    "net margin",
    "fuel cost",
    "fuel expense",
    "cash",
    "debt",
    "assets",
    "profit",
    "expense",

    # Analytical language
    "average",
    "rate",
    "percentage",
    "percent",
    "highest",
    "lowest",
    "largest",
    "smallest",
    "worst",
    "best",
    "trend",
    "increase",
    "decrease",
    "change",
    "compare",
    "comparison",
    "how many",
    "how much",
]


# Strong phrases that explicitly request document evidence
DOCUMENT_REQUEST_PATTERNS = [

    # -----------------------------------------------------
    # MANAGEMENT / DELTA COMMENTARY
    # -----------------------------------------------------

    r"\bwhat did management say\b",
    r"\bwhat did delta management say\b",
    r"\bwhat did delta say\b",

    r"\bwhat does management say\b",
    r"\bwhat does delta management say\b",
    r"\bwhat does delta say\b",

    r"\bdid management (?:say|mention|discuss|explain)\b",
    r"\bdid delta management (?:say|mention|discuss|explain)\b",
    r"\bdid delta (?:say|mention|discuss|explain)\b",

    r"\bmanagement (?:said|stated|mentioned|discussed|explained)\b",
    r"\bdelta management (?:said|stated|mentioned|discussed|explained)\b",
    r"\bdelta (?:said|stated|mentioned|discussed|explained)\b",

    r"\baccording to management\b",
    r"\baccording to delta management\b",
    r"\baccording to delta\b",

    # -----------------------------------------------------
    # MANAGEMENT EXPLANATION REQUESTS
    # -----------------------------------------------------

    r"\bwhat explanation did management provide\b",
    r"\bwhat explanation did delta management provide\b",
    r"\bwhat explanation did delta provide\b",

    r"\bwhat explanation does management provide\b",
    r"\bwhat explanation does delta management provide\b",
    r"\bwhat explanation does delta provide\b",

    r"\bwhat did management explain\b",
    r"\bwhat did delta management explain\b",
    r"\bwhat did delta explain\b",

    r"\bhow did management explain\b",
    r"\bhow did delta management explain\b",
    r"\bhow did delta explain\b",

    r"\bmanagement(?:'s)? explanation\b",
    r"\bdelta management(?:'s)? explanation\b",

    # -----------------------------------------------------
    # FILINGS / DOCUMENTS
    # -----------------------------------------------------

    r"\bin the filing\b",
    r"\bin its filing\b",
    r"\bin delta(?:'s)? filing\b",

    r"\b10-k\b",
    r"\b10-q\b",

    r"\brisk factors?\b",
    r"\bmanagement discussion\b",
    r"\bmd&a\b",
]


EXPLANATION_PATTERNS = [

    r"\bwhy\b",

    r"\bwhat caused\b",
    r"\bwhat drove\b",
    r"\bwhat affected\b",

    r"\breason\b",
    r"\breasons\b",

    r"\bdriver\b",
    r"\bdrivers\b",

    r"\bexplain\b",
    r"\bexplained\b",
]


OPERATIONAL_TERMS = [
    "flight",
    "flights",
    "airport",
    "airports",
    "route",
    "routes",
    "delay",
    "delays",
    "on-time",
    "on time",
    "cancel",
    "cancellation",
    "divert",
    "operator",
    "operating carrier",
    "carrier",
    "mainline",
    "partner airline",
    "regional carrier",
    "operational",
    "operations",
]


FINANCIAL_TERMS = [
    "revenue",
    "operating income",
    "net income",
    "margin",
    "fuel cost",
    "fuel expense",
    "cash",
    "debt",
    "assets",
    "profit",
    "expense",
    "financial",
]


# =========================================================
# SEMANTIC PROTOTYPES
# =========================================================

PROTOTYPES = {

    "SQL": [
        "Calculate a business metric from structured data.",
        "Find the highest or lowest airport or route using SQL.",
        "Compare financial metrics between quarters.",
        "Calculate cancellation rates delays revenue margins or flight counts.",
        "Show numerical trends and performance from database tables.",
        "Rank airports routes carriers or financial metrics.",
    ],

    "RAG": [
        "Find what management said in company filings.",
        "Retrieve commentary from annual or quarterly reports.",
        "Find risk factors discussed in SEC documents.",
        "Search documents for explanations or statements.",
        "Answer using textual evidence from company filings.",
        "Tell me what Delta said about a business topic.",
    ],

    "HYBRID": [
        "Calculate a metric and explain it using management commentary.",
        "Use SQL to identify what changed and documents to explain why.",
        "Combine operational data with SEC filing evidence.",
        "Compare financial performance and retrieve management explanations.",
        "Find a numerical trend then search documents for the reason.",
        "Rank a business metric and determine whether management discussed it.",
    ],
}


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 78)
print("DELTA BUSINESS ANALYST")
print("QUESTION ROUTER")
print("=" * 78)

print(
    f"\nLoading routing model:\n"
    f"{MODEL_NAME}"
)

model = SentenceTransformer(
    MODEL_NAME
)


prototype_embeddings = {}


for route, examples in PROTOTYPES.items():

    embeddings = model.encode(
        examples,
        normalize_embeddings=True,
    )

    prototype_embeddings[
        route
    ] = embeddings.mean(
        axis=0
    )


print("Router model ready.")


# =========================================================
# HELPERS
# =========================================================

def find_terms(
    text: str,
    terms: list[str],
):

    return [
        term
        for term in terms
        if term in text
    ]


def matching_patterns(
    text: str,
    patterns: list[str],
):

    matches = []

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):

            matches.append(
                pattern
            )

    return matches


def extract_years(
    question: str,
):

    years = re.findall(
        r"\b(20\d{2})\b",
        question,
    )

    return sorted(
        {
            int(year)
            for year in years
        }
    )


def extract_quarter(
    question: str,
):

    text = question.lower()


    patterns = {

        1: [
            r"\bq1\b",
            r"\bfirst quarter\b",
            r"\bmarch quarter\b",
        ],

        2: [
            r"\bq2\b",
            r"\bsecond quarter\b",
            r"\bjune quarter\b",
        ],

        3: [
            r"\bq3\b",
            r"\bthird quarter\b",
            r"\bseptember quarter\b",
        ],

        4: [
            r"\bq4\b",
            r"\bfourth quarter\b",
            r"\bdecember quarter\b",
        ],
    }


    for quarter, expressions in (
        patterns.items()
    ):

        for expression in expressions:

            if re.search(
                expression,
                text,
            ):

                return quarter


    return None


def extract_month(
    question: str,
):

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }


    text = question.lower()


    for name, number in (
        months.items()
    ):

        if name in text:
            return number


    return None


def detect_domain(
    question: str,
):

    text = question.lower()

    operational_matches = find_terms(
        text,
        OPERATIONAL_TERMS,
    )

    financial_matches = find_terms(
        text,
        FINANCIAL_TERMS,
    )


    if (
        operational_matches
        and financial_matches
    ):

        return "mixed"


    if operational_matches:
        return "operations"


    if financial_matches:
        return "financial"


    return "document/general"


# =========================================================
# DETECT MULTI-PART QUESTIONS
# =========================================================

def has_structured_clause(
    text: str,
):

    structured_terms = find_terms(
        text,
        STRUCTURED_TERMS,
    )

    analytical_patterns = [

        r"\bwhich\b",
        r"\bhow many\b",
        r"\bhow much\b",

        r"\bcompare\b",
        r"\bcomparison\b",

        r"\bchange\b",
        r"\bchanged\b",

        r"\bincrease\b",
        r"\bincreased\b",

        r"\bdecrease\b",
        r"\bdecreased\b",

        r"\bimprove\b",
        r"\bimproved\b",
        r"\bimprovement\b",

        r"\bworsen\b",
        r"\bworsened\b",
        r"\bworsening\b",

        r"\bdeteriorate\b",
        r"\bdeteriorated\b",
        r"\bdeterioration\b",

        r"\bdecline\b",
        r"\bdeclined\b",

        r"\bhighest\b",
        r"\blowest\b",
        r"\bworst\b",
        r"\bbest\b",

        r"\brate\b",
        r"\bmargin\b",
        r"\bperformance\b",
    ]


    analytical = any(
        re.search(
            pattern,
            text,
        )

        for pattern
        in analytical_patterns
    )


    return bool(
        structured_terms
        and analytical
    )


def has_document_clause(
    text: str,
):

    return bool(
        matching_patterns(
            text,
            DOCUMENT_REQUEST_PATTERNS,
        )
    )


# =========================================================
# SEMANTIC ROUTING
# =========================================================

def semantic_route(
    question: str,
):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
    )[0]


    scores = {}


    for route in ROUTES:

        scores[
            route
        ] = float(
            query_embedding
            @ prototype_embeddings[
                route
            ]
        )


    best_route = max(
        scores,
        key=scores.get,
    )


    return (
        best_route,
        scores,
    )


# =========================================================
# MAIN ROUTER
# =========================================================

def route_question(
    question: str,
):

    text = (
        question
        .lower()
        .strip()
    )


    structured_matches = (
        find_terms(
            text,
            STRUCTURED_TERMS,
        )
    )


    document_matches = (
        matching_patterns(
            text,
            DOCUMENT_REQUEST_PATTERNS,
        )
    )


    explanation_matches = (
        matching_patterns(
            text,
            EXPLANATION_PATTERNS,
        )
    )


    semantic_choice, semantic_scores = (
        semantic_route(
            question
        )
    )


    structured_clause = (
        has_structured_clause(
            text
        )
    )


    document_clause = (
        has_document_clause(
            text
        )
    )


    reasons = []


    # =====================================================
    # MULTI-PART:
    # structured calculation + document question
    # =====================================================

    if (
        structured_clause
        and document_clause
    ):

        route = "HYBRID"

        reasons.append(
            "Question contains both a "
            "structured analytical request "
            "and a document-evidence request."
        )


    # =====================================================
    # PURE DOCUMENT REQUEST
    # Example:
    # What did Delta say about liquidity and debt?
    # =====================================================

    elif document_clause:

        route = "RAG"

        reasons.append(
            "Question primarily asks what "
            "Delta or management said in documents."
        )


    # =====================================================
    # WHY / DRIVERS OF A MEASURABLE METRIC
    # =====================================================

    elif (
        explanation_matches
        and structured_matches
    ):

        route = "HYBRID"

        reasons.append(
            "Question asks for the drivers "
            "of a measurable business outcome."
        )


    # =====================================================
    # STRUCTURED ANALYTICS
    # =====================================================

    elif structured_matches:

        route = "SQL"

        reasons.append(
            "Question can primarily be answered "
            "using structured operational or "
            "financial data."
        )


    # =====================================================
    # AMBIGUOUS
    # =====================================================

    else:

        route = semantic_choice

        reasons.append(
            "No strong deterministic rule matched; "
            "semantic routing was used."
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    score_values = sorted(
        semantic_scores.values(),
        reverse=True,
    )


    semantic_margin = (
        score_values[0]
        - score_values[1]
    )


    if (
        structured_clause
        and document_clause
    ):

        confidence = 0.97


    elif document_clause:

        confidence = 0.96


    elif (
        explanation_matches
        and structured_matches
    ):

        confidence = 0.94


    elif structured_matches:

        confidence = 0.93


    else:

        confidence = min(
            0.90,
            0.55
            + semantic_margin
            * 3,
        )


    # =====================================================
    # TIME EXTRACTION
    # =====================================================

    years = extract_years(
        question
    )

    quarter = extract_quarter(
        question
    )

    month = extract_month(
        question
    )


    # =====================================================
    # COMPARISON DETECTION
    # =====================================================

    comparison_patterns = [

        r"\bcompare\b",
        r"\bversus\b",
        r"\bvs\.?\b",

        r"\bchange\b",
        r"\bchanged\b",

        r"\bincrease\b",
        r"\bincreased\b",

        r"\bdecrease\b",
        r"\bdecreased\b",

        r"\bimprove\b",
        r"\bimproved\b",

        r"\bworsen\b",
        r"\bworsened\b",

        r"\bdeteriorate\b",
        r"\bdeteriorated\b",

        r"\bdecline\b",
        r"\bdeclined\b",

        r"\byear over year\b",
        r"\byoy\b",

        r"\bquarter over quarter\b",
        r"\bqoq\b",
    ]


    is_comparison = any(
        re.search(
            pattern,
            text,
        )

        for pattern
        in comparison_patterns
    )


    return {

        "question":
            question,

        "route":
            route,

        "confidence":
            round(
                confidence,
                3,
            ),

        "domain":
            detect_domain(
                question
            ),

        "years":
            years,

        "quarter":
            quarter,

        "month":
            month,

        "is_comparison":
            is_comparison,

        "structured_matches":
            structured_matches,

        "document_matches":
            document_matches,

        "explanation_matches":
            explanation_matches,

        "structured_clause":
            structured_clause,

        "document_clause":
            document_clause,

        "semantic_sql":
            round(
                semantic_scores[
                    "SQL"
                ],
                3,
            ),

        "semantic_rag":
            round(
                semantic_scores[
                    "RAG"
                ],
                3,
            ),

        "semantic_hybrid":
            round(
                semantic_scores[
                    "HYBRID"
                ],
                3,
            ),

        "reason":
            " ".join(
                reasons
            ),
    }


# =========================================================
# TEST QUESTIONS
# =========================================================


if __name__ == "__main__":
    TEST_QUESTIONS = [
    
        # SQL
        "What was Delta's cancellation rate in June 2026?",
    
        "Which five airports had the worst on-time performance in 2025?",
    
        "How many Delta-marketed flights operated in 2024?",
    
        "Compare Delta operating margin in Q2 2025 and Q2 2026.",
    
        "Which operating carrier had the highest cancellation rate in 2025?",
    
    
        # RAG
        "What did management say about premium travel demand?",
    
        "What risks did Delta discuss in its 2025 10-K?",
    
        "What did Delta say about liquidity and debt?",
    
        "According to management, how is Delta thinking about premium products?",

        "What did Delta management say about fuel costs?",
    
    
        # HYBRID
        "Why did operating margin decline in Q2 2026?",
    
        "Which airport had the largest on-time decline in 2025 and what did management say about operational performance?",
    
        "How did revenue change in Q2 2026 and what did management say caused the change?",

        "How did Delta's operating margin change from Q2 2025 to Q2 2026, and what explanation did management provide?",
    
        "Did Delta's operational performance worsen in 2025 and how did management explain the year?",
    
        "How did fuel cost change in Q2 2026 and what affected it?",
    
        "Which partner airline performed worst in 2025 and did Delta discuss regional carrier performance?",
    ]
    
    
    EXPECTED_ROUTES = [
        "SQL",
        "SQL",
        "SQL",
        "SQL",
        "SQL",
    
        "RAG",
        "RAG",
        "RAG",
        "RAG",
    
        "HYBRID",
        "HYBRID",
        "HYBRID",
        "HYBRID",
        "HYBRID",
        "HYBRID",
    ]
    
    
    # =========================================================
    # RUN TESTS
    # =========================================================
    
    results = []
    
    
    for number, (
        question,
        expected_route,
    ) in enumerate(
        zip(
            TEST_QUESTIONS,
            EXPECTED_ROUTES,
        ),
        start=1,
    ):
    
        result = route_question(
            question
        )
    
    
        result[
            "expected_route"
        ] = expected_route
    
    
        result[
            "correct"
        ] = (
            result["route"]
            == expected_route
        )
    
    
        results.append(
            result
        )
    
    
        print(
            "\n"
            + "=" * 78
        )
    
        print(
            f"[{number}] "
            f"{question}"
        )
    
        print(
            "=" * 78
        )
    
    
        print(
            f"Route:      "
            f"{result['route']}"
        )
    
        print(
            f"Expected:   "
            f"{expected_route}"
        )
    
        print(
            f"Correct:    "
            f"{result['correct']}"
        )
    
        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )
    
        print(
            f"Domain:     "
            f"{result['domain']}"
        )
    
        print(
            f"Years:      "
            f"{result['years']}"
        )
    
        print(
            f"Quarter:    "
            f"{result['quarter']}"
        )
    
        print(
            f"Month:      "
            f"{result['month']}"
        )
    
        print(
            f"Comparison: "
            f"{result['is_comparison']}"
        )
    
    
        print(
            "\nSemantic scores:"
        )
    
        print(
            f"  SQL:    "
            f"{result['semantic_sql']}"
        )
    
        print(
            f"  RAG:    "
            f"{result['semantic_rag']}"
        )
    
        print(
            f"  HYBRID: "
            f"{result['semantic_hybrid']}"
        )
    
    
        print(
            f"\nReason: "
            f"{result['reason']}"
        )
    
    
    # =========================================================
    # SAVE RESULTS
    # =========================================================
    
    results_df = pd.DataFrame(
        results
    )
    
    
    for column in [
        "years",
        "structured_matches",
        "document_matches",
        "explanation_matches",
    ]:
    
        results_df[
            column
        ] = (
            results_df[
                column
            ]
            .apply(str)
        )
    
    
    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )
    
    
    # =========================================================
    # EVALUATION
    # =========================================================
    
    correct_count = sum(
        result["correct"]
        for result in results
    )
    
    accuracy = (
        correct_count
        / len(results)
        * 100
    )
    
    
    print(
        "\n"
        + "=" * 78
    )
    
    print(
        "QUESTION ROUTER SUMMARY"
    )
    
    print(
        "=" * 78
    )
    
    
    route_counts = (
        pd.DataFrame(
            results
        )[
            "route"
        ]
        .value_counts()
    )
    
    
    print(
        "\nRoutes assigned:"
    )
    
    print(
        route_counts.to_string()
    )
    
    
    print(
        f"\nCorrect routes: "
        f"{correct_count}/"
        f"{len(results)}"
    )
    
    print(
        f"Router test accuracy: "
        f"{accuracy:.2f}%"
    )
    
    
    incorrect = [
        result
        for result in results
        if not result[
            "correct"
        ]
    ]
    
    
    if incorrect:
    
        print(
            "\nIncorrect routes:"
        )
    
        for result in incorrect:
    
            print(
                f"\n  Question: "
                f"{result['question']}"
            )
    
            print(
                f"  Expected: "
                f"{result['expected_route']}"
            )
    
            print(
                f"  Predicted: "
                f"{result['route']}"
            )
    
    
    print(
        f"\nResults saved:\n"
        f"{OUTPUT_PATH}"
    )
    
    
    print(
        "\n"
        + "=" * 78
    )
    
    print(
        "QUESTION ROUTING COMPLETE"
    )
    
    print(
        "=" * 78
    )
