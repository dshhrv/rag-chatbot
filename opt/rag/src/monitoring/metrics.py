from prometheus_client import Counter, Histogram, Gauge


RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
    ["lang", "intent", "status"],
)

RAG_ERRORS = Counter(
    "rag_errors_total",
    "Total number of RAG errors",
    ["stage"],
)

RAG_LATENCY = Histogram(
    "rag_latency_seconds",
    "Total RAG request latency",
    ["lang", "intent"],
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 120, 180, 300],
)

RAG_ESCALATIONS = Counter(
    "rag_escalations_total",
    "Total number of escalations",
    ["reason"],
)

RAG_HALLUCINATION_BLOCKS = Counter(
    "rag_hallucination_blocks_total",
    "Total number of hallucination blocks",
)

RAG_FEEDBACK = Counter(
    "rag_feedback_total",
    "Total number of user feedback events",
    ["rating"],
)

RAG_IN_PROGRESS = Gauge(
    "rag_requests_in_progress",
    "Current number of active RAG requests",
)