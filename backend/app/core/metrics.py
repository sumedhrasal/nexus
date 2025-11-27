"""Prometheus metrics for monitoring."""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from contextlib import contextmanager

# Info metric for application metadata
app_info = Info('nexus_app', 'Nexus application information')
app_info.info({'version': '1.0.0', 'environment': 'production'})

# Request metrics
http_requests_total = Counter(
    'nexus_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'nexus_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Search metrics
search_requests_total = Counter(
    'nexus_search_requests_total',
    'Total search requests',
    ['collection_id', 'cache_hit']
)

search_duration_seconds = Histogram(
    'nexus_search_duration_seconds',
    'Search operation duration in seconds',
    ['collection_id'],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

search_results_count = Histogram(
    'nexus_search_results_count',
    'Number of search results returned',
    ['collection_id'],
    buckets=[0, 1, 5, 10, 20, 50, 100]
)

# Query expansion metrics
query_expansion_requests_total = Counter(
    'nexus_query_expansion_requests_total',
    'Total query expansion requests'
)

query_expansion_duration_seconds = Histogram(
    'nexus_query_expansion_duration_seconds',
    'Query expansion duration in seconds',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Response synthesis metrics
response_synthesis_requests_total = Counter(
    'nexus_response_synthesis_requests_total',
    'Total response synthesis requests'
)

response_synthesis_duration_seconds = Histogram(
    'nexus_response_synthesis_duration_seconds',
    'Response synthesis duration in seconds',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

response_synthesis_tokens = Histogram(
    'nexus_response_synthesis_tokens',
    'Tokens used in response synthesis',
    buckets=[100, 500, 1000, 2000, 5000, 10000]
)

# Ingestion metrics
ingestion_requests_total = Counter(
    'nexus_ingestion_requests_total',
    'Total ingestion requests',
    ['collection_id']
)

ingestion_duration_seconds = Histogram(
    'nexus_ingestion_duration_seconds',
    'Ingestion operation duration in seconds',
    ['collection_id'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

ingestion_chunks_created = Histogram(
    'nexus_ingestion_chunks_created',
    'Number of chunks created during ingestion',
    ['collection_id'],
    buckets=[1, 5, 10, 20, 50, 100, 200, 500]
)

ingestion_documents_processed = Counter(
    'nexus_ingestion_documents_processed_total',
    'Total documents processed',
    ['collection_id']
)

# Embedding metrics
embedding_requests_total = Counter(
    'nexus_embedding_requests_total',
    'Total embedding requests',
    ['provider']
)

embedding_duration_seconds = Histogram(
    'nexus_embedding_duration_seconds',
    'Embedding operation duration in seconds',
    ['provider'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

embedding_cost_usd = Counter(
    'nexus_embedding_cost_usd_total',
    'Total embedding cost in USD',
    ['provider']
)

# Cache metrics
cache_hits_total = Counter(
    'nexus_cache_hits_total',
    'Total cache hits',
    ['collection_id']
)

cache_misses_total = Counter(
    'nexus_cache_misses_total',
    'Total cache misses',
    ['collection_id']
)

cache_size_bytes = Gauge(
    'nexus_cache_size_bytes',
    'Current cache size in bytes'
)

# Provider metrics
provider_requests_total = Counter(
    'nexus_provider_requests_total',
    'Total provider requests',
    ['provider', 'operation', 'status']
)

provider_fallback_total = Counter(
    'nexus_provider_fallback_total',
    'Total provider fallbacks',
    ['from_provider', 'to_provider']
)

# Database metrics
db_connections_active = Gauge(
    'nexus_db_connections_active',
    'Active database connections'
)

db_query_duration_seconds = Histogram(
    'nexus_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

# Error metrics
errors_total = Counter(
    'nexus_errors_total',
    'Total errors',
    ['endpoint', 'error_type']
)


# Context managers for timing operations
@contextmanager
def time_operation(histogram, *labels):
    """Context manager to time an operation and record to histogram.

    Args:
        histogram: Prometheus Histogram metric
        *labels: Labels for the histogram

    Example:
        with time_operation(search_duration_seconds, collection_id):
            # perform search
            pass
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        histogram.labels(*labels).observe(duration)


def record_search_metrics(
    collection_id: str,
    duration_seconds: float,
    result_count: int,
    cache_hit: bool
):
    """Record search operation metrics.

    Args:
        collection_id: Collection ID
        duration_seconds: Search duration in seconds
        result_count: Number of results returned
        cache_hit: Whether the search hit cache
    """
    search_requests_total.labels(
        collection_id=collection_id,
        cache_hit=str(cache_hit)
    ).inc()

    search_duration_seconds.labels(collection_id=collection_id).observe(duration_seconds)
    search_results_count.labels(collection_id=collection_id).observe(result_count)

    if cache_hit:
        cache_hits_total.labels(collection_id=collection_id).inc()
    else:
        cache_misses_total.labels(collection_id=collection_id).inc()


def record_ingestion_metrics(
    collection_id: str,
    duration_seconds: float,
    documents_processed: int,
    chunks_created: int
):
    """Record ingestion operation metrics.

    Args:
        collection_id: Collection ID
        duration_seconds: Ingestion duration in seconds
        documents_processed: Number of documents processed
        chunks_created: Number of chunks created
    """
    ingestion_requests_total.labels(collection_id=collection_id).inc()
    ingestion_duration_seconds.labels(collection_id=collection_id).observe(duration_seconds)
    ingestion_chunks_created.labels(collection_id=collection_id).observe(chunks_created)
    ingestion_documents_processed.labels(collection_id=collection_id).inc(documents_processed)


def record_embedding_metrics(
    provider: str,
    duration_seconds: float,
    cost_usd: float = 0.0
):
    """Record embedding operation metrics.

    Args:
        provider: Provider name (ollama, gemini, openai)
        duration_seconds: Embedding duration in seconds
        cost_usd: Cost in USD
    """
    embedding_requests_total.labels(provider=provider).inc()
    embedding_duration_seconds.labels(provider=provider).observe(duration_seconds)

    if cost_usd > 0:
        embedding_cost_usd.labels(provider=provider).inc(cost_usd)


def record_synthesis_metrics(
    duration_seconds: float,
    tokens_used: int
):
    """Record response synthesis metrics.

    Args:
        duration_seconds: Synthesis duration in seconds
        tokens_used: Number of tokens used
    """
    response_synthesis_requests_total.inc()
    response_synthesis_duration_seconds.observe(duration_seconds)
    response_synthesis_tokens.observe(tokens_used)
