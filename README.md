# Nexus - AI Context Retrieval System

> ⚠️ **Alpha/Experimental**: This project is in early development. It's functional but may have bugs and breaking changes. Use for learning and experimentation, not production systems.

**Fast, intelligent semantic search with RAG capabilities**

Nexus is an AI-powered context retrieval system that combines semantic search, query expansion, and response synthesis to provide intelligent answers from your documents. Built to showcase modern RAG architecture and help developers learn advanced search techniques.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## ✨ Key Features

### 🎯 Intelligent Search
- **Query Expansion**: Uses LLM to generate query variations for better recall
- **Response Synthesis**: RAG-powered natural language answers from search results
- **Hybrid Search**: Combines semantic (vector) + keyword (BM25) search for best quality
- **Semantic Caching**: 10x faster repeat queries with Redis-based vector similarity cache

### 🔧 Production-Ready Features
- **Structured Logging**: Request tracing, contextual logs, JSON output
- **Prometheus Metrics**: 20+ metrics for monitoring search, ingestion, and costs
- **Cost Tracking**: Real-time tracking of embedding and LLM costs per provider
- **Rate Limiting**: Protect your API from abuse
- **Multi-Provider Support**: Ollama (free), Gemini, OpenAI with smart fallbacks

### 🚀 Developer Experience
- **One-Command Setup**: `docker-compose up -d` and you're running
- **Zero Cost**: Use Ollama for free embeddings, query expansion, and synthesis
- **REST API**: Clean FastAPI with auto-generated docs
- **File Upload**: Direct markdown file ingestion via multipart/form-data

---

## 🎨 What Makes Nexus Different?

**Semantic Query Caching**: Unlike traditional RAG systems, Nexus caches based on query vector similarity, not exact string matching. Repeat queries are 10x faster (~50ms vs ~500ms).

**Cost Optimization**: Built-in cost tracking per operation, intelligent provider fallback (free Ollama → cheap Gemini → expensive OpenAI), and configurable strategies.

**Complete Observability**: Structured logs, Prometheus metrics, request tracing, and cost analytics out of the box - not an afterthought.

**Multi-Query Search**: LLM-powered query expansion generates 4-5 variations, searches all, and merges results for significantly better recall.

---

## 🏗️ Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│  FastAPI + Logging + Metrics + Rate Limiting + Caching  │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────┐            ┌──────────────┐
│   Search     │            │   Ingest     │
│              │            │              │
│ • Expand     │            │ • Chunk      │
│ • Embed      │            │ • Hash       │
│ • Retrieve   │            │ • Embed      │
│ • Synthesize │            │ • Store      │
└──────┬───────┘            └──────┬───────┘
       │                           │
       └─────────────┬─────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌─────────────┐          ┌──────────────┐
│   Qdrant    │          │  PostgreSQL  │
│   Vector DB │          │  Metadata    │
└─────────────┘          └──────────────┘
        │
        ▼
┌─────────────┐
│   Redis     │
│   Cache     │
└─────────────┘
```

**Components**:
- **FastAPI**: REST API with automatic docs
- **Qdrant**: Vector database for hybrid search
- **PostgreSQL**: Metadata, analytics, entities
- **Redis**: Semantic query cache
- **Ollama/Gemini/OpenAI**: Embeddings and LLM
- **Prometheus**: Metrics collection

---

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 8GB RAM minimum
- 10GB disk space (for Ollama models)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/nexus.git
cd nexus

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env - Add at least one provider API key (or use Ollama for free)
# GEMINI_API_KEY=your_key_here  # Recommended (cheap)
# OPENAI_API_KEY=sk-...         # Optional (expensive)
# Ollama runs locally - no API key needed!

# 4. Start all services
docker-compose up -d

# 5. Wait for Ollama to download models (first time only, ~5-10 min)
docker-compose logs -f ollama

# 6. Verify services are running
docker-compose ps

# 7. Access API documentation
open http://localhost:8000/docs
```

### First Search in 3 Steps

```bash
# 1. Create a collection
COLLECTION_ID=$(curl -s -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my-docs", "embedding_provider": "gemini"}' \
  | jq -r '.id')

# 2. Ingest a document
curl -X POST "http://localhost:8000/collections/$COLLECTION_ID/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [{
      "content": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
      "title": "FastAPI Overview"
    }]
  }'

# 3. Search with AI-generated answer
curl -X POST "http://localhost:8000/collections/$COLLECTION_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is FastAPI",
    "synthesize": true,
    "limit": 5
  }'
```

**Response**:
```json
{
  "query": "what is FastAPI",
  "synthesized_answer": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It is based on standard Python type hints and is designed for high performance.",
  "results": [...],
  "latency_ms": 1200,
  "tokens_used": 850,
  "synthesis_cost_usd": 0.0
}
```

---

## 📚 API Examples

### Search with Query Expansion

Expand query into multiple variations for better recall:

```bash
curl -X POST "http://localhost:8000/collections/$COLLECTION_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I auth users",
    "expand_query": true,
    "limit": 10
  }'
```

**Response includes**:
```json
{
  "expanded_queries": [
    "how do I auth users",
    "how do I authenticate users",
    "user authentication methods",
    "login and authorization implementation"
  ],
  "results": [...],
  "latency_ms": 2100
}
```

### Search with Everything Enabled

Get the best quality results:

```bash
curl -X POST "http://localhost:8000/collections/$COLLECTION_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication best practices",
    "expand_query": true,
    "synthesize": true,
    "hybrid": true,
    "use_cache": true,
    "limit": 10
  }'
```

**Features**:
- ✅ Query expansion (better recall)
- ✅ Hybrid search (semantic + keyword)
- ✅ Response synthesis (natural language answer)
- ✅ Caching (faster repeat queries)

### File Upload Ingestion

```bash
curl -X POST "http://localhost:8000/collections/$COLLECTION_ID/ingest/file" \
  -F "file=@document.md" \
  -F "title=My Document"
```

### View Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

---

## 🎯 Configuration

### Provider Selection Strategy

Set in `.env`:

```bash
# Embedding provider strategy
EMBEDDING_PROVIDER_STRATEGY=cost  # Options: cost, local-first, quality

# Cost: Ollama (free) → Gemini ($0.00001/1k) → OpenAI ($0.0001/1k)
# Local-first: Ollama → Gemini → OpenAI
# Quality: OpenAI → Gemini → Ollama
```

### Logging

```bash
# Development: Human-readable console logs
LOG_LEVEL=INFO
JSON_LOGS=false

# Production: JSON structured logs
LOG_LEVEL=WARNING
JSON_LOGS=true
```

### Performance Tuning

```bash
# Chunking
CHUNK_SIZE=8192       # Max tokens per chunk
CHUNK_OVERLAP=100     # Overlap between chunks

# Search
SEARCH_CACHE_TTL=3600 # Cache TTL in seconds (1 hour)

# Workers
API_WORKERS=4         # Uvicorn workers
```

---

## 📊 Performance & Cost

### Latency (approximate)

| Configuration | Latency | Quality | Cost |
|--------------|---------|---------|------|
| Basic search | ~200ms | Good | $0.00001/search |
| + Cache hit | ~50ms | Good | $0 |
| + Hybrid | ~250ms | Better | $0.00001/search |
| + Expansion | ~2s | Better recall | $0 (Ollama) |
| + Synthesis | ~3s | Best UX | $0 (Ollama) |
| All features | ~5s | Maximum | $0.00001/search |

### Cost Breakdown

**Using Gemini (recommended)**:
- Embedding: $0.00001 per 1K tokens
- 1000 searches/day: ~$0.30/month
- 10K documents ingested: ~$0.50 one-time

**Using Ollama (free)**:
- Everything: $0
- Trade-off: Requires 8GB RAM, slower embeddings

---

## 🗺️ Roadmap

### ✅ Completed (Current - v0.1 Alpha)

- ✅ Multi-provider support (Ollama, Gemini, OpenAI)
- ✅ Hybrid search (semantic + BM25)
- ✅ Query expansion with LLM
- ✅ Response synthesis (RAG)
- ✅ Semantic caching
- ✅ Structured logging + Prometheus metrics
- ✅ Cost tracking
- ✅ Rate limiting
- ✅ File upload ingestion
- ✅ Basic authentication

### 🚧 Planned Features

**Next (v0.2)**:
- [ ] Real-time updates via webhooks (not just batch syncs)
- [ ] Personalized search results per user
- [ ] Cross-source deduplication
- [ ] Search quality metrics (precision@k, NDCG)
- [ ] Multi-model embedding support (different models per collection)

**Future**:
- [ ] Python SDK
- [ ] Web UI for search testing
- [ ] Grafana dashboard templates
- [ ] Scheduled batch ingestion
- [ ] Advanced filters (date ranges, metadata queries)
- [ ] More source integrations (Slack, Notion, Google Drive)

---

## ⚠️ Current Limitations

Being transparent about what's not ready yet:

### Performance
- **Single-node only**: No horizontal scaling yet
- **Ollama is slow**: ~2-3s for embeddings vs ~200ms for Gemini
- **Large files**: No chunking strategy for 100MB+ files
- **No batching**: Ingestion is serial, not parallelized

### Features
- **No user management**: Basic API key auth only, no multi-user
- **No fine-grained permissions**: All collections accessible with one key
- **Limited source integrations**: Only file upload, GitHub, Gmail (basic)
- **No async jobs**: Large ingestions block the request

### Stability
- **Alpha quality**: Expect bugs and breaking changes
- **Limited testing**: Core features tested, edge cases may fail
- **No migration strategy**: Database schema may change

### Known Issues
- [ ] Ollama occasionally fails to start in Docker (restart fixes it)
- [ ] Cache hit rate is lower than expected for similar queries (~60% vs target 90%)
- [ ] Query expansion can be slow if Ollama is cold (~5s first request)
- [ ] No retry logic for provider failures

---

## 🤝 Contributing

**Current Status**: Not accepting contributions yet as we stabilize the codebase.

However, we welcome:
- 🐛 **Bug reports**: Open an issue with reproduction steps
- 💡 **Feature suggestions**: Discuss in GitHub Discussions
- 📣 **Feedback**: Share your experience, pain points, ideas

**Future Contributions** (once stable):
- Performance improvements
- New provider integrations (Cohere, Anthropic, etc.)

---

## 🧪 Testing Locally

```bash
# Run all tests
pytest backend/tests/ -v

# Run specific test
pytest backend/tests/test_search.py -v

# Check logs
docker-compose logs -f backend

# Monitor metrics
watch -n 1 'curl -s http://localhost:8000/metrics | grep nexus_search'
```

---

## 📖 Learn More

### Understanding the Architecture

**Why Hybrid Search?**
Semantic search (vectors) is great for concepts, but misses exact keyword matches. BM25 (keyword) catches those. Combining both with RRF (Reciprocal Rank Fusion) gives best quality.

**Why Query Expansion?**
User queries are often vague or use different terminology than documents. LLM generates variations ("auth" → "authentication", "login", "authorization") to find all relevant results.

**Why Semantic Caching?**
Traditional caching uses exact query strings. Nexus caches based on query vector similarity - so "how to login" and "user login methods" hit the same cache.

**Why Multi-Provider?**
Flexibility and cost optimization. Start free with Ollama, switch to Gemini for production speed, fall back to OpenAI if needed.

### Project Goals

This project exists to:
1. **Showcase** modern RAG architecture with production features
2. **Help developers** learn about semantic search, vector databases, and LLMs
3. **Experiment** with new techniques (semantic caching, query expansion, cost optimization)

Not goals:
- Replace mature solutions like LangChain/LlamaIndex (use those for production)
- Be the fastest or most feature-rich (it's about learning)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Free to use, modify, and distribute. Attribution appreciated!

---

## 🙏 Acknowledgments

- Inspired by [Airweave](https://github.com/airweave-ai/airweave) and the RAG community
- Built with [FastAPI](https://fastapi.tiangolo.com/), [Qdrant](https://qdrant.tech/), [Ollama](https://ollama.ai/)
- Thanks to all the open-source projects that made this possible

---

## 💬 Community & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/nexus/issues) for bugs
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/nexus/discussions) for questions and ideas
- **Feedback**: Share your experience! What works, what doesn't, what's confusing?

---

## 📊 Project Status

- **Build Status**: Alpha (functional, not stable)
- **Last Updated**: November 2025
- **Maintenance**: Active development
- **Production Ready**: ❌ No (use for learning/experimentation only)

---

**Built by developers, for developers** 🚀

If you find this project helpful, give it a ⭐️ on GitHub!
