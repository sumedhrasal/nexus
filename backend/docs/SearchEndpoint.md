# Search Endpoint Documentation

## Overview

The search endpoint (`/collections/{collection_id}/search`) provides powerful document retrieval capabilities with optional AI-powered enhancements. The endpoint can operate in two main modes based on configuration, with multiple optional features that can be toggled via request parameters and environment settings.

## How the Search Endpoint Works

### Core Flow

Every search request follows this basic pattern:

1. **Query Reformulation** - Automatically converts natural language queries into search-friendly format
2. **Route Selection** - Either uses Adaptive RAG (experimental) or Legacy Pipeline (production)
3. **Result Retrieval** - Fetches relevant documents based on the selected strategy
4. **Post-Processing** - Optionally reranks, diversifies, and synthesizes results
5. **Response** - Returns ranked search results with optional AI-generated answer

### Two Main Operating Modes

The endpoint can operate in two fundamentally different modes based on the `ENABLE_ADAPTIVE_RAG` setting:

---

## Mode 1: Adaptive RAG Pipeline (Experimental)

**When it's used:** `ENABLE_ADAPTIVE_RAG=true`
**Status:** Experimental - disabled by default
**Best for:** Complex queries requiring intelligent routing and multi-step retrieval

### What It Does

Adaptive RAG uses an AI planner to analyze each query and choose the best retrieval strategy. The flow works like this:

1. **Query Analysis** - An LLM examines your query to understand its complexity and information needs
2. **Strategy Selection** - Based on analysis, picks one of three strategies (explained below)
3. **Intelligent Retrieval** - Executes the chosen strategy to gather relevant information
4. **Result Processing** - Reranks and diversifies results for quality
5. **Smart Answer Generation** - Creates a response tailored to the query's complexity level

### The Three Retrieval Strategies

The planner automatically routes queries to one of these strategies:

#### Strategy 1: DIRECT
**When used:** Simple, straightforward questions
**What it does:** Performs a single focused search with the exact query
**Example queries:**
- "What is REFRAG?"
- "When was the transformer architecture introduced?"
- "Define neural networks"

**How it works:** Converts your query to an embedding, searches the vector database, returns the most relevant chunks. Fast and efficient for simple lookups.

---

#### Strategy 2: DECOMPOSE
**When used:** Complex questions with multiple aspects
**When enabled:** `ENABLE_QUERY_DECOMPOSITION=true` (on by default)
**What it does:** Breaks your question into 2-5 focused sub-questions, searches each separately, then combines results

**Example query:** "Compare BERT and GPT architectures"

**How it works:**
1. LLM breaks down into: "What is BERT architecture?", "What is GPT architecture?", "Key differences between BERT and GPT?"
2. Each sub-question is searched independently
3. Results are merged using Reciprocal Rank Fusion (prioritizes documents that appear in multiple result sets)
4. Duplicates are removed
5. Final ranked list covers all aspects of your question

**Configuration:** `MAX_SUB_QUERIES` controls how many sub-questions can be generated (default: 5, range: 2-10)

---

#### Strategy 3: ITERATIVE
**When used:** Deep research questions requiring comprehensive information
**When enabled:** `ENABLE_ITERATIVE_RAG=true` (on by default)
**What it does:** Searches multiple times, using an AI to check if enough information has been gathered

**Example query:** "Explain all approaches to attention mechanisms in deep learning"

**How it works:**
1. Performs initial search with your query
2. LLM reviews the retrieved information and asks: "Is this sufficient to answer the question?"
3. If not sufficient, LLM generates follow-up queries to fill gaps (e.g., "multi-head attention details", "cross-attention mechanisms")
4. Repeats until information is sufficient or max iterations reached
5. Can optionally combine with DECOMPOSE for ultra-complex queries

**Configuration:** `MAX_RAG_ITERATIONS` controls maximum search rounds (default: 3, range: 1-5)

---

### How Complexity Determines Strategy

The AI planner classifies queries into four complexity levels:

| Complexity | Strategy Used | Description |
|-----------|---------------|-------------|
| **Simple** | DIRECT | Single-fact lookups, definitions |
| **Moderate** | DIRECT | How/why questions, single-topic explanations |
| **Complex** | DECOMPOSE | Multi-aspect comparisons, multi-topic queries |
| **Research** | ITERATIVE | Comprehensive analysis, "explain everything about X" type questions |

---

## Mode 2: Legacy Pipeline (Production)

**When it's used:** `ENABLE_ADAPTIVE_RAG=false` (default)
**Status:** Production-ready, stable
**Best for:** Most queries, especially when speed is important

### What It Does

The legacy pipeline uses a proven, optimized search flow with multiple enhancement layers. It processes every query the same way, applying various techniques to improve result quality.

### The Search Flow

#### Step 1: Query Classification
**What happens:** System analyzes your query to determine if it's semantic (concept-based) or keyword (term-matching) focused
**Why:** Different query types benefit from different search approaches
**Example:**
- "machine learning algorithms" → Semantic (70% dense vectors, 30% keywords)
- "error code 404" → Keyword (40% dense vectors, 60% keywords)

---

#### Step 2: Query Expansion (Optional)
**When enabled:** `expand_query=true` in request
**What it does:** Uses an LLM to generate 2-4 variations of your query
**Why:** Catches relevant documents that use different terminology

**Example query:** "neural networks"
**Expanded to:**
- "neural networks"
- "artificial neural networks"
- "deep learning models"
- "connectionist systems"

**What happens next:** Searches with all variations, merges results using Reciprocal Rank Fusion

---

#### Step 3: Hybrid Search
**When enabled:** `hybrid=true` in request (default)
**What it does:** Combines two search methods:
1. **Dense Vector Search** - Finds semantically similar content using embeddings
2. **BM25 Keyword Search** - Finds exact term matches using traditional text search

**How they combine:** Both methods produce ranked lists, which are merged using weights determined in Step 1

**Why it's powerful:** Dense vectors catch conceptual matches, BM25 catches precise terminology. Together they cover both broad concepts and specific terms.

---

#### Step 4: Cross-Encoder Reranking (Optional)
**When enabled:** `ENABLE_RERANKING=true` (on by default)
**What it does:** Takes the top 20-40 candidates and re-scores them with a more sophisticated model
**Why:** Initial search is fast but approximate; reranking is slower but more accurate

**How it works:**
1. Initial search returns 40 candidates (5x more than requested)
2. Cross-encoder model deeply analyzes each candidate against your query
3. Results are re-sorted based on these refined scores
4. Top results are kept

**Models used:** Configurable via `RERANKER_MODEL` (default: BAAI/bge-reranker-base)

---

#### Step 5: MMR Diversity
**What it does:** Removes redundant results while keeping relevance high
**Why:** Sometimes multiple chunks from the same document or very similar chunks rank highly - this ensures variety

**How it works:** Balances two goals:
- 70% weight on relevance to your query
- 30% weight on being different from already-selected results

**Effect:** You get diverse perspectives rather than subtle variations of the same content

---

#### Step 6: Result Caching (Optional)
**When enabled:** `use_cache=true` in request (default)
**What it does:** Saves search results for identical queries
**Why:** Dramatically speeds up repeated searches

**How it works:** Query embedding is used as cache key; if found, skip all search steps and return cached results immediately

---

## Response Synthesis (Optional - Both Modes)

**When enabled:** `synthesize=true` in request
**What it does:** Uses an LLM to generate a natural language answer from the search results
**Why:** Instead of just getting document chunks, you get a coherent answer to your question

### How Synthesis Works

1. **Context Preparation** - Top search results are assembled as context
2. **Answer Generation** - LLM reads the context and writes an answer to your query
3. **Style Selection** - Answer style varies based on query complexity

### Answer Styles

The synthesis adapts to query complexity:

| Style | When Used | Characteristics | Example Output Length |
|-------|-----------|----------------|----------------------|
| **Concise** | Simple lookups (Legacy mode) | 2-4 sentences, key facts only | ~50-100 words |
| **Structured** | Moderate/Complex (Adaptive RAG) | Organized sections, bullet points | ~150-300 words |
| **Comprehensive** | Research queries (Adaptive RAG) | Thorough multi-paragraph analysis | ~300-500 words |

**In Legacy Mode:** Always uses concise style
**In Adaptive RAG:** Style matches the complexity level identified by the planner

---

## Request Parameters

When making a search request, you can control behavior with these parameters:

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Your search question |
| `limit` | int | 40 | Maximum number of results to return (1-100) |
| `offset` | int | 0 | Skip first N results (for pagination) |

### Feature Toggles (Request Level)

| Parameter | Type | Default | What It Controls |
|-----------|------|---------|------------------|
| `use_cache` | bool | true | Enable result caching for faster repeated queries |
| `expand_query` | bool | false | Generate multiple query variations with LLM (Legacy mode only) |
| `synthesize` | bool | false | Generate AI-written answer from search results |
| `hybrid` | bool | true | Use both vector and keyword search (vs vector only) |
| `provider` | string | null | Override embedding provider (ollama, gemini, openai) |
| `filters` | object | null | Metadata filters to narrow search scope |

### Example Requests

**Simple search (just results):**
```json
{
  "query": "what is transfer learning?",
  "limit": 10
}
```

**Enhanced search (with synthesis):**
```json
{
  "query": "explain attention mechanisms",
  "limit": 20,
  "synthesize": true,
  "hybrid": true
}
```

**Complex search (query expansion + synthesis):**
```json
{
  "query": "compare supervised and unsupervised learning",
  "limit": 40,
  "expand_query": true,
  "synthesize": true,
  "use_cache": true
}
```

---

## Environment Configuration

These settings control which features are available system-wide:

### Mode Selection

```env
# Choose between Adaptive RAG (experimental) or Legacy (production)
ENABLE_ADAPTIVE_RAG=false
```

### Adaptive RAG Settings (when enabled)

```env
# Enable query decomposition strategy
ENABLE_QUERY_DECOMPOSITION=true

# Enable iterative retrieval strategy
ENABLE_ITERATIVE_RAG=true

# Max sub-questions for decomposition (2-10)
MAX_SUB_QUERIES=5

# Max search iterations for iterative strategy (1-5)
MAX_RAG_ITERATIONS=3
```

### Legacy Pipeline Settings

```env
# Enable cross-encoder reranking
ENABLE_RERANKING=true

# Reranker model to use
RERANKER_MODEL=BAAI/bge-reranker-base

# Number of top candidates to rerank
RERANKER_TOP_K=20
```

---

## Understanding the Response

When you search, the response includes:

### Result Fields

| Field | Description |
|-------|-------------|
| `results` | Array of matching document chunks, ranked by relevance |
| `total_results` | Number of results returned |
| `latency_ms` | Time taken to process the search (milliseconds) |
| `from_cache` | Whether results came from cache (true = instant) |
| `provider_used` | Which embedding provider was used |
| `synthesized_answer` | AI-generated answer (when `synthesize=true`) |
| `expanded_queries` | Query variations used (when `expand_query=true`) |

### Each Result Contains

| Field | Description |
|-------|-------------|
| `entity_id` | Unique identifier for the chunk |
| `content` | The actual text content |
| `title` | Document title this chunk came from |
| `score` | Relevance score (0.0 to 1.0, higher = more relevant) |
| `metadata` | Additional information (file path, page number, etc.) |

---

## Performance & Cost

### Expected Latency

| Configuration | Typical Response Time |
|--------------|----------------------|
| **Simple search (cached)** | <100ms |
| **Simple search (not cached)** | 300-500ms |
| **Legacy + synthesis** | 1-2 seconds |
| **Legacy + expansion + synthesis** | 2-3 seconds |
| **Adaptive DIRECT** | 1-2 seconds |
| **Adaptive DECOMPOSE** | 3-5 seconds |
| **Adaptive ITERATIVE** | 5-10 seconds |

**Speed tips:**
- Enable caching (`use_cache=true`) for faster repeated queries
- Disable query expansion for faster searches
- Use Legacy mode for sub-second response times

### LLM Costs

**Free operations (no LLM calls):**
- Basic search (no synthesis, no expansion)
- Cached results
- Reranking (uses local model)

**Costs LLM calls:**
- Query expansion (1 call per search)
- Response synthesis (1 call per search)
- Adaptive RAG planning (1 call per search)
- Query decomposition (1 call when triggered)
- Iterative retrieval (1-5 calls when triggered)

**Cost optimization:** If using paid LLM providers (OpenAI, Gemini), consider using free Ollama for planning/expansion and paid providers only for final synthesis

---

## Automatic Fallback

### When Adaptive RAG Falls Back to Legacy

If Adaptive RAG is enabled but encounters a problem, the system automatically switches to Legacy mode without failing your request. This happens when:

1. The AI planner fails to generate a valid plan
2. LLM is unreachable or times out
3. Any error occurs during plan execution

**Result:** You always get search results, even if the advanced features fail. The response will indicate which mode was actually used.

---

## Choosing the Right Configuration

### When to Use Legacy Mode (Default)

✅ **Best for:**
- General-purpose search needs
- Speed is important (need sub-second response)
- Simple to moderate query complexity
- Cost-conscious deployments
- Production systems requiring stability

**Recommended settings:**
```json
{
  "synthesize": false,     // Fast, just return chunks
  "expand_query": false,   // Single query is sufficient
  "hybrid": true,          // Best quality with minimal cost
  "use_cache": true        // Speed up repeated queries
}
```

### When to Use Adaptive RAG Mode

✅ **Best for:**
- Research or analysis applications
- Complex, multi-faceted questions
- Users who need comprehensive answers
- Quality matters more than speed
- Development/testing environments

**Recommended settings:**
```json
{
  "synthesize": true,      // Get comprehensive answers
  "hybrid": true,          // Still use hybrid search
  "use_cache": true        // Cache to reduce LLM calls
}
```

**Environment:** Set `ENABLE_ADAPTIVE_RAG=true`

### Feature Combination Guide

| Use Case | `expand_query` | `synthesize` | `hybrid` | `use_cache` | Mode |
|----------|----------------|--------------|----------|-------------|------|
| Fast lookup | false | false | true | true | Legacy |
| Simple Q&A | false | true | true | true | Legacy |
| Deep research | false | true | true | true | Adaptive |
| Maximum quality | true | true | true | true | Legacy |
| Development/testing | varies | true | true | false | Either |

---

## Summary

### What You Can Control

**Request-level flags** (change per search):
- `query` - What you're searching for
- `limit` - How many results you want
- `synthesize` - Get an AI-generated answer (true/false)
- `expand_query` - Use multiple query variations (true/false, Legacy mode only)
- `hybrid` - Combine vector + keyword search (true/false)
- `use_cache` - Use cached results when available (true/false)
- `provider` - Override which embedding provider to use
- `filters` - Narrow search by metadata

**Environment-level settings** (system-wide):
- `ENABLE_ADAPTIVE_RAG` - Switch between Adaptive (experimental) and Legacy (production) modes
- `ENABLE_RERANKING` - Use cross-encoder for better result ranking
- `ENABLE_QUERY_DECOMPOSITION` - Allow query splitting in Adaptive mode
- `ENABLE_ITERATIVE_RAG` - Allow multi-round retrieval in Adaptive mode
- `MAX_SUB_QUERIES` - Limit sub-questions in decomposition
- `MAX_RAG_ITERATIONS` - Limit search rounds in iterative mode

### Key Capabilities

**Search quality features:**
- ✅ Hybrid search (vector + keyword)
- ✅ Query reformulation (automatic)
- ✅ Cross-encoder reranking (optional)
- ✅ MMR diversity filtering (automatic)
- ✅ Result caching (optional)

**AI-powered enhancements:**
- ✅ Query expansion (Legacy mode, optional)
- ✅ Intelligent query routing (Adaptive mode)
- ✅ Query decomposition (Adaptive mode)
- ✅ Iterative retrieval (Adaptive mode)
- ✅ Response synthesis (both modes, optional)

**Current status:**
- **Production ready:** Legacy mode with all features
- **Experimental:** Adaptive RAG mode (test in staging first)
- **Automatic fallback:** Always returns results even if advanced features fail
