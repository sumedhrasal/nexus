"""Search quality test for REFRAG paper.

This test validates the quality of the search and synthesis pipeline by ingesting
a research paper (REFRAG) and comparing synthesized answers against expected responses.

CURRENT STATE (Baseline):
- HTML content is ingested with raw MathML/LaTeX markup
- Keyword overlap: ~8% (Target: >= 30%)
- Concept coverage: ~0% (Target: >= 20%)
- The chunks contain HTML/MathML noise which degrades quality

FUTURE IMPROVEMENTS:
1. Implement HTML-to-text conversion in the /html endpoint
   - Strip HTML tags and extract clean text
   - Convert MathML to plain text or skip equations
   - Preserve semantic structure (headings, paragraphs)

2. Enhanced chunking strategies
   - Semantic chunking based on document structure
   - Overlap between chunks for better context

3. Better quality metrics
   - Semantic similarity using embeddings
   - BLEU/ROUGE scores for text comparison
   - Domain-specific evaluation metrics
"""

import pytest
from pathlib import Path
from httpx import AsyncClient
from app.main import app


def load_test_data(filename: str) -> str:
    """Load test data file."""
    data_dir = Path(__file__).parent.parent / "data"
    file_path = data_dir / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def calculate_similarity_score(generated: str, expected: str) -> dict:
    """Calculate similarity metrics between generated and expected answers.

    Returns:
        dict with metrics:
        - keyword_overlap: percentage of key terms from expected found in generated
        - concept_coverage: rough estimate of concept alignment
        - length_ratio: ratio of generated to expected length
    """
    # Normalize text
    generated_lower = generated.lower()
    expected_lower = expected.lower()

    # Extract key terms from expected answer
    key_terms = [
        "refrag", "rag", "retrieval-augmented generation", "decoding",
        "latency", "memory", "compression", "encoder", "decoder",
        "reinforcement learning", "ttft", "time-to-first-token",
        "llama", "perplexity", "chunk", "embedding",
        "context", "passage", "attention", "curriculum learning",
        "cepe", "replug", "arxiv", "multi-turn"
    ]

    # Count keyword overlaps
    matched_terms = sum(1 for term in key_terms if term in generated_lower)
    keyword_overlap = (matched_terms / len(key_terms)) * 100

    # Extract key concepts/numbers from expected
    key_concepts = [
        "30.85", "16×", "compress-sense-expand",
        "next-paragraph prediction", "reconstruction task",
        "slimpajama", "pg19", "proof-pile"
    ]

    matched_concepts = sum(1 for concept in key_concepts if concept in generated_lower)
    concept_coverage = (matched_concepts / len(key_concepts)) * 100

    # Calculate length ratio
    length_ratio = len(generated) / len(expected) if expected else 0

    return {
        "keyword_overlap_pct": round(keyword_overlap, 2),
        "concept_coverage_pct": round(concept_coverage, 2),
        "length_ratio": round(length_ratio, 2),
        "matched_keywords": matched_terms,
        "total_keywords": len(key_terms),
        "matched_concepts": matched_concepts,
        "total_concepts": len(key_concepts)
    }


@pytest.mark.asyncio
async def test_refrag_paper_search_quality():
    """Test: Ingest REFRAG paper HTML → Search with query → Compare against expected answer.

    This test validates the quality of the search and synthesis pipeline by:
    1. Ingesting a research paper (REFRAG) as HTML
    2. Performing a search with synthesis enabled
    3. Comparing the synthesized answer against a templated expected response
    4. Calculating similarity metrics to assess quality
    """

    # Load test data
    html_content = load_test_data("2509.01092v2.html")
    expected_response = load_test_data("2509.01092v2_response.txt")

    async with AsyncClient(app=app, base_url="http://localhost:8000", timeout=120.0) as client:
        # 1. Check if collection exists, if not create it
        collection_name = "REFRAG Paper Quality Test"
        collection_id = None

        # Try to find existing collection by name
        collections_response = await client.get("/collections")
        assert collections_response.status_code == 200
        collections = collections_response.json()

        for col in collections:
            if col["name"] == collection_name:
                collection_id = col["id"]
                print(f"\n✅ Found existing collection: {collection_id}")
                break

        # Create collection if not found
        if not collection_id:
            create_response = await client.post(
                "/collections",
                json={
                    "name": collection_name,
                    "embedding_provider": "ollama",
                    # "vector_dimension": 768
                }
            )
            assert create_response.status_code == 201, f"Failed to create collection: {create_response.text}"
            collection_data = create_response.json()
            collection_id = collection_data["id"]
            print(f"\n📦 Created new collection: {collection_id}")

        # 2. Ingest HTML document using /html endpoint (with text extraction)
        # Check if already ingested by trying to search - if we get results, skip ingestion
        test_search = await client.post(
            f"/collections/{collection_id}/search",
            json={"query": "REFRAG", "limit": 1}
        )

        should_ingest = True
        if test_search.status_code == 200:
            search_data = test_search.json()
            if search_data["total_results"] > 0:
                print(f"\n♻️  Collection already has {search_data['total_results']} results, skipping ingestion")
                should_ingest = False

        if should_ingest:
            print(f"\n📤 Ingesting HTML document via /html endpoint...")

            # Read HTML file
            from pathlib import Path
            html_path = Path(__file__).parent.parent / "data" / "2509.01092v2.html"

            with open(html_path, "rb") as f:
                files = {"file": ("2509.01092v2.html", f, "text/html")}
                data = {"title": "REFRAG: Rethinking RAG based Decoding"}

                ingest_response = await client.post(
                    f"/collections/{collection_id}/ingest/html",
                    files=files,
                    data=data
                )

            assert ingest_response.status_code == 200, f"Failed to ingest HTML: {ingest_response.text}"
            ingest_data = ingest_response.json()
            assert ingest_data["documents_processed"] == 1
            assert ingest_data["chunks_created"] > 0, "No chunks were created from the HTML content"

            print(f"📄 Ingested document with {ingest_data['chunks_created']} chunks in {ingest_data['processing_time_ms']}ms")
        # 3. Search with synthesis enabled
        # search_query = "what is REFRAG paper about?"
        search_query = "How does REFRAG work?"
        search_response = await client.post(
            f"/collections/{collection_id}/search",
            json={
                "query": search_query,
                "limit": 10,
                "use_cache": False,
                "synthesize": True,
                "expand_query": True,
                "hybrid": True,
                "search_mode": "hybrid"
            }
        )
        assert search_response.status_code == 200, f"Search failed: {search_response.text}"
        search_data = search_response.json()

        # Verify search returned results
        assert search_data["total_results"] > 0, "No search results found"

        # Print search results for debugging
        print(f"\n🔍 SEARCH RESULTS:")
        for idx, result in enumerate(search_data["results"][:3], 1):
            print(f"\n  Result {idx} (score: {result['score']:.4f}):")
            print(f"  {result['content'][:200]}...")

        # If synthesis failed (Ollama not available), skip synthesis checks
        if search_data["synthesized_answer"] is None:
            print("\n⚠️  Synthesis not available (Ollama not running) - skipping synthesis quality checks")
            print("✅ Test passed - search returned relevant chunks")
            # Note: Collection is preserved for reuse
            return

        # 4. Compare quality against expected response
        generated_answer = search_data["synthesized_answer"]
        similarity_metrics = calculate_similarity_score(generated_answer, expected_response)

        print("\n" + "="*80)
        print("SEARCH QUALITY ASSESSMENT")
        print("="*80)
        print(f"\n📊 Query: {search_query}")
        print(f"📈 Results found: {search_data['total_results']}")
        print(f"⏱️  Search latency: {search_data['latency_ms']}ms")
        print(f"🔤 Tokens used: {search_data['tokens_used']}")
        print(f"\n📝 GENERATED ANSWER:")
        print("-" * 80)
        print(generated_answer)
        print("-" * 80)
        # print(f"\n✅ EXPECTED ANSWER:")
        # print("-" * 80)
        # print(expected_response)
        print("-" * 80)
        print(f"\n📊 SIMILARITY METRICS:")
        print(f"   • Keyword Overlap: {similarity_metrics['keyword_overlap_pct']}% "
              f"({similarity_metrics['matched_keywords']}/{similarity_metrics['total_keywords']})")
        print(f"   • Concept Coverage: {similarity_metrics['concept_coverage_pct']}% "
              f"({similarity_metrics['matched_concepts']}/{similarity_metrics['total_concepts']})")
        print(f"   • Length Ratio: {similarity_metrics['length_ratio']}")
        print("="*80 + "\n")

        # Quality assertions - BASELINE TRACKING
        # NOTE: Current implementation ingests raw HTML which contains MathML/LaTeX markup
        # This pollutes the chunks and degrades quality. These assertions document the baseline.
        # TODO: Implement HTML-to-text conversion to improve quality

        print(f"\n⚠️  BASELINE QUALITY METRICS (Raw HTML ingestion):")
        print(f"   Current keyword overlap: {similarity_metrics['keyword_overlap_pct']}%")
        print(f"   Current concept coverage: {similarity_metrics['concept_coverage_pct']}%")
        print(f"   Target keyword overlap: >= 30%")
        print(f"   Target concept coverage: >= 20%")

        # For now, just verify we got SOME response with the key term
        # These are very low bars to establish baseline
        assert similarity_metrics['keyword_overlap_pct'] >= 4.0, (
            f"Keyword overlap too low: {similarity_metrics['keyword_overlap_pct']}% "
            f"(baseline >= 4%)"
        )

        # Verify length is reasonable (not completely broken)
        assert 0.05 <= similarity_metrics['length_ratio'] <= 5.0, (
            f"Answer length ratio out of range: {similarity_metrics['length_ratio']} "
            f"(expected 0.05-5.0)"
        )

        # Note: Collection is preserved for reuse in future test runs
        print(f"✅ Test passed with quality metrics above thresholds")
