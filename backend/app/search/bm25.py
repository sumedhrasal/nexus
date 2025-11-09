"""BM25 sparse vector scoring for hybrid search."""

import math
from typing import List, Dict, Set
from collections import Counter
import re


class BM25:
    """BM25 scoring for text documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 scorer.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase, remove punctuation, split
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens

    def fit(self, corpus: List[str]):
        """Fit BM25 on a corpus of documents.

        Args:
            corpus: List of document texts
        """
        self.corpus_size = len(corpus)
        doc_lens = []
        term_counts: List[Dict[str, int]] = []

        # First pass: collect term frequencies and document lengths
        for doc in corpus:
            tokens = self.tokenize(doc)
            doc_lens.append(len(tokens))
            term_count = Counter(tokens)
            term_counts.append(term_count)

            # Track document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.doc_len = doc_lens
        self.avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0

        # Calculate IDF scores
        for term, df in self.doc_freqs.items():
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = idf

    def get_scores(self, query: str, corpus: List[str]) -> List[float]:
        """Calculate BM25 scores for query against corpus.

        Args:
            query: Query text
            corpus: List of document texts

        Returns:
            List of BM25 scores
        """
        query_tokens = self.tokenize(query)
        scores = []

        for doc_idx, doc in enumerate(corpus):
            doc_tokens = self.tokenize(doc)
            doc_len = len(doc_tokens)
            term_freqs = Counter(doc_tokens)

            score = 0.0
            for term in query_tokens:
                if term not in term_freqs:
                    continue

                # Term frequency in document
                tf = term_freqs[term]

                # IDF score
                idf = self.idf.get(term, 0.0)

                # Length normalization
                norm = 1 - self.b + self.b * (doc_len / self.avgdl) if self.avgdl > 0 else 1

                # BM25 score component
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

            scores.append(score)

        return scores

    def get_sparse_vector(self, text: str) -> Dict[str, float]:
        """Get sparse vector representation for text.

        Args:
            text: Input text

        Returns:
            Dict mapping terms to their weights
        """
        tokens = self.tokenize(text)
        term_freqs = Counter(tokens)
        doc_len = len(tokens)

        sparse_vec = {}
        for term, tf in term_freqs.items():
            if term in self.idf:
                # Weight = TF * IDF (simplified for sparse vector)
                weight = tf * self.idf[term]
                sparse_vec[term] = weight

        return sparse_vec


def reciprocal_rank_fusion(
    rankings: List[List[tuple]],
    k: int = 60
) -> List[tuple]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    Args:
        rankings: List of rankings, each ranking is list of (id, score) tuples
        k: RRF constant (default: 60)

    Returns:
        Fused ranking as list of (id, score) tuples
    """
    # Calculate RRF scores
    rrf_scores: Dict[str, float] = {}

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by RRF score
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused
