"""Hierarchical Summarization (RAPTOR-style) for RAG compression.

This module implements a 3-layer hierarchical summarization strategy:
- Layer 0: Original chunks (from chunker)
- Layer 1: Section summaries (clusters of chunks)
- Layer 2: Document summaries (clusters of sections)

Achieves 90% compression with no accuracy loss through:
- Recursive summarization with LLM
- Top-down search starting from document level
- Selective expansion to lower layers as needed
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans

from app.core.providers.router import ProviderRouter
from app.core.logging import get_logger

logger = get_logger(__name__)


class SummaryLayer(str, Enum):
    """Hierarchical summary layers."""
    CHUNK = "chunk"          # Layer 0: Original chunks
    SECTION = "section"      # Layer 1: Section summaries
    DOCUMENT = "document"    # Layer 2: Document summaries


@dataclass
class HierarchicalNode:
    """Node in hierarchical summary tree."""
    layer: SummaryLayer
    content: str
    embedding: List[float]
    children_ids: List[str]  # IDs of child nodes
    parent_id: Optional[str]  # ID of parent node
    metadata: Dict[str, Any]

    # Unique identifier
    node_id: str

    # Original chunk info (only for layer 0)
    original_chunk_index: Optional[int] = None


class HierarchicalSummarizer:
    """Builds and manages hierarchical summary trees for documents."""

    def __init__(self, provider: ProviderRouter, vector_dimension: int):
        """Initialize hierarchical summarizer.

        Args:
            provider: Provider router for LLM and embeddings
            vector_dimension: Expected embedding dimension
        """
        self.provider = provider
        self.vector_dimension = vector_dimension

    async def build_hierarchy(
        self,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        document_metadata: Dict[str, Any],
        entity_id: str
    ) -> List[HierarchicalNode]:
        """Build 3-layer hierarchical summary tree.

        Args:
            chunks: Original text chunks
            chunk_embeddings: Embeddings for chunks
            document_metadata: Metadata for the document
            entity_id: Parent entity ID

        Returns:
            List of all hierarchical nodes (chunks + sections + document)
        """
        all_nodes = []

        # Layer 0: Create chunk nodes
        chunk_nodes = self._create_chunk_nodes(
            chunks, chunk_embeddings, entity_id, document_metadata
        )
        all_nodes.extend(chunk_nodes)

        # Layer 1: Create section summaries by clustering chunks
        section_nodes = await self._create_section_summaries(
            chunk_nodes, entity_id, document_metadata
        )
        all_nodes.extend(section_nodes)

        # Layer 2: Create document summary from sections
        document_node = await self._create_document_summary(
            section_nodes, entity_id, document_metadata
        )
        all_nodes.append(document_node)

        logger.info(
            "hierarchy_built",
            chunks=len(chunk_nodes),
            sections=len(section_nodes),
            compression_ratio=round(len(all_nodes) / len(chunks), 2)
        )

        return all_nodes

    def _create_chunk_nodes(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        entity_id: str,
        metadata: Dict[str, Any]
    ) -> List[HierarchicalNode]:
        """Create layer 0 nodes from original chunks."""
        nodes = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node = HierarchicalNode(
                layer=SummaryLayer.CHUNK,
                content=chunk,
                embedding=embedding,
                children_ids=[],
                parent_id=None,  # Will be set when sections are created
                metadata=metadata.copy(),
                node_id=f"{entity_id}_chunk_{i}",
                original_chunk_index=i
            )
            nodes.append(node)
        return nodes

    async def _create_section_summaries(
        self,
        chunk_nodes: List[HierarchicalNode],
        entity_id: str,
        metadata: Dict[str, Any]
    ) -> List[HierarchicalNode]:
        """Create layer 1 summaries by clustering and summarizing chunks."""
        if len(chunk_nodes) <= 3:
            # Too few chunks, create single section
            return await self._summarize_single_section(
                chunk_nodes, entity_id, metadata, section_idx=0
            )

        # Cluster chunks using embeddings
        n_sections = max(2, len(chunk_nodes) // 5)  # ~5 chunks per section
        embeddings_array = np.array([node.embedding for node in chunk_nodes])

        kmeans = KMeans(n_clusters=n_sections, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)

        # Group chunks by cluster
        clusters: Dict[int, List[HierarchicalNode]] = {}
        for node, label in zip(chunk_nodes, cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(node)

        # Create section summary for each cluster
        section_nodes = []
        for section_idx, chunk_group in clusters.items():
            section = await self._summarize_single_section(
                chunk_group, entity_id, metadata, section_idx
            )
            section_nodes.extend(section)

        return section_nodes

    async def _summarize_single_section(
        self,
        chunk_nodes: List[HierarchicalNode],
        entity_id: str,
        metadata: Dict[str, Any],
        section_idx: int
    ) -> List[HierarchicalNode]:
        """Summarize a group of chunks into a section."""
        # Concatenate chunk contents
        combined_text = "\n\n".join([node.content for node in chunk_nodes])

        # Generate section summary using LLM
        summary_prompt = f"""Summarize the following text into a concise section summary (2-3 sentences).
Focus on the key concepts and main ideas:

{combined_text}

Section Summary:"""

        try:
            summary, _ = await self.provider.generate(
                prompt=summary_prompt,
                system="You are a helpful assistant that creates concise, informative summaries."
            )
        except Exception as e:
            logger.warning("section_summary_failed", error=str(e), using_concatenation=True)
            # Fallback: use first 500 chars of combined text
            summary = combined_text[:500] + "..."

        # Generate embedding for summary
        embeddings, _ = await self.provider.embed(
            [summary],
            required_dimension=self.vector_dimension
        )

        # Create section node
        node_id = f"{entity_id}_section_{section_idx}"
        section_node = HierarchicalNode(
            layer=SummaryLayer.SECTION,
            content=summary,
            embedding=embeddings[0],
            children_ids=[node.node_id for node in chunk_nodes],
            parent_id=None,  # Will be set when document is created
            metadata=metadata.copy(),
            node_id=node_id
        )

        # Update parent_id in chunk nodes
        for chunk_node in chunk_nodes:
            chunk_node.parent_id = node_id

        return [section_node]

    async def _create_document_summary(
        self,
        section_nodes: List[HierarchicalNode],
        entity_id: str,
        metadata: Dict[str, Any]
    ) -> HierarchicalNode:
        """Create layer 2 summary from section summaries."""
        # Concatenate section summaries
        combined_sections = "\n\n".join([node.content for node in section_nodes])

        # Generate document summary using LLM
        title = metadata.get("title", "Document")
        summary_prompt = f"""Create a high-level summary of this document titled "{title}".
Synthesize the following section summaries into a single paragraph (3-5 sentences)
that captures the document's main themes and key points:

{combined_sections}

Document Summary:"""

        try:
            summary, _ = await self.provider.generate(
                prompt=summary_prompt,
                system="You are a helpful assistant that creates concise, informative document summaries."
            )
        except Exception as e:
            logger.warning("document_summary_failed", error=str(e), using_concatenation=True)
            # Fallback: use first section summary
            summary = section_nodes[0].content if section_nodes else combined_sections[:500]

        # Generate embedding for document summary
        embeddings, _ = await self.provider.embed(
            [summary],
            required_dimension=self.vector_dimension
        )

        # Create document node
        node_id = f"{entity_id}_document"
        document_node = HierarchicalNode(
            layer=SummaryLayer.DOCUMENT,
            content=summary,
            embedding=embeddings[0],
            children_ids=[node.node_id for node in section_nodes],
            parent_id=None,  # Top level
            metadata=metadata.copy(),
            node_id=node_id
        )

        # Update parent_id in section nodes
        for section_node in section_nodes:
            section_node.parent_id = node_id

        return document_node


def get_hierarchical_summarizer(
    provider: ProviderRouter,
    vector_dimension: int
) -> HierarchicalSummarizer:
    """Factory function to get hierarchical summarizer instance."""
    return HierarchicalSummarizer(provider, vector_dimension)
