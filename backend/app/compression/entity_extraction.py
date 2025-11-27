"""Entity extraction for knowledge graph construction.

Extracts entities and relationships from text using:
1. NER (Named Entity Recognition) with spaCy
2. LLM-based extraction for complex relationships
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import re

from app.core.providers.router import ProviderRouter
from app.core.logging import get_logger

logger = get_logger(__name__)


class EntityType(str, Enum):
    """Common entity types for knowledge graph."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECH"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    OTHER = "OTHER"


@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    name: str
    entity_type: EntityType
    description: Optional[str]
    source_chunk_id: str
    mentions: List[str]  # All text spans where entity appears
    confidence: float = 1.0


@dataclass
class ExtractedRelationship:
    """Relationship between two entities."""
    source_entity: str
    target_entity: str
    relationship_type: str  # "mentions", "implements", "uses", etc.
    source_chunk_id: str
    context: str  # Sentence/context where relationship appears
    confidence: float = 1.0


class EntityExtractor:
    """Extract entities and relationships from text."""

    def __init__(self, provider: ProviderRouter, use_llm: bool = True):
        """Initialize entity extractor.

        Args:
            provider: Provider router for LLM-based extraction
            use_llm: Whether to use LLM for enhanced extraction
        """
        self.provider = provider
        self.use_llm = use_llm
        self.spacy_model = None

    def _load_spacy_model(self):
        """Lazy load spaCy model."""
        if self.spacy_model is None:
            try:
                import spacy
                # Try loading English model
                try:
                    self.spacy_model = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spacy_model_not_found", using_rule_based=True)
                    # Fallback to rule-based extraction only
                    self.spacy_model = False
            except ImportError:
                logger.warning("spacy_not_installed", using_rule_based=True)
                self.spacy_model = False

    async def extract_entities(
        self,
        text: str,
        chunk_id: str,
        max_entities: int = 50
    ) -> List[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: Text to extract from
            chunk_id: Source chunk identifier
            max_entities: Maximum entities to extract

        Returns:
            List of extracted entities
        """
        entities: List[ExtractedEntity] = []

        # Try spaCy-based extraction first
        if self.spacy_model is None:
            self._load_spacy_model()

        if self.spacy_model and self.spacy_model is not False:
            entities.extend(
                self._extract_with_spacy(text, chunk_id)
            )

        # Enhance with LLM if enabled
        if self.use_llm and len(entities) < max_entities:
            try:
                llm_entities = await self._extract_with_llm(text, chunk_id)
                # Merge with spaCy entities (avoid duplicates)
                existing_names = {e.name.lower() for e in entities}
                for entity in llm_entities:
                    if entity.name.lower() not in existing_names:
                        entities.append(entity)
                        if len(entities) >= max_entities:
                            break
            except Exception as e:
                logger.warning("llm_extraction_failed", error=str(e))

        return entities[:max_entities]

    def _extract_with_spacy(
        self,
        text: str,
        chunk_id: str
    ) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        if not self.spacy_model or self.spacy_model is False:
            return []

        doc = self.spacy_model(text[:100000])  # Limit text length

        entities = []
        for ent in doc.ents:
            # Map spaCy labels to our EntityType
            entity_type = self._map_spacy_label(ent.label_)

            entity = ExtractedEntity(
                name=ent.text,
                entity_type=entity_type,
                description=None,
                source_chunk_id=chunk_id,
                mentions=[ent.text],
                confidence=0.8  # spaCy confidence
            )
            entities.append(entity)

        return entities

    async def _extract_with_llm(
        self,
        text: str,
        chunk_id: str
    ) -> List[ExtractedEntity]:
        """Extract entities using LLM."""
        # Limit text to avoid token limits
        truncated_text = text[:2000]

        prompt = f"""Extract key entities from the following text.
For each entity, identify:
1. Entity name
2. Entity type (PERSON, ORG, CONCEPT, TECH, PRODUCT, etc.)
3. Brief description

Text:
{truncated_text}

Format your response as a list with each entity on a new line:
[EntityType] EntityName: Description

Entities:"""

        response, _ = await self.provider.generate(
            prompt=prompt,
            system="You are an expert at identifying and classifying entities in text."
        )

        # Parse LLM response
        entities = self._parse_llm_response(response, chunk_id)
        return entities

    def _parse_llm_response(
        self,
        response: str,
        chunk_id: str
    ) -> List[ExtractedEntity]:
        """Parse LLM entity extraction response."""
        entities = []

        # Expected format: [TYPE] Name: Description
        pattern = r'\[([^\]]+)\]\s+([^:]+):\s*(.+)'

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                type_str, name, description = match.groups()

                # Map to EntityType
                try:
                    entity_type = EntityType[type_str.upper()]
                except KeyError:
                    entity_type = EntityType.OTHER

                entity = ExtractedEntity(
                    name=name.strip(),
                    entity_type=entity_type,
                    description=description.strip(),
                    source_chunk_id=chunk_id,
                    mentions=[name.strip()],
                    confidence=0.7  # LLM confidence
                )
                entities.append(entity)

        return entities

    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "NORP": EntityType.ORGANIZATION,
            "FAC": EntityType.LOCATION,
        }
        return mapping.get(label, EntityType.OTHER)

    async def extract_relationships(
        self,
        entities: List[ExtractedEntity],
        text: str,
        chunk_id: str
    ) -> List[ExtractedRelationship]:
        """Extract relationships between entities using LLM.

        Args:
            entities: Extracted entities
            text: Source text
            chunk_id: Source chunk identifier

        Returns:
            List of extracted relationships
        """
        if not self.use_llm or len(entities) < 2:
            return []

        # Create entity list for prompt
        entity_names = [e.name for e in entities[:20]]  # Limit to top 20

        truncated_text = text[:2000]

        prompt = f"""Given these entities from a text:
{', '.join(entity_names)}

And this text:
{truncated_text}

Identify relationships between the entities.
For each relationship, specify:
- Source entity
- Target entity
- Relationship type (e.g., "uses", "implements", "mentions", "located_in", etc.)

Format: SourceEntity -> RelationType -> TargetEntity

Relationships:"""

        try:
            response, _ = await self.provider.generate(
                prompt=prompt,
                system="You are an expert at identifying relationships between entities in text."
            )

            relationships = self._parse_relationships(
                response, text, chunk_id, entity_names
            )
            return relationships

        except Exception as e:
            logger.warning("relationship_extraction_failed", error=str(e))
            return []

    def _parse_relationships(
        self,
        response: str,
        context: str,
        chunk_id: str,
        valid_entities: List[str]
    ) -> List[ExtractedRelationship]:
        """Parse LLM relationship extraction response."""
        relationships = []

        # Expected format: Entity1 -> relationship -> Entity2
        pattern = r'(.+?)\s*->\s*(.+?)\s*->\s*(.+)'

        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                source, rel_type, target = match.groups()

                source = source.strip()
                target = target.strip()
                rel_type = rel_type.strip()

                # Validate entities exist in our extracted list
                if source in valid_entities and target in valid_entities:
                    relationship = ExtractedRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type=rel_type,
                        source_chunk_id=chunk_id,
                        context=context[:500],  # Store snippet
                        confidence=0.6
                    )
                    relationships.append(relationship)

        return relationships


def get_entity_extractor(provider: ProviderRouter, use_llm: bool = True) -> EntityExtractor:
    """Factory function to get entity extractor instance."""
    return EntityExtractor(provider, use_llm)
