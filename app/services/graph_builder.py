"""
Graph Builder Service for Storyline Analysis
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import networkx as nx
from neo4j import GraphDatabase, basic_auth
import json
from collections import Counter, defaultdict
from app.utils.logging_utils import get_logger, log_performance
from .semantic_chunker import SemanticChunk
from ..config.settings_config import get_settings

logger = get_logger("graph_builder")


@dataclass
class StorylineNode:
    """Graph node representing semantic clusters and storylines"""
    id: str
    summary: str
    temporal_info: Dict[str, Any]
    participants: List[str]
    themes: List[str]
    chunk_ids: List[str]
    centrality_score: float
    confidence: float
    entity_density: float
    temporal_span_days: Optional[float] = None
    main_storyline: bool = False


@dataclass
class RelationshipEdge:
    """Typed connections between storyline nodes"""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float
    confidence: float
    evidence: List[str]
    metadata: Dict[str, Any]


class RelationshipType:
    """Relationship type constants"""
    TEMPORAL = "TEMPORAL_SEQUENCE"
    CAUSAL = "CAUSAL_LINK"
    THEMATIC = "THEMATIC_SIMILARITY"
    SPATIAL = "SPATIAL_CONNECTION"
    PARTICIPANT = "SHARED_PARTICIPANTS"


class GraphBuilder:
    """
    Advanced graph builder for creating storyline networks from semantic chunks.
    Integrates with Neo4j for persistent storage and complex graph operations.
    """

    def __init__(self):
        """
        Initialize graph builder with Neo4j connection
        """
        settings = get_settings()
        try:
            self.driver = GraphDatabase.driver(
                settings.database.neo4j_uri,
                auth=basic_auth(settings.database.neo4j_user, settings.database.neo4j_password)
            )
            logger.info("Neo4j connection established", uri=settings.database.neo4j_uri,
                        user=settings.database.neo4j_user)
        except Exception as e:
            logger.error("Failed to connect to Neo4j", error=str(e), uri=settings.database.neo4j_uri)
            self.driver = None

        # Graph analysis parameters
        self.similarity_threshold = 0.4
        self.entity_overlap_threshold = 0.3
        self.temporal_proximity_days = 30
        self.centrality_threshold = 0.1
        self.main_storyline_count = 7

    @log_performance(logger)
    async def build_storyline_graph(self, chunks: List[SemanticChunk]) -> nx.Graph:
        """
        Build complete storyline graph from semantic chunks
        
        Args:
            chunks: List of processed semantic chunks
            
        Returns:
            NetworkX graph with nodes and weighted edges
        """
        logger.info("Starting storyline graph construction", total_chunks=len(chunks))

        # Create base graph
        graph = nx.Graph()

        # Add nodes from chunks
        await self._add_chunk_nodes(graph, chunks)

        # Calculate and add edges
        await self._add_relationship_edges(graph, chunks)

        # Enrich graph with community detection
        await self._enrich_with_communities(graph)

        # Calculate centrality measures
        await self._calculate_centrality_measures(graph)

        logger.info("Storyline graph construction completed",
                    nodes=graph.number_of_nodes(),
                    edges=graph.number_of_edges(),
                    avg_degree=sum(dict(graph.degree()).values()) / graph.number_of_nodes())

        return graph

    async def _add_chunk_nodes(self, graph: nx.Graph, chunks: List[SemanticChunk]) -> None:
        """Add chunk nodes to graph with attributes"""
        for chunk in chunks:
            # Calculate node attributes
            entity_density = len(chunk.entities) / chunk.word_count if chunk.word_count > 0 else 0

            # Extract participant names
            participants = [
                ent['text'] for ent in chunk.entities
                if ent['label'] == 'PERSON'
            ]

            # Extract themes (placeholder - would use topic modeling)
            themes = self._extract_themes_from_entities(chunk.entities)

            graph.add_node(chunk.id, **{
                'content': chunk.content,
                'embedding': chunk.embedding.tolist(),
                'word_count': chunk.word_count,
                'speaker': chunk.speaker,
                'entities': chunk.entities,
                'temporal_info': chunk.temporal_info,
                'participants': participants,
                'themes': themes,
                'entity_density': entity_density,
                'confidence': chunk.confidence,
                'start_position': chunk.start_position,
                'end_position': chunk.end_position
            })

    def _extract_themes_from_entities(self, entities: List[Dict]) -> List[str]:
        """Extract themes from entity types and content"""
        themes = []

        # Simple theme extraction based on entity types
        entity_counts = Counter([ent['label'] for ent in entities])

        if entity_counts.get('ORG', 0) > 0:
            themes.append('organizations')
        if entity_counts.get('GPE', 0) > 0:  # Geopolitical entities
            themes.append('locations')
        if entity_counts.get('PERSON', 0) > 2:
            themes.append('social_interactions')
        if entity_counts.get('DATE', 0) > 0:
            themes.append('temporal_events')
        if entity_counts.get('MONEY', 0) > 0:
            themes.append('financial')
        if entity_counts.get('EVENT', 0) > 0:
            themes.append('significant_events')

        return themes[:3]  # Limit to top 3 themes

    async def _add_relationship_edges(self, graph: nx.Graph, chunks: List[SemanticChunk]) -> None:
        """Calculate and add weighted edges between nodes"""
        chunk_dict = {chunk.id: chunk for chunk in chunks}

        # Calculate relationships between all pairs
        node_ids = list(graph.nodes())
        for i, node1_id in enumerate(node_ids):
            for node2_id in node_ids[i + 1:]:
                chunk1 = chunk_dict[node1_id]
                chunk2 = chunk_dict[node2_id]

                edge_weight, relationship_types = self._calculate_edge_weight(chunk1, chunk2)

                if edge_weight > self.similarity_threshold:
                    # Determine primary relationship type
                    primary_type = max(relationship_types.items(), key=lambda x: x[1])[0]

                    graph.add_edge(node1_id, node2_id, **{
                        'weight': edge_weight,
                        'relationship_type': primary_type,
                        'type_scores': relationship_types,
                        'confidence': min(chunk1.confidence, chunk2.confidence)
                    })

    def _calculate_edge_weight(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> Tuple[float, Dict[str, float]]:
        """Calculate edge weight and relationship types between two chunks"""
        relationship_scores = {}

        # 1. Semantic similarity
        semantic_sim = self._cosine_similarity(chunk1.embedding, chunk2.embedding)
        relationship_scores[RelationshipType.THEMATIC] = semantic_sim

        # 2. Entity overlap (participant connections)
        entity_overlap = self._calculate_entity_overlap(chunk1.entities, chunk2.entities)
        relationship_scores[RelationshipType.PARTICIPANT] = entity_overlap

        # 3. Temporal proximity
        temporal_score = self._calculate_temporal_proximity(chunk1.temporal_info, chunk2.temporal_info)
        relationship_scores[RelationshipType.TEMPORAL] = temporal_score

        # 4. Spatial connections (location overlap)
        spatial_score = self._calculate_spatial_overlap(chunk1.entities, chunk2.entities)
        relationship_scores[RelationshipType.SPATIAL] = spatial_score

        # 5. Positional proximity in transcript
        position_score = self._calculate_position_proximity(chunk1, chunk2)
        relationship_scores[
            RelationshipType.CAUSAL] = position_score * semantic_sim  # Causal likely if close and similar

        # Combined weight with emphasis on different factors
        weights = {
            RelationshipType.THEMATIC: 0.3,
            RelationshipType.PARTICIPANT: 0.25,
            RelationshipType.TEMPORAL: 0.2,
            RelationshipType.SPATIAL: 0.15,
            RelationshipType.CAUSAL: 0.1
        }

        combined_weight = sum(
            relationship_scores[rel_type] * weight
            for rel_type, weight in weights.items()
        )

        return combined_weight, relationship_scores

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        return float(np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))

    def _calculate_entity_overlap(self, entities1: List[Dict], entities2: List[Dict]) -> float:
        """Calculate entity overlap score between two chunks"""
        if not entities1 or not entities2:
            return 0.0

        # Normalize entity texts for comparison
        entities1_normalized = {ent['text'].lower().strip() for ent in entities1}
        entities2_normalized = {ent['text'].lower().strip() for ent in entities2}

        if not entities1_normalized or not entities2_normalized:
            return 0.0

        intersection = entities1_normalized.intersection(entities2_normalized)
        union = entities1_normalized.union(entities2_normalized)

        return len(intersection) / len(union)

    def _calculate_temporal_proximity(self, temporal1: Dict, temporal2: Dict) -> float:
        """Calculate temporal proximity score"""
        if not temporal1 or not temporal2:
            return 0.0

        # Simple heuristic - would need more sophisticated temporal parsing
        dates1 = temporal1.get('explicit_dates', [])
        dates2 = temporal2.get('explicit_dates', [])

        if not dates1 or not dates2:
            # Check for relative expressions that might indicate proximity
            rel1 = temporal1.get('relative_expressions', [])
            rel2 = temporal2.get('relative_expressions', [])

            if rel1 and rel2:
                # Simple check for similar relative expressions
                rel1_texts = {expr['text'].lower() for expr in rel1}
                rel2_texts = {expr['text'].lower() for expr in rel2}
                overlap = len(rel1_texts.intersection(rel2_texts))
                return min(overlap / max(len(rel1_texts), len(rel2_texts)), 1.0)

            return 0.0

        # For now, return moderate score if both have dates
        return 0.5

    def _calculate_spatial_overlap(self, entities1: List[Dict], entities2: List[Dict]) -> float:
        """Calculate spatial/location overlap score"""
        # Extract location entities
        locations1 = {
            ent['text'].lower() for ent in entities1
            if ent['label'] in ['GPE', 'LOC', 'FAC']  # Geopolitical, location, facility
        }
        locations2 = {
            ent['text'].lower() for ent in entities2
            if ent['label'] in ['GPE', 'LOC', 'FAC']
        }

        if not locations1 or not locations2:
            return 0.0

        intersection = locations1.intersection(locations2)
        union = locations1.union(locations2)

        return len(intersection) / len(union)

    def _calculate_position_proximity(self, chunk1: SemanticChunk, chunk2: SemanticChunk) -> float:
        """Calculate proximity based on position in transcript"""
        distance = abs(chunk1.start_position - chunk2.start_position)
        # Exponential decay with distance
        return np.exp(-distance * 5)  # Adjust decay rate as needed

    async def _enrich_with_communities(self, graph: nx.Graph) -> None:
        """Add community detection results to graph"""
        if graph.number_of_nodes() < 2:
            return

        try:
            # Apply Louvain community detection
            communities = nx.community.louvain_communities(graph, seed=42)

            # Add community labels to nodes
            for i, community in enumerate(communities):
                for node_id in community:
                    graph.nodes[node_id]['community'] = i

            logger.info("Community detection completed",
                        total_communities=len(communities),
                        largest_community=max(len(c) for c in communities),
                        smallest_community=min(len(c) for c in communities))

        except Exception as e:
            logger.error("Community detection failed", error=str(e))
            # Assign all nodes to single community as fallback
            for node_id in graph.nodes():
                graph.nodes[node_id]['community'] = 0

    async def _calculate_centrality_measures(self, graph: nx.Graph) -> None:
        """Calculate various centrality measures for nodes"""
        if graph.number_of_nodes() < 2:
            return

        try:
            # Degree centrality
            degree_centrality = nx.degree_centrality(graph)

            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(graph)

            # Eigenvector centrality (handle disconnected graphs)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                eigenvector_centrality = {node: 0.0 for node in graph.nodes()}

            # PageRank
            pagerank = nx.pagerank(graph)

            # Add centrality measures to nodes
            for node_id in graph.nodes():
                graph.nodes[node_id].update({
                    'degree_centrality': degree_centrality.get(node_id, 0.0),
                    'betweenness_centrality': betweenness_centrality.get(node_id, 0.0),
                    'eigenvector_centrality': eigenvector_centrality.get(node_id, 0.0),
                    'pagerank': pagerank.get(node_id, 0.0)
                })

        except Exception as e:
            logger.error("Centrality calculation failed", error=str(e))

    @log_performance(logger)
    async def identify_main_storylines(self, graph: nx.Graph) -> List[StorylineNode]:
        """Identify main storylines from graph communities"""
        if graph.number_of_nodes() == 0:
            return []

        logger.info("Identifying main storylines", graph_nodes=graph.number_of_nodes())

        # Group nodes by community
        communities = defaultdict(list)
        for node_id, node_data in graph.nodes(data=True):
            community_id = node_data.get('community', 0)
            communities[community_id].append((node_id, node_data))

        storylines = []
        for community_id, nodes in communities.items():
            storyline = await self._create_storyline_from_community(
                community_id, nodes, graph
            )
            storylines.append(storyline)

        # Sort by centrality score (most important first)
        storylines.sort(key=lambda s: s.centrality_score, reverse=True)

        # Mark top storylines as main storylines
        for i, storyline in enumerate(storylines[:self.main_storyline_count]):
            storyline.main_storyline = True

        logger.info("Main storylines identified",
                    total_storylines=len(storylines),
                    main_storylines=sum(1 for s in storylines if s.main_storyline))

        return storylines

    async def _create_storyline_from_community(self,
                                               community_id: int,
                                               nodes: List[Tuple[str, Dict]],
                                               graph: nx.Graph) -> StorylineNode:
        """Create storyline node from community of chunks"""
        node_ids = [node_id for node_id, _ in nodes]
        node_data_list = [data for _, data in nodes]

        # Calculate average centrality
        centrality_scores = []
        for _, data in nodes:
            score = (
                    data.get('degree_centrality', 0) * 0.3 +
                    data.get('betweenness_centrality', 0) * 0.3 +
                    data.get('pagerank', 0) * 0.4
            )
            centrality_scores.append(score)

        avg_centrality = np.mean(centrality_scores) if centrality_scores else 0.0

        # Aggregate participants
        all_participants = []
        for _, data in nodes:
            all_participants.extend(data.get('participants', []))

        participant_counts = Counter(all_participants)
        top_participants = [name for name, count in participant_counts.most_common(5)]

        # Aggregate themes
        all_themes = []
        for _, data in nodes:
            all_themes.extend(data.get('themes', []))

        theme_counts = Counter(all_themes)
        top_themes = [theme for theme, count in theme_counts.most_common(3)]

        # Generate summary (placeholder - would use LLM)
        summary = self._generate_storyline_summary(top_participants, top_themes, len(nodes))

        # Calculate temporal span
        temporal_span = self._calculate_temporal_span(node_data_list)

        # Calculate overall confidence
        confidences = [data.get('confidence', 0.5) for _, data in nodes]
        avg_confidence = np.mean(confidences) if confidences else 0.5

        # Calculate entity density
        entity_densities = [data.get('entity_density', 0) for _, data in nodes]
        avg_entity_density = np.mean(entity_densities) if entity_densities else 0.0

        return StorylineNode(
            id=f"storyline_{community_id}",
            summary=summary,
            temporal_info=self._aggregate_temporal_info(node_data_list),
            participants=top_participants,
            themes=top_themes,
            chunk_ids=node_ids,
            centrality_score=avg_centrality,
            confidence=avg_confidence,
            entity_density=avg_entity_density,
            temporal_span_days=temporal_span
        )

    def _generate_storyline_summary(self,
                                    participants: List[str],
                                    themes: List[str],
                                    chunk_count: int) -> str:
        """Generate a summary for the storyline"""
        # Simple template-based summary generation
        if participants and themes:
            return f"Storyline involving {', '.join(participants[:2])} related to {', '.join(themes[:2])} ({chunk_count} segments)"
        elif participants:
            return f"Storyline about {', '.join(participants[:3])} ({chunk_count} segments)"
        elif themes:
            return f"Storyline focused on {', '.join(themes[:2])} ({chunk_count} segments)"
        else:
            return f"General storyline with {chunk_count} segments"

    def _calculate_temporal_span(self, node_data_list: List[Dict]) -> Optional[float]:
        """Calculate temporal span of storyline in days"""
        # Placeholder implementation - would need proper date parsing
        dates = []
        for data in node_data_list:
            temporal_info = data.get('temporal_info', {})
            explicit_dates = temporal_info.get('explicit_dates', [])
            dates.extend(explicit_dates)

        if len(dates) >= 2:
            # Would need to parse actual dates and calculate span
            return 30.0  # Placeholder: assume 30 days average span

        return None

    def _aggregate_temporal_info(self, node_data_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate temporal information from multiple nodes"""
        aggregated = {
            'explicit_dates': [],
            'relative_expressions': [],
            'temporal_entities': []
        }

        for data in node_data_list:
            temporal_info = data.get('temporal_info', {})
            for key in aggregated.keys():
                if key in temporal_info:
                    aggregated[key].extend(temporal_info[key])

        # Remove duplicates while preserving order
        for key in aggregated.keys():
            seen = set()
            unique_items = []
            for item in aggregated[key]:
                item_key = item.get('text', str(item)) if isinstance(item, dict) else str(item)
                if item_key not in seen:
                    seen.add(item_key)
                    unique_items.append(item)
            aggregated[key] = unique_items

        return aggregated

    @log_performance(logger)
    async def save_graph_to_neo4j(self, graph: nx.Graph, session_id: str) -> bool:
        """Save graph structure to Neo4j database"""
        if not self.driver:
            logger.warning("No Neo4j connection available, skipping graph save")
            return False

        try:
            with self.driver.session() as session:
                # Clear existing data for this session
                session.run(
                    "MATCH (n) WHERE n.session_id = $session_id DETACH DELETE n",
                    session_id=session_id
                )

                # Create nodes
                for node_id, node_data in graph.nodes(data=True):
                    session.run("""
                        CREATE (n:Chunk {
                            id: $id,
                            session_id: $session_id,
                            content: $content,
                            word_count: $word_count,
                            speaker: $speaker,
                            confidence: $confidence,
                            community: $community,
                            degree_centrality: $degree_centrality,
                            pagerank: $pagerank,
                            participants: $participants,
                            themes: $themes
                        })
                    """,
                                id=node_id,
                                session_id=session_id,
                                content=node_data.get('content', ''),
                                word_count=node_data.get('word_count', 0),
                                speaker=node_data.get('speaker'),
                                confidence=node_data.get('confidence', 0.5),
                                community=node_data.get('community', 0),
                                degree_centrality=node_data.get('degree_centrality', 0.0),
                                pagerank=node_data.get('pagerank', 0.0),
                                participants=node_data.get('participants', []),
                                themes=node_data.get('themes', [])
                                )

                # Create relationships
                for source, target, edge_data in graph.edges(data=True):
                    session.run("""
                        MATCH (a:Chunk {id: $source, session_id: $session_id})
                        MATCH (b:Chunk {id: $target, session_id: $session_id})
                        CREATE (a)-[r:RELATED {
                            weight: $weight,
                            relationship_type: $rel_type,
                            confidence: $confidence
                        }]->(b)
                    """,
                                source=source,
                                target=target,
                                session_id=session_id,
                                weight=edge_data.get('weight', 0.0),
                                rel_type=edge_data.get('relationship_type', 'UNKNOWN'),
                                confidence=edge_data.get('confidence', 0.5)
                                )

                logger.info("Graph saved to Neo4j successfully",
                            session_id=session_id,
                            nodes_saved=graph.number_of_nodes(),
                            edges_saved=graph.number_of_edges())
                return True

        except Exception as e:
            logger.error("Failed to save graph to Neo4j",
                         error=str(e), session_id=session_id)
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Health check for graph builder service"""
        health_status = {
            "neo4j_connection": "disconnected",
            "graph_analysis": "ready",
            "community_detection": "ready",
            "centrality_calculation": "ready"
        }

        # Test Neo4j connection
        if self.driver:
            try:
                with self.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    result.single()
                    health_status["neo4j_connection"] = "connected"
            except Exception as e:
                logger.error("Neo4j health check failed", error=str(e))
                health_status["neo4j_connection"] = f"error: {str(e)}"

        return {
            "status": "healthy" if health_status["neo4j_connection"] != "disconnected" else "degraded",
            "components": health_status,
            "configuration": {
                "similarity_threshold": self.similarity_threshold,
                "entity_overlap_threshold": self.entity_overlap_threshold,
                "main_storyline_count": self.main_storyline_count
            }
        }

    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

