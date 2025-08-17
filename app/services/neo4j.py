from neo4j import GraphDatabase
from app.config.settings_config import get_settings
from app.utils.logging_utils import get_logger
import asyncio
from typing import Dict, List, Optional

logger = get_logger(__name__)

class Neo4jService:
    def __init__(self):
        settings = get_settings()
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                settings.database.neo4j_uri,
                auth=(settings.database.neo4j_user, settings.database.neo4j_password)
            )
            self.driver.verify_connectivity()
            logger.info("Neo4j driver initialized", uri=settings.database.neo4j_uri)
        except Exception as e:
            logger.warning("Neo4j connection failed, running in offline mode", error=str(e))
            self.driver = None
        
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver connection closed")
        
    async def test_connection(self):
        """Test Neo4j connection"""
        if not self.driver:
            logger.warning("Neo4j driver not available, skipping connection test")
            return
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("Neo4j connection test successful")
                else:
                    raise Exception("Unexpected test result")
        except Exception as e:
            logger.error("Neo4j connection test failed", error=str(e))
            raise
            
    async def create_story_node(self, node_data: dict) -> str:
        """Create a story node in the graph"""
        if not self.driver:
            logger.warning("Neo4j driver not available, skipping story node creation")
            return "offline_node_" + node_data.get("id", "unknown")
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._create_node, node_data)
                logger.info("Created story node", node_id=result)
                return result
        except Exception as e:
            logger.error("Failed to create story node", error=str(e))
            raise
            
    @staticmethod
    def _create_node(tx, node_data):
        """Transaction function to create a node"""
        query = """
        CREATE (n:StoryNode {
            id: $id,
            summary: $summary,
            temporal_marker: $temporal_marker,
            chunk_ids: $chunk_ids
        })
        RETURN n.id as node_id
        """
        result = tx.run(query, **node_data)
        return result.single()["node_id"]
        
    async def create_relationship(self, from_node_id: str, to_node_id: str, 
                                relationship_type: str, properties: dict = None) -> bool:
        """Create a relationship between two nodes"""
        if not self.driver:
            logger.warning("Neo4j driver not available, skipping relationship creation")
            return False
        try:
            with self.driver.session() as session:
                result = session.execute_write(
                    self._create_relationship, 
                    from_node_id, to_node_id, relationship_type, properties or {}
                )
                logger.info("Created relationship", 
                           from_node=from_node_id, 
                           to_node=to_node_id, 
                           type=relationship_type)
                return result
        except Exception as e:
            logger.error("Failed to create relationship", 
                        from_node=from_node_id, 
                        to_node=to_node_id, 
                        type=relationship_type, 
                        error=str(e))
            raise
            
    @staticmethod
    def _create_relationship(tx, from_node_id, to_node_id, relationship_type, properties):
        """Transaction function to create a relationship"""
        query = f"""
        MATCH (a:StoryNode {{id: $from_id}})
        MATCH (b:StoryNode {{id: $to_id}})
        CREATE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        RETURN count(r) as created
        """
        result = tx.run(query, 
                       from_id=from_node_id, 
                       to_id=to_node_id, 
                       properties=properties)
        return result.single()["created"] > 0
        
    async def get_story_node(self, node_id: str) -> Optional[dict]:
        """Get a story node by ID"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_node, node_id)
                if result:
                    logger.debug("Retrieved story node", node_id=node_id)
                else:
                    logger.warning("Story node not found", node_id=node_id)
                return result
        except Exception as e:
            logger.error("Failed to get story node", node_id=node_id, error=str(e))
            raise
            
    @staticmethod
    def _get_node(tx, node_id):
        """Transaction function to get a node"""
        query = """
        MATCH (n:StoryNode {id: $node_id})
        RETURN n.id as id, n.summary as summary, 
               n.temporal_marker as temporal_marker, 
               n.chunk_ids as chunk_ids
        """
        result = tx.run(query, node_id=node_id)
        record = result.single()
        return dict(record) if record else None
        
    async def get_connected_nodes(self, node_id: str) -> List[dict]:
        """Get all nodes connected to a given node"""
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_connected_nodes, node_id)
                logger.debug("Retrieved connected nodes", 
                           node_id=node_id, 
                           connected_count=len(result))
                return result
        except Exception as e:
            logger.error("Failed to get connected nodes", 
                        node_id=node_id, error=str(e))
            raise
            
    @staticmethod
    def _get_connected_nodes(tx, node_id):
        """Transaction function to get connected nodes"""
        query = """
        MATCH (n:StoryNode {id: $node_id})-[r]-(connected:StoryNode)
        RETURN connected.id as id, connected.summary as summary,
               connected.temporal_marker as temporal_marker,
               type(r) as relationship_type
        """
        result = tx.run(query, node_id=node_id)
        return [dict(record) for record in result]
        
    async def delete_node(self, node_id: str) -> bool:
        """Delete a story node and all its relationships"""
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._delete_node, node_id)
                logger.info("Deleted story node", node_id=node_id)
                return result
        except Exception as e:
            logger.error("Failed to delete story node", node_id=node_id, error=str(e))
            raise
            
    @staticmethod
    def _delete_node(tx, node_id):
        """Transaction function to delete a node"""
        query = """
        MATCH (n:StoryNode {id: $node_id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        result = tx.run(query, node_id=node_id)
        return result.single()["deleted"] > 0

    async def get_full_graph(self, limit: int = 100) -> dict:
        """Get the complete graph structure with nodes and relationships"""
        if not self.driver:
            logger.warning("Neo4j driver not available, returning empty graph")
            return {"nodes": [], "relationships": [], "metadata": {"total_nodes": 0, "total_relationships": 0}}
        
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_full_graph, limit)
                logger.info("Retrieved full graph", 
                           nodes_count=len(result["nodes"]), 
                           relationships_count=len(result["relationships"]))
                return result
        except Exception as e:
            logger.error("Failed to get full graph", error=str(e))
            raise

    @staticmethod
    def _get_full_graph(tx, limit):
        """Transaction function to get the complete graph"""
        # Get all nodes (check for both Chunk and StoryNode types)
        nodes_query = """
        MATCH (n)
        WHERE n:Chunk OR n:StoryNode
        RETURN n.id as id, 
               COALESCE(n.content, n.summary) as content, 
               n.session_id as session_id,
               n.word_count as word_count,
               n.confidence as confidence,
               labels(n) as node_type
        LIMIT $limit
        """
        nodes_result = tx.run(nodes_query, limit=limit)
        nodes = [dict(record) for record in nodes_result]
        
        # Get all relationships
        relationships_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:Chunk OR a:StoryNode) AND (b:Chunk OR b:StoryNode)
        RETURN a.id as source, b.id as target, type(r) as type, properties(r) as properties
        LIMIT $limit
        """
        rel_result = tx.run(relationships_query, limit=limit)
        relationships = [dict(record) for record in rel_result]
        
        # Get metadata
        stats_query = """
        MATCH (n)
        WHERE n:Chunk OR n:StoryNode
        WITH count(n) as node_count
        MATCH (a)-[r]->(b)
        WHERE (a:Chunk OR a:StoryNode) AND (b:Chunk OR b:StoryNode)
        RETURN node_count, count(r) as rel_count
        """
        stats_result = tx.run(stats_query)
        stats = stats_result.single()
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "total_nodes": stats["node_count"] if stats else 0,
                "total_relationships": stats["rel_count"] if stats else 0,
                "retrieved_nodes": len(nodes),
                "retrieved_relationships": len(relationships)
            }
        }

    async def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        """Get a subgraph starting from a specific node with given depth"""
        if not self.driver:
            logger.warning("Neo4j driver not available, returning empty subgraph")
            return {"nodes": [], "relationships": [], "center_node": node_id, "depth": depth}
        
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_subgraph, node_id, depth)
                logger.info("Retrieved subgraph", 
                           center_node=node_id, 
                           depth=depth,
                           nodes_count=len(result["nodes"]), 
                           relationships_count=len(result["relationships"]))
                return result
        except Exception as e:
            logger.error("Failed to get subgraph", node_id=node_id, depth=depth, error=str(e))
            raise

    @staticmethod
    def _get_subgraph(tx, node_id, depth):
        """Transaction function to get a subgraph"""
        # Get nodes within specified depth
        nodes_query = f"""
        MATCH path = (center:StoryNode {{id: $node_id}})-[*0..{depth}]-(connected:StoryNode)
        WITH DISTINCT connected
        RETURN connected.id as id, connected.summary as summary,
               connected.temporal_marker as temporal_marker,
               connected.chunk_ids as chunk_ids
        """
        nodes_result = tx.run(nodes_query, node_id=node_id)
        nodes = [dict(record) for record in nodes_result]
        
        # Get relationships between these nodes
        node_ids = [node["id"] for node in nodes]
        if node_ids:
            relationships_query = """
            MATCH (a:StoryNode)-[r]->(b:StoryNode)
            WHERE a.id IN $node_ids AND b.id IN $node_ids
            RETURN a.id as source, b.id as target, type(r) as type, properties(r) as properties
            """
            rel_result = tx.run(relationships_query, node_ids=node_ids)
            relationships = [dict(record) for record in rel_result]
        else:
            relationships = []
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "center_node": node_id,
            "depth": depth,
            "metadata": {
                "nodes_count": len(nodes),
                "relationships_count": len(relationships)
            }
        }

    async def get_graph_statistics(self) -> dict:
        """Get graph statistics and metadata"""
        if not self.driver:
            logger.warning("Neo4j driver not available, returning empty statistics")
            return {"total_nodes": 0, "total_relationships": 0, "node_types": [], "relationship_types": []}
        
        try:
            with self.driver.session() as session:
                result = session.execute_read(self._get_graph_statistics)
                logger.info("Retrieved graph statistics", **result)
                return result
        except Exception as e:
            logger.error("Failed to get graph statistics", error=str(e))
            raise

    @staticmethod
    def _get_graph_statistics(tx):
        """Transaction function to get graph statistics"""
        # Basic counts
        counts_query = """
        MATCH (n)
        WHERE n:Chunk OR n:StoryNode
        WITH count(n) as node_count
        MATCH (a)-[r]->(b)
        WHERE (a:Chunk OR a:StoryNode) AND (b:Chunk OR b:StoryNode)
        RETURN node_count, count(r) as rel_count
        """
        counts_result = tx.run(counts_query)
        counts = counts_result.single()
        
        # Relationship types
        rel_types_query = """
        MATCH (a)-[r]->(b)
        WHERE (a:Chunk OR a:StoryNode) AND (b:Chunk OR b:StoryNode)
        RETURN DISTINCT type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        rel_types_result = tx.run(rel_types_query)
        relationship_types = [dict(record) for record in rel_types_result]
        
        # Node degree distribution
        degree_query = """
        MATCH (n)
        WHERE n:Chunk OR n:StoryNode
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN degree, count(n) as node_count
        ORDER BY degree
        """
        degree_result = tx.run(degree_query)
        degree_distribution = [dict(record) for record in degree_result]
        
        # Get node type counts
        node_types_query = """
        MATCH (n)
        WHERE n:Chunk OR n:StoryNode
        RETURN DISTINCT labels(n) as node_type, count(n) as count
        ORDER BY count DESC
        """
        node_types_result = tx.run(node_types_query)
        node_types = [dict(record) for record in node_types_result]
        
        return {
            "total_nodes": counts["node_count"] if counts else 0,
            "total_relationships": counts["rel_count"] if counts else 0,
            "relationship_types": relationship_types,
            "degree_distribution": degree_distribution,
            "node_types": node_types
        }

# Global service instance
neo4j_service = Neo4jService()
