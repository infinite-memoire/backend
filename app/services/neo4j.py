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

# Global service instance
neo4j_service = Neo4jService()
