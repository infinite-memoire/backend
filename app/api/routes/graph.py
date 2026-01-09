from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.services.neo4j import neo4j_service
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/graph/full")
async def get_full_graph(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of nodes to retrieve")
):
    """
    Retrieve the complete graph structure with all nodes and relationships.
    
    Returns:
    - nodes: List of all story nodes with their properties
    - relationships: List of all relationships between nodes
    - metadata: Graph statistics and counts
    """
    try:
        graph_data = await neo4j_service.get_full_graph(limit=limit)
        return {
            "success": True,
            "data": graph_data,
            "message": f"Retrieved graph with {len(graph_data['nodes'])} nodes and {len(graph_data['relationships'])} relationships"
        }
    except Exception as e:
        logger.error("Failed to retrieve full graph", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve graph: {str(e)}")

@router.get("/graph/subgraph/{node_id}")
async def get_subgraph(
    node_id: str,
    depth: int = Query(default=2, ge=1, le=5, description="Depth of traversal from the center node")
):
    """
    Retrieve a subgraph centered around a specific node.
    
    Args:
    - node_id: The ID of the center node
    - depth: How many hops away from the center node to include
    
    Returns:
    - nodes: List of nodes within the specified depth
    - relationships: List of relationships between these nodes
    - center_node: The ID of the center node
    - depth: The traversal depth used
    """
    try:
        # First check if the center node exists
        center_node = await neo4j_service.get_story_node(node_id)
        if not center_node:
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
        
        subgraph_data = await neo4j_service.get_subgraph(node_id=node_id, depth=depth)
        return {
            "success": True,
            "data": subgraph_data,
            "message": f"Retrieved subgraph around node '{node_id}' with depth {depth}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve subgraph", node_id=node_id, depth=depth, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve subgraph: {str(e)}")

@router.get("/graph/statistics")
async def get_graph_statistics():
    """
    Get comprehensive statistics about the graph structure.
    
    Returns:
    - total_nodes: Total number of nodes in the graph
    - total_relationships: Total number of relationships
    - relationship_types: List of relationship types with counts
    - degree_distribution: Distribution of node degrees
    - node_types: List of node types (currently just StoryNode)
    """
    try:
        stats = await neo4j_service.get_graph_statistics()
        return {
            "success": True,
            "data": stats,
            "message": "Retrieved graph statistics"
        }
    except Exception as e:
        logger.error("Failed to retrieve graph statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")

@router.get("/graph/node/{node_id}")
async def get_node_details(node_id: str):
    """
    Get detailed information about a specific node and its connections.
    
    Args:
    - node_id: The ID of the node to retrieve
    
    Returns:
    - node: The node data
    - connected_nodes: List of directly connected nodes
    """
    try:
        # Get the node details
        node = await neo4j_service.get_story_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
        
        # Get connected nodes
        connected_nodes = await neo4j_service.get_connected_nodes(node_id)
        
        return {
            "success": True,
            "data": {
                "node": node,
                "connected_nodes": connected_nodes,
                "connections_count": len(connected_nodes)
            },
            "message": f"Retrieved details for node '{node_id}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve node details", node_id=node_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve node details: {str(e)}")

@router.get("/graph/health")
async def check_graph_health():
    """
    Check the health and connectivity of the Neo4j graph database.
    
    Returns:
    - connected: Whether the database is accessible
    - statistics: Basic graph statistics if connected
    """
    try:
        # Test connection
        await neo4j_service.test_connection()
        
        # Get basic statistics
        stats = await neo4j_service.get_graph_statistics()
        
        return {
            "success": True,
            "data": {
                "connected": True,
                "database_accessible": True,
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"]
            },
            "message": "Graph database is healthy and accessible"
        }
    except Exception as e:
        logger.warning("Graph database health check failed", error=str(e))
        return {
            "success": False,
            "data": {
                "connected": False,
                "database_accessible": False,
                "error": str(e)
            },
            "message": "Graph database is not accessible"
        }
