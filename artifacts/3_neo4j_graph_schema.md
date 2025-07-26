# Neo4j Graph Schema Design

Based on Firestore data models for storyline graph generation and chapter organization.

## Graph Database Purpose
Neo4j will handle the storyline graph where:
- Each node represents a semantic story element
- Nodes link to Firestore transcript chunk IDs
- Main nodes (high edge count) become chapters
- Secondary nodes are attributed to chapters by edge distance

## 1. Node Types and Labels

### 1.1 StoryNode (Primary Node Type)
```cypher
CREATE CONSTRAINT story_node_id FOR (n:StoryNode) REQUIRE n.id IS UNIQUE;

// Node properties
(:StoryNode {
  id: "story_node_uuid",                    // Unique identifier
  summary: "Brief description of story element",
  temporal_marker: "2020-05-15T00:00:00Z", // ISO format date (nullable)
  temporal_confidence: 0.85,               // Confidence in temporal assignment
  chunk_ids: ["chunk_uuid_1", "chunk_uuid_2"], // Array of Firestore chunk IDs
  node_type: "event",                      // event, person, location, theme
  importance_score: 0.78,                  // Calculated importance (0-1)
  edge_count: 5,                          // Number of connections (cached)
  is_main_node: false,                    // Auto-calculated based on edge_count
  assigned_chapters: ["chapter_1"],       // Chapters this node belongs to
  created_at: "2025-01-26T01:00:00Z",
  updated_at: "2025-01-26T01:00:00Z"
})
```

### 1.2 ChapterNode (Organizational Node Type)
```cypher
CREATE CONSTRAINT chapter_node_id FOR (n:ChapterNode) REQUIRE n.id IS UNIQUE;

// Chapter properties
(:ChapterNode {
  id: "chapter_uuid",
  chapter_number: 1,
  title: "Chapter Title",
  summary: "Chapter overview",
  temporal_range_start: "2020-01-01T00:00:00Z",
  temporal_range_end: "2020-12-31T23:59:59Z",
  story_node_count: 15,
  word_count_estimate: 2500,
  processing_status: "draft",
  created_at: "2025-01-26T01:00:00Z"
})
```

## 2. Relationship Types

### 2.1 StoryNode Relationships

#### RELATES_TO (Primary Story Relationship)
```cypher
// Semantic relationship between story elements
(n1:StoryNode)-[:RELATES_TO {
  relationship_type: "causal",           // causal, temporal, thematic, spatial
  strength: 0.85,                       // Relationship strength (0-1)
  direction: "bidirectional",           // unidirectional, bidirectional
  semantic_similarity: 0.72,           // Vector similarity score
  co_occurrence_count: 3,              // How often elements appear together
  confidence: 0.90,                    // Confidence in relationship
  created_by: "semantic_analysis",     // semantic_analysis, user_input
  created_at: "2025-01-26T01:00:00Z"
}]->(n2:StoryNode)
```

#### TEMPORAL_SEQUENCE (Time-based Ordering)
```cypher
// Chronological relationship between events
(earlier:StoryNode)-[:TEMPORAL_SEQUENCE {
  time_gap_days: 30,                   // Days between events
  sequence_confidence: 0.88,           // Confidence in temporal ordering
  inferred: false,                     // Whether relationship was inferred
  source: "explicit_date"              // explicit_date, relative_reference, inferred
}]->(later:StoryNode)
```

#### INVOLVES (Entity Participation)
```cypher
// Person/location involvement in events
(entity:StoryNode)-[:INVOLVES {
  role: "primary_participant",         // primary_participant, location, observer
  involvement_strength: 0.95,          // How central the entity is
  mention_count: 5,                    // Times mentioned in relation
  context: "family_gathering"          // Contextual description
}]->(event:StoryNode)
```

### 2.2 Chapter Organization Relationships

#### BELONGS_TO_CHAPTER
```cypher
// Story node assignment to chapters
(story:StoryNode)-[:BELONGS_TO_CHAPTER {
  assignment_confidence: 0.82,         // Confidence in chapter assignment
  assignment_method: "edge_distance",  // edge_distance, temporal_proximity, manual
  chapter_relevance: 0.90,            // How relevant to chapter theme
  is_primary_chapter: true,           // Primary vs secondary chapter assignment
  assigned_at: "2025-01-26T01:00:00Z"
}]->(chapter:ChapterNode)
```

#### CHAPTER_SEQUENCE
```cypher
// Ordering between chapters
(ch1:ChapterNode)-[:CHAPTER_SEQUENCE {
  sequence_number: 1,
  transition_strength: 0.75,          // How well chapters flow together
  temporal_gap_months: 6              // Time gap between chapter periods
}]->(ch2:ChapterNode)
```

## 3. Graph Analysis Algorithms

### 3.1 Main Node Identification
```cypher
// Identify main nodes based on edge count and centrality
MATCH (n:StoryNode)
WITH n, size((n)-[:RELATES_TO]-()) as edge_count
WHERE edge_count >= 5  // Threshold for main nodes
SET n.is_main_node = true, n.edge_count = edge_count
RETURN n.id, n.summary, edge_count
ORDER BY edge_count DESC;
```

### 3.2 Chapter Assignment Algorithm
```cypher
// Assign secondary nodes to chapters based on distance to main nodes
MATCH (main:StoryNode {is_main_node: true})-[:BELONGS_TO_CHAPTER]->(chapter:ChapterNode)
MATCH (secondary:StoryNode {is_main_node: false})
MATCH path = shortestPath((secondary)-[:RELATES_TO*1..3]-(main))
WITH secondary, chapter, length(path) as distance, main
WHERE distance <= 2  // Maximum distance from main node
MERGE (secondary)-[:BELONGS_TO_CHAPTER {
  assignment_method: "edge_distance",
  assignment_confidence: 1.0 - (distance * 0.2),
  is_primary_chapter: distance = 1
}]->(chapter);
```

### 3.3 Temporal Consistency Check
```cypher
// Identify nodes lacking temporal markers for follow-up questions
MATCH (n:StoryNode)
WHERE n.temporal_marker IS NULL
RETURN n.id, n.summary, n.chunk_ids
ORDER BY n.importance_score DESC;
```

## 4. Graph Queries for AI Agents

### 4.1 Chapter Writer Agent Queries

#### Load Main Node for Chapter
```cypher
MATCH (main:StoryNode {is_main_node: true})-[:BELONGS_TO_CHAPTER]->(chapter:ChapterNode {id: $chapter_id})
RETURN main.id, main.summary, main.chunk_ids, main.temporal_marker;
```

#### Load Adjacent Nodes (Tool Function)
```cypher
MATCH (main:StoryNode {id: $main_node_id})-[:RELATES_TO]-(adjacent:StoryNode)
RETURN adjacent.id, adjacent.summary, adjacent.chunk_ids, adjacent.temporal_marker
ORDER BY adjacent.importance_score DESC
LIMIT 10;
```

#### Get Chapter Context
```cypher
MATCH (story:StoryNode)-[:BELONGS_TO_CHAPTER]->(chapter:ChapterNode {id: $chapter_id})
RETURN story.id, story.summary, story.temporal_marker, story.chunk_ids
ORDER BY story.temporal_marker, story.importance_score DESC;
```

### 4.2 Chapter Harmonizer Agent Queries

#### Find Chapter Overlaps
```cypher
MATCH (story:StoryNode)-[:BELONGS_TO_CHAPTER]->(ch1:ChapterNode)
MATCH (story)-[:BELONGS_TO_CHAPTER]->(ch2:ChapterNode)
WHERE ch1 <> ch2
RETURN story.id, story.summary, ch1.title, ch2.title
ORDER BY story.importance_score DESC;
```

#### Get Chapter Boundaries
```cypher
MATCH (ch1:ChapterNode)-[:CHAPTER_SEQUENCE]->(ch2:ChapterNode)
MATCH (story1:StoryNode)-[:BELONGS_TO_CHAPTER]->(ch1)
MATCH (story2:StoryNode)-[:BELONGS_TO_CHAPTER]->(ch2)
WHERE story1.temporal_marker IS NOT NULL AND story2.temporal_marker IS NOT NULL
RETURN ch1.title, ch2.title, 
       max(story1.temporal_marker) as ch1_end,
       min(story2.temporal_marker) as ch2_start;
```

### 4.3 Follow-up Questions Agent Queries

#### Find Nodes Lacking Connections
```cypher
MATCH (n:StoryNode)
WHERE size((n)-[:RELATES_TO]-()) < 2  // Isolated or weakly connected nodes
RETURN n.id, n.summary, n.chunk_ids, size((n)-[:RELATES_TO]-()) as connection_count
ORDER BY n.importance_score DESC;
```

#### Find Temporal Gaps
```cypher
MATCH (n1:StoryNode)-[:TEMPORAL_SEQUENCE]->(n2:StoryNode)
WHERE n1.temporal_marker IS NOT NULL AND n2.temporal_marker IS NOT NULL
WITH n1, n2, duration.between(date(n1.temporal_marker), date(n2.temporal_marker)).days as gap_days
WHERE gap_days > 365  // Large temporal gaps
RETURN n1.summary, n2.summary, gap_days
ORDER BY gap_days DESC;
```

#### Identify Missing Temporal Information
```cypher
MATCH (n:StoryNode)
WHERE n.temporal_marker IS NULL AND n.importance_score > 0.5
RETURN n.id, n.summary, n.chunk_ids, n.importance_score
ORDER BY n.importance_score DESC
LIMIT 20;
```

## 5. Graph Performance Optimization

### 5.1 Indexes
```cypher
// Core indexes for performance
CREATE INDEX story_node_importance FOR (n:StoryNode) ON (n.importance_score);
CREATE INDEX story_node_temporal FOR (n:StoryNode) ON (n.temporal_marker);
CREATE INDEX story_node_main FOR (n:StoryNode) ON (n.is_main_node);
CREATE INDEX chapter_sequence FOR (n:ChapterNode) ON (n.chapter_number);
CREATE INDEX relationship_strength FOR ()-[r:RELATES_TO]-() ON (r.strength);
```

### 5.2 Graph Statistics
```cypher
// Maintain graph statistics for optimization
MATCH (n:StoryNode) 
WITH count(n) as total_nodes, 
     count(CASE WHEN n.is_main_node THEN 1 END) as main_nodes,
     avg(n.importance_score) as avg_importance
MATCH ()-[r:RELATES_TO]-()
WITH total_nodes, main_nodes, avg_importance, count(r) as total_relationships
RETURN total_nodes, main_nodes, total_relationships, avg_importance;
```

## 6. Data Integration with Firestore

### 6.1 Chunk ID References
- Each StoryNode contains `chunk_ids` array pointing to Firestore text_chunks
- Firestore chunks contain `neo4j_node_id` for reverse lookup
- Bidirectional references maintained for data consistency

### 6.2 Synchronization Strategy
- Neo4j nodes created during semantic analysis phase
- Chunk assignments updated in both systems simultaneously
- No complex synchronization - simple ID-based linking
- Orphaned references cleaned up by background tasks

## 7. Graph Evolution

### 7.1 Node Creation Workflow
1. Semantic analysis creates StoryNodes from text chunks
2. Relationships established based on semantic similarity
3. Temporal information extracted and assigned
4. Main nodes identified by edge count analysis
5. Chapter nodes created for main nodes
6. Secondary nodes assigned to chapters by proximity

### 7.2 Continuous Refinement
- Relationship strengths updated as more content is processed
- Chapter assignments refined based on new connections
- Temporal information improved through user feedback
- Node importance scores recalculated periodically