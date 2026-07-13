# cognigraph/neo4j_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Graph storage engine using Neo4j."""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from cognigraph.models import Entity, Relationship

logger = logging.getLogger("cognigraph.neo4j_store")

try:
    import neo4j
    NEO4J_AVAILABLE = True
except ImportError:
    import sys
    from unittest.mock import MagicMock
    # Mock neo4j for testing / when not installed
    sys.modules['neo4j'] = MagicMock()
    import neo4j
    NEO4J_AVAILABLE = False


def sanitize_rel_type(rel_type: str) -> str:
    """Sanitizes relationship type to prevent Cypher injection.

    Only allows uppercase alphanumeric characters and underscores.
    """
    sanitized = re.sub(r'[^A-Z0-9_]', '', rel_type.upper())
    if not sanitized:
        return "RELATED_TO"
    return sanitized


class Neo4jGraphStore:
    """Neo4j graph store implementing the GraphStore protocol."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        """Initializes the Neo4j driver.

        Args:
            uri: Neo4j connection URI.
            user: Username.
            password: Password.
            database: Database name.
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "The 'neo4j' package is required to use Neo4jGraphStore. "
                "Install it using 'poetry add neo4j' or 'pip install neo4j'."
            ) 
        self.driver: Any = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info("Neo4jGraphStore initialized with URI: %s", uri)

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity in Neo4j.

        Args:
            entity: The Entity model to add.
        """
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.description = $description,
            e.properties_json = $properties_json,
            e.created_at = $created_at,
            e.updated_at = $updated_at
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                id=entity.id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                properties_json=json.dumps(entity.properties),
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat()
            )
        logger.debug("Added/updated entity in Neo4j: %s", entity.id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieves an entity by its ID from Neo4j.

        Args:
            entity_id: The ID of the entity.

        Returns:
            The Entity model if found, else None.
        """
        query = "MATCH (e:Entity {id: $id}) RETURN e"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=entity_id)
            record = result.single()
            if not record:
                return None
            node = record["e"]
            props = dict(node)
            return Entity(
                id=props["id"],
                name=props["name"],
                type=props["type"],
                description=props.get("description", ""),
                properties=json.loads(props.get("properties_json", "{}")),
                created_at=datetime.fromisoformat(props["created_at"]),
                updated_at=datetime.fromisoformat(props["updated_at"])
            )

    def get_all_entities(self) -> List[Entity]:
        """Retrieves all entities in the Neo4j graph.

        Returns:
            A list of all Entity models.
        """
        query = "MATCH (e:Entity) RETURN e"
        entities = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                node = record["e"]
                props = dict(node)
                entities.append(
                    Entity(
                        id=props["id"],
                        name=props["name"],
                        type=props["type"],
                        description=props.get("description", ""),
                        properties=json.loads(props.get("properties_json", "{}")),
                        created_at=datetime.fromisoformat(props["created_at"]),
                        updated_at=datetime.fromisoformat(props["updated_at"])
                    )
                )
        return entities

    def remove_entity(self, entity_id: str) -> None:
        """Removes an entity and all its connected edges from Neo4j.

        Args:
            entity_id: The ID of the entity to remove.
        """
        query = "MATCH (e:Entity {id: $id}) DETACH DELETE e"
        with self.driver.session(database=self.database) as session:
            session.run(query, id=entity_id)
        logger.info("Removed entity from Neo4j: %s", entity_id)

    def add_relationship(self, relationship: Relationship) -> None:
        """Adds or updates a relationship in Neo4j.

        Args:
            relationship: The Relationship model to add.
        """
        rel_type = sanitize_rel_type(relationship.type)
        query = f"""
        MERGE (s:Entity {{id: $source}})
        MERGE (t:Entity {{id: $target}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r.weight = $weight,
                      r.description = $description,
                      r.properties_json = $properties_json,
                      r.created_at = $created_at,
                      r.updated_at = $updated_at
        ON MATCH SET r.weight = r.weight + $weight,
                     r.updated_at = $updated_at,
                     r.description = $description,
                     r.properties_json = $properties_json
        """
        with self.driver.session(database=self.database) as session:
            session.run(
                query,
                source=relationship.source,
                target=relationship.target,
                weight=relationship.weight,
                description=relationship.description,
                properties_json=json.dumps(relationship.properties),
                created_at=relationship.created_at.isoformat(),
                updated_at=relationship.updated_at.isoformat()
            )
        logger.debug("Added/updated relationship in Neo4j: %s -[%s]-> %s", relationship.source, rel_type, relationship.target)

    def get_relationships(self, source_id: str) -> List[Relationship]:
        """Retrieves all outgoing relationships from a source entity in Neo4j.

        Args:
            source_id: The ID of the source entity.

        Returns:
            A list of Relationship models.
        """
        query = """
        MATCH (s:Entity {id: $source_id})-[r]->(t:Entity)
        RETURN s.id AS source, t.id AS target, type(r) AS type, 
               r.description AS description, r.weight AS weight, 
               r.properties_json AS properties_json, 
               r.created_at AS created_at, r.updated_at AS updated_at
        """
        relationships = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, source_id=source_id)
            for record in result:
                relationships.append(
                    Relationship(
                        source=record["source"],
                        target=record["target"],
                        type=record["type"],
                        description=record["description"] or "",
                        weight=record["weight"] or 1.0,
                        properties=json.loads(record["properties_json"] or "{}"),
                        created_at=datetime.fromisoformat(record["created_at"]),
                        updated_at=datetime.fromisoformat(record["updated_at"])
                    )
                )
        return relationships

    def get_all_relationships(self) -> List[Relationship]:
        """Retrieves all relationships in the Neo4j graph.

        Returns:
            A list of all Relationship models.
        """
        query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        RETURN s.id AS source, t.id AS target, type(r) AS type, 
               r.description AS description, r.weight AS weight, 
               r.properties_json AS properties_json, 
               r.created_at AS created_at, r.updated_at AS updated_at
        """
        relationships = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                relationships.append(
                    Relationship(
                        source=record["source"],
                        target=record["target"],
                        type=record["type"],
                        description=record["description"] or "",
                        weight=record["weight"] or 1.0,
                        properties=json.loads(record["properties_json"] or "{}"),
                        created_at=datetime.fromisoformat(record["created_at"]),
                        updated_at=datetime.fromisoformat(record["updated_at"])
                    )
                )
        return relationships

    def remove_relationship(self, source: str, target: str, type: str) -> None:
        """Removes a specific relationship from Neo4j.

        Args:
            source: The source entity ID.
            target: The target entity ID.
            type: The relationship type.
        """
        rel_type = sanitize_rel_type(type)
        query = f"""
        MATCH (s:Entity {{id: $source}})-[r:{rel_type}]->(t:Entity {{id: $target}})
        DELETE r
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, source=source, target=target)
        logger.info("Removed relationship from Neo4j: %s -[%s]-> %s", source, rel_type, target)

    def get_neighbors(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Retrieves neighboring entities and the connecting relationships from Neo4j.

        Args:
            entity_id: The ID of the entity.

        Returns:
            A list of tuples containing (Neighbor Entity, Connecting Relationship).
        """
        query = """
        MATCH (e:Entity {id: $entity_id})-[r]-(n:Entity)
        RETURN n, r, startNode(r).id = e.id AS is_outgoing
        """
        neighbors = []
        with self.driver.session(database=self.database) as session:
            result = session.run(query, entity_id=entity_id)
            for record in result:
                node = record["n"]
                rel = record["r"]
                is_outgoing = record["is_outgoing"]
                
                node_props = dict(node)
                neighbor = Entity(
                    id=node_props["id"],
                    name=node_props["name"],
                    type=node_props["type"],
                    description=node_props.get("description", ""),
                    properties=json.loads(node_props.get("properties_json", "{}")),
                    created_at=datetime.fromisoformat(node_props["created_at"]),
                    updated_at=datetime.fromisoformat(node_props["updated_at"])
                )
                
                rel_props = dict(rel)
                relationship = Relationship(
                    source=entity_id if is_outgoing else neighbor.id,
                    target=neighbor.id if is_outgoing else entity_id,
                    type=rel.type,
                    description=rel_props.get("description", ""),
                    weight=rel_props.get("weight", 1.0),
                    properties=json.loads(rel_props.get("properties_json", "{}")),
                    created_at=datetime.fromisoformat(rel_props["created_at"]),
                    updated_at=datetime.fromisoformat(rel_props["updated_at"])
                )
                neighbors.append((neighbor, relationship))
        return neighbors

    def get_degree(self, entity_id: str) -> int:
        """Returns the degree (number of connections) of an entity in Neo4j.

        Args:
            entity_id: The ID of the entity.

        Returns:
            The degree of the entity.
        """
        query = "MATCH (e:Entity {id: $id})-[r]-() RETURN count(r) AS degree"
        with self.driver.session(database=self.database) as session:
            result = session.run(query, id=entity_id)
            record = result.single()
            return int(record["degree"]) if record else 0

    def merge_entities(self, entity_id_1: str, entity_id_2: str, merged_entity: Entity) -> None:
        """Merges two entities into a single entity in Neo4j.

        All edges connected to entity_id_2 are re-routed to entity_id_1 (or the merged entity ID).
        Entity_id_2 is then removed.

        Args:
            entity_id_1: The ID of the first entity (to keep/merge into).
            entity_id_2: The ID of the second entity (to be merged and removed).
            merged_entity: The new Entity model representing the merged entity.
        """
        if entity_id_1 == entity_id_2:
            return

        logger.info("Merging entity %s into %s in Neo4j", entity_id_2, entity_id_1)
        
        # 1. Update/create the merged entity node
        self.add_entity(merged_entity)
        
        # 2. Re-route outgoing relationships of entity_id_2 to entity_id_1
        out_query = """
        MATCH (s:Entity {id: $id2})-[r]->(t:Entity)
        RETURN type(r) AS type, t.id AS target, r.weight AS weight, 
               r.description AS description, r.properties_json AS properties_json, 
               r.created_at AS created_at, r.updated_at AS updated_at
        """
        
        with self.driver.session(database=self.database) as session:
            out_result = session.run(out_query, id2=entity_id_2)
            for record in out_result:
                rel_type = sanitize_rel_type(record["type"])
                target = record["target"]
                if target == entity_id_2:
                    target = entity_id_1
                    
                create_out_query = f"""
                MATCH (s:Entity {{id: $id1}})
                MATCH (t:Entity {{id: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                ON CREATE SET r.weight = $weight,
                              r.description = $description,
                              r.properties_json = $properties_json,
                              r.created_at = $created_at,
                              r.updated_at = $updated_at
                ON MATCH SET r.weight = r.weight + $weight,
                             r.updated_at = $updated_at
                """
                session.run(
                    create_out_query,
                    id1=entity_id_1,
                    target=target,
                    weight=record["weight"] or 1.0,
                    description=record["description"] or "",
                    properties_json=record["properties_json"] or "{}",
                    created_at=record["created_at"],
                    updated_at=record["updated_at"]
                )
                
            # 3. Re-route incoming relationships of entity_id_2 to entity_id_1
            in_query = """
            MATCH (s:Entity)-[r]->(t:Entity {id: $id2})
            RETURN type(r) AS type, s.id AS source, r.weight AS weight, 
                   r.description AS description, r.properties_json AS properties_json, 
                   r.created_at AS created_at, r.updated_at AS updated_at
            """
            in_result = session.run(in_query, id2=entity_id_2)
            for record in in_result:
                rel_type = sanitize_rel_type(record["type"])
                source = record["source"]
                if source == entity_id_2:
                    source = entity_id_1
                    
                create_in_query = f"""
                MATCH (s:Entity {{id: $source}})
                MATCH (t:Entity {{id: $id1}})
                MERGE (s)-[r:{rel_type}]->(t)
                ON CREATE SET r.weight = $weight,
                              r.description = $description,
                              r.properties_json = $properties_json,
                              r.created_at = $created_at,
                              r.updated_at = $updated_at
                ON MATCH SET r.weight = r.weight + $weight,
                             r.updated_at = $updated_at
                """
                session.run(
                    create_in_query,
                    source=source,
                    id1=entity_id_1,
                    weight=record["weight"] or 1.0,
                    description=record["description"] or "",
                    properties_json=record["properties_json"] or "{}",
                    created_at=record["created_at"],
                    updated_at=record["updated_at"]
                )
                
            # 4. Delete entity_id_2 and its relationships
            delete_query = "MATCH (e:Entity {id: $id2}) DETACH DELETE e"
            session.run(delete_query, id2=entity_id_2)
            
        logger.info("Successfully merged %s into %s in Neo4j", entity_id_2, entity_id_1)

    def decay_and_prune(self, decay_factor: float, threshold: float) -> None:
        """Optimized batch decay and pruning for Neo4j.

        Args:
            decay_factor: Daily decay rate.
            threshold: Weight threshold below which relationships are pruned.
        """
        # 1. Decay weights
        decay_query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WITH r, duration.inSeconds(datetime(r.updated_at), datetime()).seconds / 86400.0 AS elapsed_days
        WITH r, r.weight * exp(elapsed_days * log($decay_factor)) AS new_weight
        SET r.weight = new_weight, r.updated_at = datetime().isoformat()
        """
        
        # 2. Prune relationships below threshold
        prune_rel_query = """
        MATCH (s:Entity)-[r]->(t:Entity)
        WHERE r.weight < $threshold
        DELETE r
        """
        
        # 3. Prune isolated nodes
        prune_node_query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[]-()
        DETACH DELETE e
        """
        
        with self.driver.session(database=self.database) as session:
            session.run(decay_query, decay_factor=decay_factor)
            session.run(prune_rel_query, threshold=threshold)
            session.run(prune_node_query)
            
        logger.info("Neo4j decay and pruning completed")

    def save_to_disk(self, path: str) -> None:
        """No-op for Neo4j since it is persistent by default."""
        logger.warning("save_to_disk is a no-op for Neo4jGraphStore")

    def load_from_disk(self, path: str) -> None:
        """No-op for Neo4j since it is persistent by default."""
        logger.warning("load_from_disk is a no-op for Neo4jGraphStore")
