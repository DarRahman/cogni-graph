# cognigraph/graph_store.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Graph storage engine interface and NetworkX implementation."""

from datetime import datetime
import json
import logging
import os
from typing import List, Optional, Tuple, Protocol, runtime_checkable
import networkx as nx
from cognigraph.models import Entity, Relationship

logger = logging.getLogger("cognigraph.graph_store")


@runtime_checkable
class GraphStore(Protocol):
    """Protocol defining the interface for graph storage engines."""

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity in the graph."""
        ...

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieves an entity by its ID."""
        ...

    def get_all_entities(self) -> List[Entity]:
        """Retrieves all entities in the graph."""
        ...

    def remove_entity(self, entity_id: str) -> None:
        """Removes an entity and all its connected edges from the graph."""
        ...

    def add_relationship(self, relationship: Relationship) -> None:
        """Adds or updates a relationship in the graph."""
        ...

    def get_relationships(self, source_id: str) -> List[Relationship]:
        """Retrieves all outgoing relationships from a source entity."""
        ...

    def get_all_relationships(self) -> List[Relationship]:
        """Retrieves all relationships in the graph."""
        ...

    def remove_relationship(self, source: str, target: str, type: str) -> None:
        """Removes a specific relationship from the graph."""
        ...

    def get_neighbors(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Retrieves neighboring entities and the connecting relationships."""
        ...

    def get_degree(self, entity_id: str) -> int:
        """Returns the degree (number of connections) of an entity."""
        ...

    def merge_entities(self, entity_id_1: str, entity_id_2: str, merged_entity: Entity) -> None:
        """Merges two entities into a single entity, re-routing all edges."""
        ...

    def save_to_disk(self, path: str) -> None:
        """Saves the graph to disk (if supported)."""
        ...

    def load_from_disk(self, path: str) -> None:
        """Loads the graph from disk (if supported)."""
        ...


class NetworkXGraphStore:
    """In-memory graph store backed by NetworkX with serialization support."""

    def __init__(self) -> None:
        """Initializes an empty NetworkX MultiDiGraph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        logger.info("NetworkXGraphStore initialized")

    def add_entity(self, entity: Entity) -> None:
        """Adds or updates an entity in the graph.

        Args:
            entity: The Entity model to add.
        """
        node_attrs = entity.model_dump()
        # Convert datetime objects to ISO format strings for serialization
        node_attrs["created_at"] = node_attrs["created_at"].isoformat()
        node_attrs["updated_at"] = node_attrs["updated_at"].isoformat()

        self.graph.add_node(entity.id, **node_attrs)
        logger.debug("Added/updated entity: %s", entity.id)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieves an entity by its ID.

        Args:
            entity_id: The ID of the entity.

        Returns:
            The Entity model if found, else None.
        """
        if not self.graph.has_node(entity_id):
            return None
        node_data = self.graph.nodes[entity_id].copy()
        if isinstance(node_data.get("created_at"), str):
            node_data["created_at"] = datetime.fromisoformat(node_data["created_at"])
        if isinstance(node_data.get("updated_at"), str):
            node_data["updated_at"] = datetime.fromisoformat(node_data["updated_at"])
        return Entity(**node_data)

    def get_all_entities(self) -> List[Entity]:
        """Retrieves all entities in the graph.

        Returns:
            A list of all Entity models.
        """
        entities = []
        for node_id in self.graph.nodes:
            entity = self.get_entity(node_id)
            if entity:
                entities.append(entity)
        return entities

    def remove_entity(self, entity_id: str) -> None:
        """Removes an entity and all its connected edges from the graph.

        Args:
            entity_id: The ID of the entity to remove.
        """
        if self.graph.has_node(entity_id):
            self.graph.remove_node(entity_id)
            logger.info("Removed entity: %s", entity_id)

    def add_relationship(self, relationship: Relationship) -> None:
        """Adds or updates a relationship in the graph.

        If the relationship already exists (same source, target, and type),
        its weight is incremented.

        Args:
            relationship: The Relationship model to add.
        """
        # Ensure source and target nodes exist
        if not self.graph.has_node(relationship.source):
            self.add_entity(Entity(id=relationship.source, name=relationship.source.capitalize(), type="Concept"))
        if not self.graph.has_node(relationship.target):
            self.add_entity(Entity(id=relationship.target, name=relationship.target.capitalize(), type="Concept"))

        edge_attrs = relationship.model_dump()
        edge_attrs["created_at"] = edge_attrs["created_at"].isoformat()
        edge_attrs["updated_at"] = edge_attrs["updated_at"].isoformat()

        # Check if edge already exists to update weight
        source = relationship.source
        target = relationship.target
        rel_type = relationship.type

        exists = False
        if self.graph.has_edge(source, target):
            for key, data in self.graph[source][target].items():
                if data.get("type") == rel_type:
                    # Update existing edge weight and timestamp
                    new_weight = data.get("weight", 1.0) + relationship.weight
                    self.graph[source][target][key]["weight"] = new_weight
                    self.graph[source][target][key]["updated_at"] = edge_attrs["updated_at"]
                    exists = True
                    logger.debug("Updated relationship weight: %s -[%s]-> %s to %f", source, rel_type, target, new_weight)
                    break

        if not exists:
            self.graph.add_edge(source, target, key=rel_type, **edge_attrs)
            logger.debug("Added new relationship: %s -[%s]-> %s", source, rel_type, target)

    def get_relationships(self, source_id: str) -> List[Relationship]:
        """Retrieves all outgoing relationships from a source entity.

        Args:
            source_id: The ID of the source entity.

        Returns:
            A list of Relationship models.
        """
        relationships = []
        if not self.graph.has_node(source_id):
            return relationships

        for target_id in self.graph.successors(source_id):
            for key, edge_data in self.graph[source_id][target_id].items():
                data = edge_data.copy()
                if isinstance(data.get("created_at"), str):
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("updated_at"), str):
                    data["updated_at"] = datetime.fromisoformat(data["updated_at"])
                relationships.append(Relationship(**data))
        return relationships

    def get_all_relationships(self) -> List[Relationship]:
        """Retrieves all relationships in the graph.

        Returns:
            A list of all Relationship models.
        """
        relationships = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_data = data.copy()
            if isinstance(edge_data.get("created_at"), str):
                edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
            if isinstance(edge_data.get("updated_at"), str):
                edge_data["updated_at"] = datetime.fromisoformat(edge_data["updated_at"])
            relationships.append(Relationship(**edge_data))
        return relationships

    def remove_relationship(self, source: str, target: str, type: str) -> None:
        """Removes a specific relationship from the graph.

        Args:
            source: The source entity ID.
            target: The target entity ID.
            type: The relationship type.
        """
        if self.graph.has_edge(source, target, key=type):
            self.graph.remove_edge(source, target, key=type)
            logger.info("Removed relationship: %s -[%s]-> %s", source, type, target)

    def get_neighbors(self, entity_id: str) -> List[Tuple[Entity, Relationship]]:
        """Retrieves neighboring entities and the connecting relationships.

        Args:
            entity_id: The ID of the entity.

        Returns:
            A list of tuples containing (Neighbor Entity, Connecting Relationship).
        """
        neighbors = []
        if not self.graph.has_node(entity_id):
            return neighbors

        # Outgoing edges
        for target_id in self.graph.successors(entity_id):
            target_node = self.get_entity(target_id)
            if target_node:
                for key, edge_data in self.graph[entity_id][target_id].items():
                    data = edge_data.copy()
                    if isinstance(data.get("created_at"), str):
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                    if isinstance(data.get("updated_at"), str):
                        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
                    neighbors.append((target_node, Relationship(**data)))

        # Incoming edges
        for source_id in self.graph.predecessors(entity_id):
            source_node = self.get_entity(source_id)
            if source_node:
                for key, edge_data in self.graph[source_id][entity_id].items():
                    data = edge_data.copy()
                    if isinstance(data.get("created_at"), str):
                        data["created_at"] = datetime.fromisoformat(data["created_at"])
                    if isinstance(data.get("updated_at"), str):
                        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
                    neighbors.append((source_node, Relationship(**data)))

        return neighbors

    def get_degree(self, entity_id: str) -> int:
        """Returns the degree (number of connections) of an entity.

        Args:
            entity_id: The ID of the entity.

        Returns:
            The degree of the entity.
        """
        if not self.graph.has_node(entity_id):
            return 0
        return int(self.graph.degree(entity_id))

    def merge_entities(self, entity_id_1: str, entity_id_2: str, merged_entity: Entity) -> None:
        """Merges two entities into a single entity.

        All edges connected to entity_id_2 are re-routed to entity_id_1 (or the merged entity ID).
        Entity_id_2 is then removed.

        Args:
            entity_id_1: The ID of the first entity (to keep/merge into).
            entity_id_2: The ID of the second entity (to be merged and removed).
            merged_entity: The new Entity model representing the merged entity.
        """
        if entity_id_1 == entity_id_2:
            return

        logger.info("Merging entity %s into %s", entity_id_2, entity_id_1)

        if not self.graph.has_node(entity_id_1) or not self.graph.has_node(entity_id_2):
            logger.warning("Cannot merge: one or both entities do not exist")
            return

        # Add/update the merged entity
        self.add_entity(merged_entity)

        # Re-route incoming edges of entity_id_2 to merged_entity.id
        in_edges = list(self.graph.in_edges(entity_id_2, keys=True, data=True))
        for source, _, key, data in in_edges:
            if source == entity_id_2:
                # Self-loop, handle carefully
                source = merged_entity.id
            edge_data = data.copy()
            if isinstance(edge_data.get("created_at"), str):
                edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
            if isinstance(edge_data.get("updated_at"), str):
                edge_data["updated_at"] = datetime.fromisoformat(edge_data["updated_at"])
            rel = Relationship(**edge_data)
            rel.source = source
            rel.target = merged_entity.id
            self.add_relationship(rel)

        # Re-route outgoing edges of entity_id_2 to merged_entity.id
        out_edges = list(self.graph.out_edges(entity_id_2, keys=True, data=True))
        for _, target, key, data in out_edges:
            if target == entity_id_2:
                # Self-loop, handle carefully
                target = merged_entity.id
            edge_data = data.copy()
            if isinstance(edge_data.get("created_at"), str):
                edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
            if isinstance(edge_data.get("updated_at"), str):
                edge_data["updated_at"] = datetime.fromisoformat(edge_data["updated_at"])
            rel = Relationship(**edge_data)
            rel.source = merged_entity.id
            rel.target = target
            self.add_relationship(rel)

        # Remove the old entity
        self.graph.remove_node(entity_id_2)
        logger.info("Successfully merged %s into %s", entity_id_2, merged_entity.id)

    def save_to_disk(self, path: str) -> None:
        """Serializes the graph to a JSON file.

        Args:
            path: The file path to save the graph.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", path)

    def load_from_disk(self, path: str) -> None:
        """Deserializes the graph from a JSON file.

        Args:
            path: The file path to load the graph from.
        """
        if not os.path.exists(path):
            logger.warning("Graph file %s does not exist. Starting with empty graph.", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.graph = nx.node_link_graph(data)
        # Re-add 'id' to node attributes since node_link_graph strips it out by using it as node key
        for node_id, node_data in self.graph.nodes(data=True):
            if "id" not in node_data:
                node_data["id"] = node_id
        logger.info("Graph loaded from %s with %d nodes and %d edges", path, self.graph.number_of_nodes(), self.graph.number_of_edges())
