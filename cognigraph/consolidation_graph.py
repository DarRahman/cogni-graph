# cognigraph/consolidation_graph.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""LangGraph workflow to orchestrate the asynchronous consolidation loop (compaction, pruning, and forgetting)."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import networkx as nx

from cognigraph.config import settings
from cognigraph.extractor import Extractor
from cognigraph.graph_store import GraphStore
from cognigraph.models import ChatMessage, Entity, ExtractionResult, Relationship
from cognigraph.pipeline import (
    ConsolidationPipeline,
    MockEmbedder,
    cosine_similarity,
    jaro_winkler_similarity,
)
from cognigraph.vector_store import VectorStore
from cognigraph.episodic_buffer import EpisodicBuffer

logger = logging.getLogger("cognigraph.consolidation_graph")


class ConsolidationState(TypedDict):
    """State for the LangGraph consolidation workflow."""

    session_id: Optional[str]
    unprocessed_message_ids: List[str]
    messages: List[ChatMessage]
    extracted_result: Optional[ExtractionResult]
    merged_entities_count: int
    pruned_relationships_count: int
    forgotten_entities_count: int
    decay_factor: float
    pruning_threshold: float
    similarity_threshold: float
    forgetting_age_days: float
    status: str
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime]


class LangGraphConsolidator(ConsolidationPipeline):
    """Orchestrates the memory consolidation loop using a LangGraph workflow."""

    def __init__(
        self, 
        graph_store: GraphStore,
        vector_store: VectorStore,
        extractor: Extractor,
        embedder: MockEmbedder,
        episodic_buffer: Optional[EpisodicBuffer] = None,
    ) -> None: 
        """Initializes the LangGraph consolidator.

        Args:
            graph_store: The graph storage instance.
            vector_store: The vector storage instance.
            extractor: The entity-relationship extractor.
            embedder: The embedding generator.
            episodic_buffer: Optional episodic buffer instance.
        """
        super().__init__(graph_store, vector_store, extractor, embedder, episodic_buffer)
        self.workflow = self._build_workflow()
        logger.info("LangGraphConsolidator initialized")

    def fetch_unprocessed(self, state: ConsolidationState) -> Dict[str, Any]:
        """Fetches unprocessed messages from the episodic buffer.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [fetch_unprocessed]: Fetching unprocessed messages")
        if not self.episodic_buffer:
            logger.warning("No episodic buffer configured. Skipping fetch.")
            return {
                "unprocessed_message_ids": [],
                "messages": [],
                "status": "no_buffer",
            }

        session_id = state.get("session_id")
        unprocessed = self.episodic_buffer.get_messages(
            session_id=session_id,
            unprocessed_only=True,
        )

        msg_ids = [msg.id for msg in unprocessed]
        chat_msgs = [ 
            ChatMessage(
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata,
            )
            for msg in unprocessed
        ]

        logger.info("Found %d unprocessed messages", len(msg_ids))
        return {
            "unprocessed_message_ids": msg_ids,
            "messages": chat_msgs,
            "status": "fetched" if msg_ids else "no_new_messages",
        }

    def extract_facts(self, state: ConsolidationState) -> Dict[str, Any]:
        """Extracts entities and relationships from the fetched messages.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [extract_facts]: Extracting facts from messages")
        messages = state.get("messages", [])
        if not messages:
            return {"extracted_result": None, "status": "no_messages_to_extract"}

        try: 
            result = self.extractor.extract(messages)
            logger.info(
                "Extracted %d entities and %d relationships",
                len(result.entities),
                len(result.relationships),
            )
            return {"extracted_result": result, "status": "extracted"}
        except Exception as e:
            logger.exception("Failed to extract facts")
            return {
                "errors": state.get("errors", []) + [f"Extraction error: {str(e)}"],
                "status": "extraction_failed",
            }

    def merge_facts(self, state: ConsolidationState) -> Dict[str, Any]:
        """Inserts the extracted entities and relationships into the stores.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [merge_facts]: Merging extracted facts into stores")
        result = state.get("extracted_result")
        if not result:
            return {"status": "no_facts_to_merge"}

        try:
            # Add entities
            for entity in result.entities:
                self.graph_store.add_entity(entity)
                embedding_text = f"{entity.name}: {entity.description}"
                vector = self.embedder.embed_text(embedding_text)
                self.vector_store.add_vector(
                    vector_id=entity.id,
                    vector=vector,
                    metadata={"name": entity.name, "type": entity.type},
                )

            # Add relationships
            for relationship in result.relationships:
                self.graph_store.add_relationship(relationship)

            logger.info("Successfully merged facts into stores")
            return {"status": "merged"}
        except Exception as e:
            logger.exception("Failed to merge facts")
            return {
                "errors": state.get("errors", []) + [f"Merge error: {str(e)}"],
                "status": "merge_failed",
            }

    def entity_resolution(self, state: ConsolidationState) -> Dict[str, Any]:
        """Resolves duplicate entities and merges them.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [entity_resolution]: Resolving duplicate entities")
        similarity_threshold = state.get("similarity_threshold", settings.SIMILARITY_THRESHOLD)

        try:
            entities = self.graph_store.get_all_entities()
            merge_pairs = []

            for i, ent1 in enumerate(entities):
                for j, ent2 in enumerate(entities):
                    if i >= j:
                        continue

                    # Check type compatibility
                    if not self._are_types_compatible(ent1.type, ent2.type):
                        continue

                    # String similarity
                    str_sim = jaro_winkler_similarity(ent1.name, ent2.name)

                    # Vector similarity
                    vec_sim = 0.0
                    v1 = self.vector_store.get_vector(ent1.id)
                    v2 = self.vector_store.get_vector(ent2.id)
                    if v1 is not None and v2 is not None:
                        vec_sim = cosine_similarity(v1, v2)

                    # Merge criteria
                    should_merge = False
                    if str_sim >= 0.95:
                        should_merge = True
                    elif vec_sim >= similarity_threshold and str_sim >= 0.75:
                        should_merge = True

                    if should_merge:
                        merge_pairs.append((ent1.id, ent2.id))
                        logger.info(
                            "Duplicate pair: %s and %s [str_sim=%.2f, vec_sim=%.2f]",
                            ent1.id,
                            ent2.id,
                            str_sim,
                            vec_sim,
                        )

            merged_count = 0
            if merge_pairs:
                # Build temporary graph to find connected components
                merge_graph = nx.Graph()
                merge_graph.add_nodes_from([e.id for e in entities])
                for id1, id2 in merge_pairs:
                    merge_graph.add_edge(id1, id2)

                clusters = [list(c) for c in nx.connected_components(merge_graph) if len(c) > 1]

                for cluster in clusters:
                    # Sort cluster to find primary entity
                    # Criteria: 1. Degree (highest first), 2. Description length (longest first), 3. ID length (shortest first), 4. Age (oldest first), 5. Alphabetical ID
                    def sort_key(ent_id: str) -> Tuple[int, int, int, float, str]:
                        ent = self.graph_store.get_entity(ent_id)
                        if not ent:
                            return (0, 0, 0, 0.0, "")
                        deg = self.graph_store.get_degree(ent_id)
                        desc_len = len(ent.description) if ent.description else 0
                        id_len = len(ent_id)
                        created_ts = ent.created_at.timestamp()
                        return (-deg, -desc_len, id_len, created_ts, ent_id)

                    cluster.sort(key=sort_key)
                    primary_id = cluster[0]
                    duplicate_ids = cluster[1:]

                    logger.info("Merging cluster %s into primary %s", duplicate_ids, primary_id)
                    self._merge_cluster(primary_id, duplicate_ids)
                    merged_count += len(duplicate_ids)

            return {
                "merged_entities_count": merged_count,
                "status": "entity_resolution_complete",
            }
        except Exception as e:
            logger.exception("Failed to resolve entities")
            return {
                "errors": state.get("errors", []) + [f"Entity resolution error: {str(e)}"],
                "status": "entity_resolution_failed",
            }

    def temporal_decay(self, state: ConsolidationState) -> Dict[str, Any]:
        """Applies temporal decay to relationship weights.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [temporal_decay]: Applying temporal decay to relationship weights")
        decay_factor = state.get("decay_factor", settings.RECENCY_DECAY_FACTOR)

        try:
            # Check if the store has an optimized implementation
            if hasattr(self.graph_store, "decay_and_prune"):
                # If the store has decay_and_prune, we can call it.
                # Note that decay_and_prune does both decay and pruning, so we can run it here
                # and skip the pruning step, or we can just run the generic decay.
                logger.info("Using optimized decay_and_prune on graph store")
                self.graph_store.decay_and_prune(decay_factor, state.get("pruning_threshold", settings.PRUNING_THRESHOLD))  # type: ignore[attr-defined]
                return {"status": "decay_complete_optimized"}

            relationships = self.graph_store.get_all_relationships()
            now = datetime.utcnow()
            decayed_count = 0

            for rel in relationships:
                elapsed_time = now - rel.updated_at
                elapsed_days = elapsed_time.total_seconds() / 86400.0

                # Decay weight
                new_weight = rel.weight * (decay_factor**elapsed_days)

                # Update weight in store (remove and re-add to avoid incrementing)
                self.graph_store.remove_relationship(rel.source, rel.target, rel.type)
                rel.weight = new_weight
                rel.updated_at = now
                self.graph_store.add_relationship(rel)
                decayed_count += 1

            logger.info("Decayed %d relationships", decayed_count)
            return {"status": "decay_complete"}
        except Exception as e:
            logger.exception("Failed to apply temporal decay")
            return {
                "errors": state.get("errors", []) + [f"Decay error: {str(e)}"],
                "status": "decay_failed",
            }

    def prune_graph(self, state: ConsolidationState) -> Dict[str, Any]:
        """Prunes weak relationships and isolated entities.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [prune_graph]: Pruning weak relationships and isolated entities")
        pruning_threshold = state.get("pruning_threshold", settings.PRUNING_THRESHOLD)
        forgetting_age_days = state.get("forgetting_age_days", 30.0)

        try:
            # If optimized decay_and_prune was run, we can skip manual pruning
            if state.get("status") == "decay_complete_optimized":
                logger.info("Skipping manual pruning as optimized decay_and_prune was executed")
                return {"pruned_relationships_count": 0, "status": "pruning_complete_optimized"}

            relationships = self.graph_store.get_all_relationships()
            pruned_rels = 0

            for rel in relationships:
                if rel.weight < pruning_threshold:
                    self.graph_store.remove_relationship(rel.source, rel.target, rel.type)
                    logger.info(
                        "Pruned relationship: %s -[%s]-> %s (weight %f < %f)",
                        rel.source,
                        rel.type,
                        rel.target,
                        rel.weight,
                        pruning_threshold,
                    )
                    pruned_rels += 1

            # Prune isolated nodes
            entities = self.graph_store.get_all_entities()
            pruned_nodes = 0
            now = datetime.utcnow()
            for entity in entities:
                if self.graph_store.get_degree(entity.id) == 0:
                    # Check if marked as permanent or important
                    is_permanent = entity.properties.get("permanent", False)
                    importance = entity.properties.get("importance", 0.0)
                    if is_permanent or importance >= 0.8:
                        continue

                    # If it's old enough to be forgotten, let the forgetting node handle it
                    elapsed_time = now - entity.updated_at
                    age_days = elapsed_time.total_seconds() / 86400.0
                    if age_days >= forgetting_age_days:
                        continue

                    self.graph_store.remove_entity(entity.id)
                    self.vector_store.delete_vector(entity.id)
                    logger.info("Pruned isolated entity: %s", entity.id)
                    pruned_nodes += 1

            logger.info("Pruned %d relationships and %d isolated entities", pruned_rels, pruned_nodes)
            return {"pruned_relationships_count": pruned_rels, "status": "pruning_complete"}
        except Exception as e:
            logger.exception("Failed to prune graph")
            return {
                "errors": state.get("errors", []) + [f"Pruning error: {str(e)}"],
                "status": "pruning_failed",
            }

    def forgetting(self, state: ConsolidationState) -> Dict[str, Any]:
        """Implements the forgetting policy for old, inactive, and low-importance entities.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.
        """
        logger.info("Node [forgetting]: Applying forgetting policy")
        forgetting_age_days = state.get("forgetting_age_days", 30.0)

        try:
            entities = self.graph_store.get_all_entities()
            now = datetime.utcnow()
            forgotten_count = 0

            for entity in entities:
                # Calculate age since last update
                elapsed_time = now - entity.updated_at
                age_days = elapsed_time.total_seconds() / 86400.0

                if age_days >= forgetting_age_days:
                    # Check if marked as permanent or important
                    is_permanent = entity.properties.get("permanent", False)
                    importance = entity.properties.get("importance", 0.0)

                    if is_permanent or importance >= 0.8:
                        logger.debug("Entity %s is permanent/important, skipping forgetting", entity.id)
                        continue

                    # Check connectivity (degree)
                    degree = self.graph_store.get_degree(entity.id)
                    if degree >= 3:
                        # Highly connected entity, don't forget completely
                        logger.debug(
                            "Entity %s has high degree (%d), skipping forgetting", entity.id, degree
                        )
                        continue

                    # Forget the entity
                    logger.info(
                        "Forgetting entity: %s (age: %.1f days, degree: %d)",
                        entity.id,
                        age_days,
                        degree,
                    )

                    # Remove all relationships connected to it first
                    neighbors = self.graph_store.get_neighbors(entity.id)
                    for _, rel in neighbors:
                        self.graph_store.remove_relationship(rel.source, rel.target, rel.type)

                    # Remove entity from graph and vector stores
                    self.graph_store.remove_entity(entity.id)
                    self.vector_store.delete_vector(entity.id)
                    forgotten_count += 1

            logger.info("Forgot %d entities", forgotten_count)
            return {"forgotten_entities_count": forgotten_count, "status": "forgetting_complete"}
        except Exception as e:
            logger.exception("Failed to apply forgetting policy")
            return {
                "errors": state.get("errors", []) + [f"Forgetting error: {str(e)}"],
                "status": "forgetting_failed",
            }

    def finalize(self, state: ConsolidationState) -> Dict[str, Any]:
        """Finalizes the consolidation run, marking messages as processed.

        Args:
            state: The current consolidation state.

        Returns:
            State updates.

        """
        logger.info("Node [finalize]: Finalizing consolidation run")
        msg_ids = state.get("unprocessed_message_ids", [])

        try:
            if self.episodic_buffer and msg_ids:
                self.episodic_buffer.mark_as_processed(msg_ids)
                logger.info("Marked %d messages as processed in episodic buffer", len(msg_ids))

            return {"completed_at": datetime.utcnow(), "status": "success"}
        except Exception as e:
            logger.exception("Failed to finalize consolidation")
            return {
                "errors": state.get("errors", []) + [f"Finalization error: {str(e)}"],
                "status": "finalization_failed",
            }

    def _build_workflow(self) -> Any:
        """Builds and compiles the LangGraph workflow.

        Returns:
            The compiled LangGraph workflow.
        """
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(ConsolidationState)

        # Add nodes
        workflow.add_node("fetch_unprocessed", self.fetch_unprocessed)
        workflow.add_node("extract_facts", self.extract_facts)
        workflow.add_node("merge_facts", self.merge_facts)
        workflow.add_node("entity_resolution", self.entity_resolution)
        workflow.add_node("temporal_decay", self.temporal_decay)
        workflow.add_node("prune_graph", self.prune_graph)
        workflow.add_node("forgetting", self.forgetting)
        workflow.add_node("finalize", self.finalize)

        # Set entry point
        workflow.set_entry_point("fetch_unprocessed")

        # Linear edges
        workflow.add_edge("fetch_unprocessed", "extract_facts")
        workflow.add_edge("extract_facts", "merge_facts")
        workflow.add_edge("merge_facts", "entity_resolution")
        workflow.add_edge("entity_resolution", "temporal_decay")
        workflow.add_edge("temporal_decay", "prune_graph")
        workflow.add_edge("prune_graph", "forgetting")
        workflow.add_edge("forgetting", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run_consolidation_workflow(
        self,
        session_id: Optional[str] = None,
        decay_factor: Optional[float] = None,
        pruning_threshold: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
        forgetting_age_days: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs the consolidation workflow.

        Args:
            session_id: Optional session ID to filter messages.
            decay_factor: Optional decay factor override.
            pruning_threshold: Optional pruning threshold override.
            similarity_threshold: Optional similarity threshold override.
            forgetting_age_days: Optional forgetting age threshold override.

        Returns:
            The final state dictionary.
        """
        initial_state: ConsolidationState = {
            "session_id": session_id,
            "unprocessed_message_ids": [],
            "messages": [],
            "extracted_result": None,
            "merged_entities_count": 0,
            "pruned_relationships_count": 0,
            "forgotten_entities_count": 0,
            "decay_factor": (
                decay_factor if decay_factor is not None else settings.RECENCY_DECAY_FACTOR
            ),
            "pruning_threshold": (
                pruning_threshold if pruning_threshold is not None else settings.PRUNING_THRESHOLD
            ),
            "similarity_threshold": (
                similarity_threshold
                if similarity_threshold is not None
                else settings.SIMILARITY_THRESHOLD
            ),
            "forgetting_age_days": (
                forgetting_age_days if forgetting_age_days is not None else 30.0
            ),
            "status": "started",
            "errors": [],
            "started_at": datetime.utcnow(),
            "completed_at": None,
        }

        logger.info("Starting LangGraph consolidation workflow run")
        final_state = self.workflow.invoke(initial_state)
        logger.info(
            "LangGraph consolidation workflow run completed with status: %s",
            final_state.get("status"),
        )
        return final_state
