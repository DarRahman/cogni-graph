# cognigraph/episodic_buffer.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Episodic memory buffer for storing raw chat interactions with metadata."""

from datetime import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from cognigraph.models import ChatMessage, StoredMessage

logger = logging.getLogger("cognigraph.episodic_buffer")


class EpisodicBuffer:
    """In-memory episodic buffer with disk persistence for raw chat messages."""

    def __init__(self) -> None:
        """Initializes an empty episodic buffer."""
        self.messages: Dict[str, StoredMessage] = {}
        logger.info("EpisodicBuffer initialized")

    def add_message(self, message: ChatMessage) -> str:
        """Adds a single chat message to the buffer.

        Args:
            message: The ChatMessage to store.

        Returns:
            The unique ID generated for the stored message.
        """
        msg_id = str(uuid.uuid4())
        stored_msg = StoredMessage(
            id=msg_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata,
            processed=False,
            processed_at=None
        )
        self.messages[msg_id] = stored_msg
        logger.debug("Added message %s to episodic buffer", msg_id)
        return msg_id

    def add_messages(self, messages: List[ChatMessage]) -> List[str]:
        """Adds multiple chat messages to the buffer.

        Args:
            messages: A list of ChatMessage objects.

        Returns:
            A list of unique IDs generated for the stored messages.
        """
        ids = []
        for msg in messages:
            ids.append(self.add_message(msg))
        logger.info("Added %d messages to episodic buffer", len(messages))
        return ids

    def get_message(self, message_id: str) -> Optional[StoredMessage]:
        """Retrieves a single message by its ID.

        Args:
            message_id: The unique ID of the message.

        Returns:
            The StoredMessage if found, else None.
        """
        return self.messages.get(message_id)

    def get_messages(
        self,
        session_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        unprocessed_only: bool = False
    ) -> List[StoredMessage]:
        """Retrieves messages from the buffer matching the specified filters.

        Args:
            session_id: Optional session ID to filter by (stored in metadata).
            start_time: Optional start timestamp (inclusive).
            end_time: Optional end timestamp (inclusive).
            limit: Optional maximum number of messages to return.
            unprocessed_only: If True, only returns messages that haven't been consolidated.

        Returns:
            A list of StoredMessage objects sorted by timestamp ascending.
        """
        filtered = []
        for msg in self.messages.values():
            # Filter by processed status
            if unprocessed_only and msg.processed:
                continue

            # Filter by session_id in metadata
            if session_id is not None:
                msg_session_id = msg.metadata.get("session_id")
                if msg_session_id != session_id:
                    continue

            # Filter by start_time
            if start_time is not None and msg.timestamp < start_time:
                continue

            # Filter by end_time
            if end_time is not None and msg.timestamp > end_time:
                continue

            filtered.append(msg)

        # Sort by timestamp ascending
        filtered.sort(key=lambda x: x.timestamp)

        if limit is not None:
            filtered = filtered[:limit]

        return filtered

    def mark_as_processed(self, message_ids: List[str]) -> None:
        """Marks the specified messages as consolidated/processed.

        Args:
            message_ids: List of message IDs to mark as processed.
        """
        now = datetime.utcnow()
        count = 0
        for msg_id in message_ids:
            if msg_id in self.messages:
                self.messages[msg_id].processed = True
                self.messages[msg_id].processed_at = now
                count += 1
        logger.info("Marked %d messages as processed", count)

    def get_unprocessed_sessions(self) -> List[str]:
        """Retrieves a list of session IDs that have unprocessed messages.

        Returns:
            A list of unique session IDs.
        """
        sessions = set()
        for msg in self.messages.values():
            if not msg.processed:
                session_id = msg.metadata.get("session_id")
                if session_id:
                    sessions.add(str(session_id))
        return list(sessions)

    def clear_processed(self, before_timestamp: Optional[datetime] = None) -> int:
        """Deletes processed messages from the buffer to free space.

        Args:
            before_timestamp: Optional timestamp. If provided, only deletes messages
                              processed before this timestamp.

        Returns:
            The number of messages deleted.
        """
        to_delete = []
        for msg_id, msg in self.messages.items():
            if msg.processed:
                if before_timestamp is None:
                    to_delete.append(msg_id)
                elif msg.processed_at and msg.processed_at < before_timestamp:
                    to_delete.append(msg_id)

        for msg_id in to_delete:
            del self.messages[msg_id]

        logger.info("Cleared %d processed messages from buffer", len(to_delete))
        return len(to_delete)

    def save_to_disk(self, path: str) -> None:
        """Serializes the episodic buffer to a JSON file.

        Args:
            path: The file path to save the buffer.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        serialized = {
            msg_id: msg.model_dump(mode="json") for msg_id, msg in self.messages.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        logger.info("Episodic buffer saved to %s with %d messages", path, len(self.messages))

    def load_from_disk(self, path: str) -> None:
        """Deserializes the episodic buffer from a JSON file.

        Args:
            path: The file path to load the buffer from.
        """
        if not os.path.exists(path):
            logger.warning("Episodic buffer file %s does not exist. Starting empty.", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.messages = {}
        for msg_id, msg_data in data.items():
            self.messages[msg_id] = StoredMessage(**msg_data)

        logger.info("Episodic buffer loaded from %s with %d messages", path, len(self.messages))
