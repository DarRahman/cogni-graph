# tests/test_episodic_buffer.py - Stateful long-term memory engine for LLM agents using hybrid vector-graph consolidation and hierarchical entity-relation extraction.
# Contributed by Claude Code

"""Unit tests for the episodic memory buffer."""

import os
import tempfile
from datetime import datetime, timedelta
from cognigraph.episodic_buffer import EpisodicBuffer
from cognigraph.models import ChatMessage


def test_add_and_get_message() -> None:
    """Tests adding and retrieving a single message from the episodic buffer."""
    buffer = EpisodicBuffer()
    msg = ChatMessage(
        role="user",
        content="Hello, this is a test message.",
        timestamp=datetime.utcnow(),
        metadata={"session_id": "session_123"}
    )

    msg_id = buffer.add_message(msg)
    assert msg_id is not None

    stored = buffer.get_message(msg_id)
    assert stored is not None
    assert stored.id == msg_id
    assert stored.role == "user"
    assert stored.content == "Hello, this is a test message."
    assert stored.metadata["session_id"] == "session_123"
    assert not stored.processed


def test_get_messages_filtering() -> None:
    """Tests filtering messages by session ID, processed status, and time range."""
    buffer = EpisodicBuffer()
    now = datetime.utcnow()

    # Add messages with different sessions and timestamps
    msg1 = ChatMessage(
        role="user",
        content="Message 1",
        timestamp=now - timedelta(minutes=10),
        metadata={"session_id": "session_A"}
    )
    msg2 = ChatMessage(
        role="assistant",
        content="Message 2",
        timestamp=now - timedelta(minutes=5),
        metadata={"session_id": "session_A"}
    )
    msg3 = ChatMessage(
        role="user",
        content="Message 3",
        timestamp=now,
        metadata={"session_id": "session_B"}
    )

    ids = buffer.add_messages([msg1, msg2, msg3])

    # Filter by session_id
    session_a_msgs = buffer.get_messages(session_id="session_A")
    assert len(session_a_msgs) == 2
    assert session_a_msgs[0].content == "Message 1"
    assert session_a_msgs[1].content == "Message 2"

    # Filter by time range
    time_filtered = buffer.get_messages(start_time=now - timedelta(minutes=7), end_time=now - timedelta(minutes=2))
    assert len(time_filtered) == 1
    assert time_filtered[0].content == "Message 2"

    # Mark as processed and filter
    buffer.mark_as_processed([ids[0]])
    unprocessed = buffer.get_messages(unprocessed_only=True)
    assert len(unprocessed) == 2
    assert {msg.content for msg in unprocessed} == {"Message 2", "Message 3"}

    # Get unprocessed sessions
    unprocessed_sessions = buffer.get_unprocessed_sessions()
    assert set(unprocessed_sessions) == {"session_A", "session_B"}

    # Mark all as processed
    buffer.mark_as_processed(ids)
    assert len(buffer.get_unprocessed_sessions()) == 0


def test_clear_processed() -> None:
    """Tests clearing processed messages from the buffer."""
    buffer = EpisodicBuffer()
    msg1 = ChatMessage(role="user", content="Msg 1")
    msg2 = ChatMessage(role="user", content="Msg 2")

    ids = buffer.add_messages([msg1, msg2])
    buffer.mark_as_processed([ids[0]])

    # Clear processed
    deleted = buffer.clear_processed()
    assert deleted == 1
    assert len(buffer.messages) == 1
    assert ids[1] in buffer.messages


def test_save_and_load_disk() -> None:
    """Tests saving and loading the episodic buffer to/from disk."""
    buffer = EpisodicBuffer()
    msg = ChatMessage(role="user", content="Persist me", metadata={"session_id": "s1"})
    buffer.add_message(msg)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        buffer.save_to_disk(tmp_path)

        new_buffer = EpisodicBuffer()
        new_buffer.load_from_disk(tmp_path)

        assert len(new_buffer.messages) == 1
        stored = list(new_buffer.messages.values())[0]
        assert stored.content == "Persist me"
        assert stored.metadata["session_id"] == "s1"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
