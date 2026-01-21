"""
Helper utilities for the LATS agent.

This module provides checkpoint management utilities.
"""

import json
import os
import sqlite3
from datetime import datetime

# Import from main module
from lats_agent import CHECKPOINT_DB, TreeState


def list_checkpoints() -> list[dict]:
    """List all saved checkpoints."""
    if not os.path.exists(CHECKPOINT_DB):
        return []

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT thread_id, thread_ts, parent_ts
            FROM checkpoints
            ORDER BY thread_ts DESC
        """)

        checkpoints = []
        for row in cursor.fetchall():
            checkpoints.append(
                {
                    "thread_id": row[0],
                    "timestamp": row[1],
                    "parent_ts": row[2],
                }
            )

        return checkpoints
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def delete_checkpoint(thread_id: str) -> bool:
    """Delete a checkpoint by thread ID."""
    if not os.path.exists(CHECKPOINT_DB):
        return False

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
        return cursor.rowcount > 0
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def inspect_checkpoint(thread_id: str) -> dict | None:
    """Inspect a specific checkpoint by thread ID."""
    if not os.path.exists(CHECKPOINT_DB):
        return None

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT thread_id, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
            FROM checkpoints
            WHERE thread_id = ?
            ORDER BY rowid DESC
            LIMIT 1
        """,
            (thread_id,),
        )

        row = cursor.fetchone()
        if row:
            return {
                "thread_id": row[0],
                "checkpoint_id": row[1],
                "parent_checkpoint_id": row[2],
                "checkpoint_data": row[3],
                "metadata": row[4],
            }
        return None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def export_state(state: TreeState, filepath: str):
    """Export the current state to a JSON file."""
    # Convert state to JSON-serializable format
    export_data = {
        "nodes": state["nodes"],
        "root_id": state["root_id"],
        "selected_node_id": state.get("selected_node_id"),
        "candidate_ids": state.get("candidate_ids", []),
        "iteration": state["iteration"],
        "status": state["status"],
        "best_node_id": state.get("best_node_id"),
        "input": state.get("input", ""),
        "exported_at": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)
