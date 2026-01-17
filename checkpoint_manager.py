#!/usr/bin/env python3
"""
Checkpoint Manager for LATS Tree Search

This script provides utilities to manage LATS checkpoints stored in SQLite.
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lats_helpers import list_checkpoints, delete_checkpoint, inspect_checkpoint, export_state
from lats_agent import CHECKPOINT_DB
import json
import sqlite3
from datetime import datetime


def show_usage():
    """Show usage information."""
    print("LATS Checkpoint Manager")
    print("=" * 50)
    print()
    print("Usage:")
    print("  python checkpoint_manager.py list          - List all checkpoints")
    print("  python checkpoint_manager.py inspect <id>  - Inspect a specific checkpoint")
    print("  python checkpoint_manager.py export <id> <file> - Export checkpoint to JSON")
    print("  python checkpoint_manager.py export-latest <file> - Export latest checkpoint to JSON")
    print("  python checkpoint_manager.py delete <id>  - Delete a checkpoint")
    print("  python checkpoint_manager.py clean         - Delete all checkpoints")
    print("  python checkpoint_manager.py stats         - Show database statistics")
    print()


def list_all_checkpoints():
    """List all available checkpoints."""
    checkpoints = list_checkpoints()

    if not checkpoints:
        print("No checkpoints found.")
        return

    print(f"Found {len(checkpoints)} checkpoint(s):")
    print()
    print(f"{'Thread ID':<40} {'Timestamp':<20} {'Parent TS':<15}")
    print("-" * 75)

    for cp in checkpoints:
        ts = cp['timestamp']
        if isinstance(ts, str) and len(ts) > 19:
            ts = ts[:19]  # Truncate to YYYY-MM-DD HH:MM:SS
        print(f"{cp['thread_id']:<40} {str(ts):<20} {str(cp.get('parent_ts', 'N/A')):<15}")


def inspect_specific_checkpoint(thread_id: str):
    """Inspect a specific checkpoint."""
    checkpoint = inspect_checkpoint(thread_id)

    if not checkpoint:
        print(f"No checkpoint found for thread ID: {thread_id}")
        return

    print(f"Checkpoint Details for Thread: {thread_id}")
    print("=" * 50)
    print(f"Timestamp: {checkpoint['timestamp']}")
    print(f"Parent TS: {checkpoint.get('parent_ts', 'N/A')}")

    # Try to parse checkpoint data
    try:
        if checkpoint['checkpoint_data']:
            cp_data = json.loads(checkpoint['checkpoint_data'])
            print(f"Checkpoint Keys: {list(cp_data.keys())}")

            # Show some basic info if it's a TreeState
            if 'iteration' in cp_data:
                print(f"  Iteration: {cp_data['iteration']}")
            if 'status' in cp_data:
                print(f"  Status: {cp_data['status']}")
            if 'nodes' in cp_data:
                print(f"  Nodes: {len(cp_data['nodes'])}")
    except json.JSONDecodeError:
        print("  Checkpoint data: <binary data>")

    if checkpoint.get('metadata'):
        try:
            meta = json.loads(checkpoint['metadata'])
            print(f"Metadata: {meta}")
        except json.JSONDecodeError:
            print("Metadata: <binary data>")


def get_latest_thread_id() -> str | None:
    """Get the thread ID of the most recent checkpoint."""
    if not Path(CHECKPOINT_DB).exists():
        return None

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        # Get the most recent checkpoint by checking the writes table for latest activity
        # or the checkpoints table for the most recent checkpoint_id
        cursor.execute("""
            SELECT thread_id
            FROM checkpoints
            ORDER BY rowid DESC
            LIMIT 1
        """)

        row = cursor.fetchone()
        return row[0] if row else None

    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


def export_checkpoint_to_file(thread_id: str, filename: str):
    """Export a checkpoint to a JSON file."""
    checkpoint = inspect_checkpoint(thread_id)

    if not checkpoint:
        print(f"No checkpoint found for thread ID: {thread_id}")
        return

    try:
        # Try to reconstruct a TreeState-like object
        export_data = {
            "thread_id": checkpoint["thread_id"],
            "checkpoint_id": checkpoint["checkpoint_id"],
            "exported_at": datetime.now().isoformat(),
        }

        # Add timestamp if available
        if "timestamp" in checkpoint:
            export_data["timestamp"] = checkpoint["timestamp"]

        # Parse checkpoint data if possible (it's stored as binary blob)
        if checkpoint['checkpoint_data']:
            try:
                # Try to decode as UTF-8 string first
                checkpoint_str = checkpoint['checkpoint_data']
                if isinstance(checkpoint_str, bytes):
                    checkpoint_str = checkpoint_str.decode('utf-8')
                cp_data = json.loads(checkpoint_str)
                export_data["checkpoint"] = cp_data
            except (json.JSONDecodeError, UnicodeDecodeError):
                export_data["checkpoint_raw"] = str(checkpoint['checkpoint_data'])

        if checkpoint.get('metadata'):
            try:
                meta_str = checkpoint['metadata']
                if isinstance(meta_str, bytes):
                    meta_str = meta_str.decode('utf-8')
                meta = json.loads(meta_str)
                export_data["metadata"] = meta
            except (json.JSONDecodeError, UnicodeDecodeError):
                export_data["metadata_raw"] = str(checkpoint['metadata'])

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Checkpoint exported to: {filename}")

    except Exception as e:
        print(f"Error exporting checkpoint: {e}")


def export_latest_checkpoint(filename: str):
    """Export the latest checkpoint to a JSON file."""
    thread_id = get_latest_thread_id()

    if not thread_id:
        print("No checkpoints found.")
        return

    print(f"Exporting latest checkpoint from thread: {thread_id}")
    export_checkpoint_to_file(thread_id, filename)


def delete_specific_checkpoint(thread_id: str):
    """Delete a specific checkpoint."""
    if delete_checkpoint(thread_id):
        print(f"Checkpoint {thread_id} deleted successfully.")
    else:
        print(f"Failed to delete checkpoint {thread_id}.")


def clean_all_checkpoints():
    """Delete all checkpoints."""
    if not Path(CHECKPOINT_DB).exists():
        print("No checkpoint database found.")
        return

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        count = cursor.fetchone()[0]

        if count == 0:
            print("No checkpoints to delete.")
            return

        confirm = input(f"Delete {count} checkpoint(s)? (y/N): ")
        if confirm.lower() == 'y':
            cursor.execute("DELETE FROM checkpoints")
            conn.commit()
            print(f"Deleted {count} checkpoint(s).")
        else:
            print("Operation cancelled.")

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def show_database_stats():
    """Show database statistics."""
    if not Path(CHECKPOINT_DB).exists():
        print("No checkpoint database found.")
        return

    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()

    try:
        # Count checkpoints
        cursor.execute("SELECT COUNT(*) FROM checkpoints")
        checkpoint_count = cursor.fetchone()[0]

        # Get database size
        db_size = Path(CHECKPOINT_DB).stat().st_size

        # Get oldest and newest timestamps
        cursor.execute("SELECT MIN(thread_ts), MAX(thread_ts) FROM checkpoints")
        ts_result = cursor.fetchone()

        print("Database Statistics")
        print("=" * 30)
        print(f"Database file: {CHECKPOINT_DB}")
        print(f"File size: {db_size:,} bytes ({db_size/1024/1024:.2f} MB)")
        print(f"Total checkpoints: {checkpoint_count}")

        if ts_result and ts_result[0]:
            print(f"Oldest checkpoint: {ts_result[0]}")
            print(f"Newest checkpoint: {ts_result[1]}")

        # Show table info
        cursor.execute("PRAGMA table_info(checkpoints)")
        columns = cursor.fetchall()
        print(f"\nTable columns: {len(columns)}")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")

    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def main():
    if len(sys.argv) < 2:
        show_usage()
        return

    command = sys.argv[1].lower()

    if command == "list":
        list_all_checkpoints()

    elif command == "inspect" and len(sys.argv) >= 3:
        inspect_specific_checkpoint(sys.argv[2])

    elif command == "export" and len(sys.argv) >= 4:
        export_checkpoint_to_file(sys.argv[2], sys.argv[3])

    elif command == "export-latest" and len(sys.argv) >= 3:
        export_latest_checkpoint(sys.argv[2])

    elif command == "delete" and len(sys.argv) >= 3:
        delete_specific_checkpoint(sys.argv[2])

    elif command == "clean":
        clean_all_checkpoints()

    elif command == "stats":
        show_database_stats()

    else:
        show_usage()


if __name__ == "__main__":
    main()