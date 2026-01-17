"""File system operations tool."""

import json
import os
from pathlib import Path


def read_file(path: str) -> str:
    """Read contents of a file."""
    # Check if there's a modified version in the current node's code snapshot
    code_snapshot_json = os.getenv("CURRENT_NODE_CODE_SNAPSHOT")
    if code_snapshot_json:
        try:
            code_snapshot = json.loads(code_snapshot_json)
            # Check if the requested file is in the snapshot
            if path in code_snapshot:
                print(f"📄 [TOOL] read_file: Returning modified version from node snapshot: {path}")
                return code_snapshot[path]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Failed to parse code snapshot: {e}")

    # Fall back to reading from the filesystem
    return Path(path).read_text()


def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def list_directory(path: str, recursive: bool = False) -> list[str]:
    """List files in a directory."""
    p = Path(path)
    if recursive:
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    return [f.name for f in p.iterdir()]


def file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


def delete_file(path: str) -> None:
    """Delete a file."""
    Path(path).unlink()


def create_directory(path: str) -> None:
    """Create a directory."""
    Path(path).mkdir(parents=True, exist_ok=True)


def move_file(src: str, dst: str) -> None:
    """Move a file from src to dst."""
    Path(src).rename(dst)


def copy_file(src: str, dst: str) -> None:
    """Copy a file from src to dst."""
    import shutil
    shutil.copy2(src, dst)


def get_file_info(path: str) -> dict:
    """Get file metadata."""
    p = Path(path)
    stat = p.stat()
    return {
        "name": p.name,
        "path": str(p.absolute()),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
    }


def search_files(directory: str, pattern: str) -> list[str]:
    """Search for files matching a glob pattern."""
    return [str(f) for f in Path(directory).rglob(pattern)]
