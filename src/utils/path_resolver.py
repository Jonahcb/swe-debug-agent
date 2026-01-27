"""File path resolution utilities for finding closest matching file paths in the repository."""

import difflib
from pathlib import Path
from typing import List, Optional, Tuple


class PathResolver:
    """Utility class for resolving potentially incorrect file paths to correct ones."""

    def __init__(self, repo_root: Optional[str] = None):
        """Initialize with repository root path.

        Args:
            repo_root: Root directory of the repository. Defaults to current working directory.
        """
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()

    def _get_all_files(self) -> List[Path]:
        """Get all files in the repository, excluding common irrelevant directories."""
        if not self.repo_root.is_dir():
            return []

        # Directories to exclude from search
        exclude_dirs = {
            '.git', '__pycache__', 'node_modules', '.pytest_cache',
            'venv', 'env', '.env', 'virtualenv', 'site-packages',
            '.next', '.nuxt', 'dist', 'build', 'target',
            '.vscode', '.idea', 'coverage', '.tox'
        }

        # Also exclude directories that start with these patterns
        exclude_patterns = ['venv', 'env', 'virtualenv', 'site-packages']

        all_files = []
        for file_path in self.repo_root.rglob("*"):
            if not file_path.is_file():
                continue

            # Check if file is in excluded directory
            parts = file_path.relative_to(self.repo_root).parts

            # Skip if any part is in exclude_dirs
            if any(part in exclude_dirs for part in parts):
                continue

            # Skip if any part starts with exclude patterns
            if any(part.startswith(pattern) for part in parts for pattern in exclude_patterns):
                continue

            all_files.append(file_path)

        return all_files

    def _extract_filename(self, path: str) -> str:
        """Extract filename from a path string."""
        return Path(path).name

    def _find_files_by_name(self, filename: str) -> List[Path]:
        """Find all files with the given filename."""
        all_files = self._get_all_files()
        return [f for f in all_files if f.name == filename and f.is_file()]

    def _find_similar_files(self, filename: str, cutoff: float = 0.6) -> List[Tuple[Path, float]]:
        """Find files with similar names using fuzzy matching.

        Args:
            filename: The target filename
            cutoff: Minimum similarity ratio (0.0 to 1.0)

        Returns:
            List of tuples (file_path, similarity_score) sorted by similarity descending
        """
        all_files = self._get_all_files()
        matches = []

        for file_path in all_files:
            if not file_path.is_file():
                continue

            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, filename, file_path.name).ratio()
            if ratio >= cutoff:
                matches.append((file_path, ratio))

        # Sort by similarity score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def resolve_path(self, incorrect_path: str) -> dict:
        """Resolve an incorrect file path to the most likely correct path(s).

        Args:
            incorrect_path: The potentially incorrect file path

        Returns:
            Dict containing:
            - 'resolved_path': Single resolved path if unique filename match found
            - 'candidates': List of candidate paths (for multiple matches or similar names)
            - 'strategy': The resolution strategy used ('exact_filename', 'multiple_matches', 'similar_names')
            - 'confidence': Confidence level ('high', 'medium', 'low')
        """
        filename = self._extract_filename(incorrect_path)

        # Strategy 1: Check for unique filename match
        matching_files = self._find_files_by_name(filename)
        if len(matching_files) == 1:
            resolved_path = matching_files[0].relative_to(self.repo_root)
            return {
                'resolved_path': str(resolved_path),
                'candidates': [str(resolved_path)],
                'strategy': 'exact_filename',
                'confidence': 'high'
            }
        elif len(matching_files) > 1:
            # Multiple files with same name - return all candidates
            candidates = [str(f.relative_to(self.repo_root)) for f in matching_files]
            return {
                'resolved_path': None,
                'candidates': candidates,
                'strategy': 'multiple_matches',
                'confidence': 'medium'
            }

        # Strategy 2: No exact matches - find similar filenames
        similar_files = self._find_similar_files(filename)
        if similar_files:
            candidates = [str(f[0].relative_to(self.repo_root)) for f in similar_files[:5]]  # Top 5 matches
            return {
                'resolved_path': None,
                'candidates': candidates,
                'strategy': 'similar_names',
                'confidence': 'low'
            }

        # Strategy 3: No matches at all
        return {
            'resolved_path': None,
            'candidates': [],
            'strategy': 'no_matches',
            'confidence': 'none'
        }

    def resolve_multiple_paths(self, incorrect_paths: List[str]) -> List[dict]:
        """Resolve multiple incorrect file paths.

        Args:
            incorrect_paths: List of potentially incorrect file paths

        Returns:
            List of resolution results, one for each input path
        """
        return [self.resolve_path(path) for path in incorrect_paths]

    def get_file_info(self, resolved_path: str) -> Optional[dict]:
        """Get information about a resolved file path.

        Args:
            resolved_path: The resolved file path (relative to repo root)

        Returns:
            Dict with file information or None if file doesn't exist
        """
        full_path = self.repo_root / resolved_path
        if not full_path.exists() or not full_path.is_file():
            return None

        stat = full_path.stat()
        return {
            'name': full_path.name,
            'path': str(full_path),
            'relative_path': resolved_path,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'exists': True
        }


def resolve_file_path(incorrect_path: str, repo_root: Optional[str] = None) -> dict:
    """Convenience function to resolve a single file path.

    Args:
        incorrect_path: The potentially incorrect file path
        repo_root: Root directory of the repository

    Returns:
        Resolution result dict
    """
    resolver = PathResolver(repo_root)
    return resolver.resolve_path(incorrect_path)


def resolve_file_paths(incorrect_paths: List[str], repo_root: Optional[str] = None) -> List[dict]:
    """Convenience function to resolve multiple file paths.

    Args:
        incorrect_paths: List of potentially incorrect file paths
        repo_root: Root directory of the repository

    Returns:
        List of resolution result dicts
    """
    resolver = PathResolver(repo_root)
    return resolver.resolve_multiple_paths(incorrect_paths)