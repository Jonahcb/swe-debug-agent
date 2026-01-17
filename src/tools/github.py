"""GitHub operations tool using PyGithub."""

import os
from dataclasses import dataclass
from github import Github, Auth


@dataclass
class PRInfo:
    number: int
    title: str
    body: str
    state: str
    url: str
    diff_url: str
    head_sha: str
    base_branch: str
    head_branch: str


@dataclass
class IssueInfo:
    number: int
    title: str
    body: str
    state: str
    url: str
    labels: list[str]


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(self, token: str | None = None):
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GitHub token required")
        self.gh = Github(auth=Auth.Token(token))

    def get_repo(self, owner: str, repo: str):
        """Get a repository object."""
        return self.gh.get_repo(f"{owner}/{repo}")

    def get_pr(self, owner: str, repo: str, pr_number: int) -> PRInfo:
        """Get pull request information."""
        repo_obj = self.get_repo(owner, repo)
        pr = repo_obj.get_pull(pr_number)
        return PRInfo(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            state=pr.state,
            url=pr.html_url,
            diff_url=pr.diff_url,
            head_sha=pr.head.sha,
            base_branch=pr.base.ref,
            head_branch=pr.head.ref,
        )

    def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Get the diff for a pull request."""
        import requests

        pr_info = self.get_pr(owner, repo, pr_number)
        response = requests.get(pr_info.diff_url)
        return response.text

    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[dict]:
        """Get list of files changed in a PR."""
        repo_obj = self.get_repo(owner, repo)
        pr = repo_obj.get_pull(pr_number)
        return [
            {
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "patch": f.patch,
            }
            for f in pr.get_files()
        ]

    def get_issue(self, owner: str, repo: str, issue_number: int) -> IssueInfo:
        """Get issue information."""
        repo_obj = self.get_repo(owner, repo)
        issue = repo_obj.get_issue(issue_number)
        return IssueInfo(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            url=issue.html_url,
            labels=[l.name for l in issue.labels],
        )

    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> str:
        """Get content of a file from the repository."""
        repo_obj = self.get_repo(owner, repo)
        content = repo_obj.get_contents(path, ref=ref)

        # Handle both files and directories
        if isinstance(content, list):
            # This is a directory, return list of files
            return "\n".join([c.path for c in content])
        else:
            # This is a file, return its content
            return content.decoded_content.decode("utf-8")

    def list_repo_files(
        self, owner: str, repo: str, path: str = "", ref: str = "main"
    ) -> list[str]:
        """List files in a repository directory."""
        repo_obj = self.get_repo(owner, repo)
        contents = repo_obj.get_contents(path, ref=ref)
        return [c.path for c in contents]

    def create_comment(self, owner: str, repo: str, issue_number: int, body: str) -> None:
        """Create a comment on an issue or PR."""
        repo_obj = self.get_repo(owner, repo)
        issue = repo_obj.get_issue(issue_number)
        issue.create_comment(body)

    def clone_repo(self, owner: str, repo: str, path: str) -> None:
        """Clone a repository to local path."""
        import subprocess

        url = f"https://github.com/{owner}/{repo}.git"
        subprocess.run(["git", "clone", url, path], check=True)
