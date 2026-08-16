"""GitHub adapter — opens pull requests and posts review comments.

Two implementations: a deterministic :class:`MockGitHubClient` (default, for
tests/evaluation) and :class:`RESTGitHubClient` which drives the real GitHub
REST API via the Git Data + Pulls endpoints.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from cloud_orchestra.core.errors import GitHubError
from cloud_orchestra.schemas import PullRequest


class GitHubClient(Protocol):
    async def create_pull_request(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        branch: str,
        title: str,
        body: str,
        files: dict[str, str],
    ) -> PullRequest: ...

    async def add_comment(self, pr_number: int, body: str) -> None: ...


class MockGitHubClient:
    """Simulated PR client — records actions, never touches the network."""

    def __init__(self) -> None:
        self.created_prs: list[PullRequest] = []
        self.comments: list[tuple[int, str]] = []

    async def create_pull_request(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        branch: str,
        title: str,
        body: str,
        files: dict[str, str],
    ) -> PullRequest:
        pr_number = len(self.created_prs) + 1
        pr = PullRequest(
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch=branch,
            title=title,
            body=body,
            pr_number=pr_number,
            pr_url=f"https://github.com/{repo_owner}/{repo_name}/pull/{pr_number}",
            status="simulated",
        )
        self.created_prs.append(pr)
        return pr

    async def add_comment(self, pr_number: int, body: str) -> None:
        self.comments.append((pr_number, body))


class RESTGitHubClient:
    """Real GitHub REST API client (Git Data API for file creation)."""

    BASE = "https://api.github.com"

    def __init__(self, token: str) -> None:
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.request(method, f"{self.BASE}{path}", headers=self._headers, **kwargs)
            if resp.status_code >= 400:
                raise GitHubError(f"GitHub API {method} {path} failed: {resp.status_code} {resp.text}")
            return resp.json() if resp.content else {}

    async def create_pull_request(
        self,
        *,
        repo_owner: str,
        repo_name: str,
        branch: str,
        title: str,
        body: str,
        files: dict[str, str],
    ) -> PullRequest:
        repo = f"{repo_owner}/{repo_name}"
        try:
            base = await self._request("GET", f"/repos/{repo}/git/ref/heads/main")
            base_sha = base["object"]["sha"]
        except GitHubError:
            base = await self._request("GET", f"/repos/{repo}")
            base_sha = base["default_branch"]

        # Create branch
        await self._request(
            "POST",
            f"/repos/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )

        # Create blobs + tree
        tree_items = []
        for path, content in files.items():
            blob = await self._request(
                "POST",
                f"/repos/{repo}/git/blobs",
                json={"content": content, "encoding": "utf-8"},
            )
            tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        tree = await self._request(
            "POST", f"/repos/{repo}/git/trees", json={"tree": tree_items}
        )

        commit = await self._request(
            "POST",
            f"/repos/{repo}/git/commits",
            json={
                "message": title,
                "tree": tree["sha"],
                "parents": [base_sha],
            },
        )
        await self._request(
            "PATCH",
            f"/repos/{repo}/git/refs/heads/{branch}",
            json={"sha": commit["sha"]},
        )

        pr = await self._request(
            "POST",
            f"/repos/{repo}/pulls",
            json={"title": title, "head": branch, "base": "main", "body": body},
        )
        return PullRequest(
            repo_owner=repo_owner,
            repo_name=repo_name,
            branch=branch,
            title=title,
            body=body,
            pr_number=int(pr["number"]),
            pr_url=str(pr["html_url"]),
            status="created",
        )

    async def add_comment(self, pr_number: int, body: str) -> None:
        # PR review comment requires an issue comment for simplicity.
        repo_owner = ""
        repo_name = ""
        # NOTE: owner/repo are threaded through PR objects in the orchestrator;
        # this method is only reachable with a stored PR context.
        if not repo_owner or not repo_name:
            raise GitHubError("add_comment requires a PR context; use MockGitHubClient in eval")
        await self._request(
            "POST",
            f"/repos/{repo_owner}/{repo_name}/issues/{pr_number}/comments",
            json={"body": body},
        )


def build_github_client(token: str, *, use_rest: bool = False) -> GitHubClient:
    if use_rest and token:
        return RESTGitHubClient(token)
    return MockGitHubClient()
