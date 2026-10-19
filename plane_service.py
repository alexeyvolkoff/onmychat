"""
Plane.so Integration Service for OnMyChat and MCP.

Supports:
- Listing workspaces and projects
- Listing issues with state & priority filtering
- Retrieving full issue details (descriptions, assignees, dates, sub-tasks)
- Creating new issues / tasks
- Transitioning issue states (Todo, In Progress, In Review, Done, etc.)
- Adding comments / daily work reports to issues

Compatible with both Plane Cloud and Self-Hosted Plane instances.
"""

import os
import re
import html
import logging
from typing import Optional, Dict, Any, List, Tuple
import httpx

logger = logging.getLogger(__name__)

DEFAULT_PLANE_BASE_URL = "https://api.plane.so"


def resolve_plane_token(ctx: Any = None, token: Optional[str] = None) -> Optional[str]:
    """
    Resolve Plane Personal Access Token / API Key from:
    1. Explicit parameter
    2. ctx.mcp_keys['plane'] (from user DiD graph / profile)
    3. Environment variable PLANE_API_KEY or PLANE_TOKEN
    """
    if token and isinstance(token, str) and token.strip():
        return token.strip()
    if ctx and hasattr(ctx, "mcp_keys") and isinstance(ctx.mcp_keys, dict):
        key = ctx.mcp_keys.get("plane")
        if key and isinstance(key, str) and key.strip():
            return key.strip()
    env_token = os.environ.get("PLANE_API_KEY") or os.environ.get("PLANE_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    return None


def resolve_plane_base_url(ctx: Any = None, base_url: Optional[str] = None) -> str:
    """
    Resolve Plane base URL (Cloud or Self-Hosted):
    1. Explicit parameter
    2. ctx.mcp_keys['plane_url']
    3. Environment variable PLANE_BASE_URL
    4. Default https://api.plane.so
    """
    url = base_url
    if not url and ctx and hasattr(ctx, "mcp_keys") and isinstance(ctx.mcp_keys, dict):
        url = ctx.mcp_keys.get("plane_url")
    if not url:
        url = os.environ.get("PLANE_BASE_URL")
    if not url or not str(url).strip():
        url = DEFAULT_PLANE_BASE_URL

    url = str(url).strip().rstrip("/")
    if url.endswith("/api/v1") or url.endswith("/api/v2"):
        url = url.rsplit("/api/", 1)[0]
    return url


def _make_headers(token: str) -> Dict[str, str]:
    return {
        "X-API-Key": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _strip_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', raw_html, flags=re.I)
    text = re.sub(r'</p>', '\n', text, flags=re.I)
    text = re.sub(r'</li>', '\n', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    return text.strip()


async def _plane_get(client: httpx.AsyncClient, url: str, headers: Dict[str, str], fallback_url: Optional[str] = None) -> Tuple[int, Any]:
    try:
        resp = await client.get(url, headers=headers, timeout=12.0)
        if resp.status_code == 404 and fallback_url and fallback_url != url:
            resp = await client.get(fallback_url, headers=headers, timeout=12.0)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return resp.status_code, data
    except Exception as e:
        logger.warning(f"[Plane API] GET {url} failed: {e}")
        return 500, str(e)


async def _plane_post(client: httpx.AsyncClient, url: str, headers: Dict[str, str], json_data: dict, fallback_url: Optional[str] = None) -> Tuple[int, Any]:
    try:
        resp = await client.post(url, headers=headers, json=json_data, timeout=12.0)
        if resp.status_code == 404 and fallback_url and fallback_url != url:
            resp = await client.post(fallback_url, headers=headers, json=json_data, timeout=12.0)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return resp.status_code, data
    except Exception as e:
        logger.warning(f"[Plane API] POST {url} failed: {e}")
        return 500, str(e)


async def _plane_patch(client: httpx.AsyncClient, url: str, headers: Dict[str, str], json_data: dict, fallback_url: Optional[str] = None) -> Tuple[int, Any]:
    try:
        resp = await client.patch(url, headers=headers, json=json_data, timeout=12.0)
        if resp.status_code == 404 and fallback_url and fallback_url != url:
            resp = await client.patch(fallback_url, headers=headers, json=json_data, timeout=12.0)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return resp.status_code, data
    except Exception as e:
        logger.warning(f"[Plane API] PATCH {url} failed: {e}")
        return 500, str(e)


async def list_plane_workspaces(ctx: Any = None, token: Optional[str] = None) -> str:
    """
    List all workspaces available to the user in Plane.
    Returns names and slugs (e.g. slug: 'team-workspace').
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    base = resolve_plane_base_url(ctx)
    url = f"{base}/api/v1/workspaces/"
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        status, data = await _plane_get(client, url, headers)
        if status != 200:
            return f"Error retrieving workspaces from Plane (HTTP {status}): {data}"

        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not results:
            return "No workspaces found in this Plane account."

        lines = ["**Your Plane Workspaces:**\n"]
        for ws in results:
            name = ws.get("name", "Untitled")
            slug = ws.get("slug", "")
            ws_id = ws.get("id", "")
            lines.append(f"- **{name}** (slug: `{slug}` | id: `{ws_id}`)")

        lines.append("\n*Use the workspace slug for further project requests.*")
        return "\n".join(lines)


async def list_plane_projects(ctx: Any = None, workspace: str = "", token: Optional[str] = None) -> str:
    """
    List all projects in a specific Plane workspace.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace:
        return "Error: workspace slug is required. Call `list_plane_workspaces` first if unknown."

    base = resolve_plane_base_url(ctx)
    url = f"{base}/api/v1/workspaces/{workspace}/projects/"
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        status, data = await _plane_get(client, url, headers)
        if status != 200:
            return f"Error retrieving projects from workspace '{workspace}' (HTTP {status}): {data}"

        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not results:
            return f"No projects found in workspace '{workspace}'."

        lines = [f"**Projects in Plane workspace '{workspace}':**\n"]
        for p in results:
            name = p.get("name", "Untitled")
            identifier = p.get("identifier", "")
            pid = p.get("id", "")
            desc = p.get("description", "")
            desc_str = f" — {desc[:60]}..." if desc else ""
            lines.append(f"- **{name}** [{identifier}] (Project ID: `{pid}`){desc_str}")

        lines.append("\n*Use the Project ID or identifier to list tasks.*")
        return "\n".join(lines)


async def _fetch_project_states(client: httpx.AsyncClient, base: str, workspace: str, project_id: str, headers: Dict[str, str]) -> List[dict]:
    states_url = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/states/"
    status, data = await _plane_get(client, states_url, headers)
    if status == 200:
        if isinstance(data, dict):
            return data.get("results", [])
        if isinstance(data, list):
            return data
    return []


async def list_plane_issues(ctx: Any = None, workspace: str = "", project_id: str = "", state: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    List tasks/issues from a Plane project with their sequence ID, name, status, priority, and UUID.
    Can filter by state (e.g. 'todo', 'in progress', 'done', 'backlog').
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace or not project_id:
        return "Error: Both workspace and project_id are required."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)
    url_items = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/"
    url_legacy = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"

    async with httpx.AsyncClient() as client:
        # Fetch states map in parallel/beforehand for human-readable state names
        states = await _fetch_project_states(client, base, workspace, project_id, headers)
        state_map = {s.get("id"): s.get("name", "") for s in states}

        status, data = await _plane_get(client, url_items, headers, fallback_url=url_legacy)
        if status != 200:
            return f"Error retrieving tasks from Plane project '{project_id}' (HTTP {status}): {data}"

        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not results:
            return f"No tasks found in project '{project_id}'."

        # Filter by state if requested
        filter_state = (state or "").strip().lower()
        filtered = []
        for issue in results:
            st_id = issue.get("state") or issue.get("state_id") or ""
            st_name = state_map.get(st_id) or issue.get("state__name") or "Unknown"
            if filter_state:
                if filter_state not in st_name.lower():
                    continue
            filtered.append((issue, st_name))

        if not filtered:
            return f"No tasks found matching status filter '{state}' in project '{project_id}'."

        lines = [f"**Tasks in Plane project (Total: {len(filtered)}):**\n"]
        for issue, st_name in filtered:
            name = issue.get("name", "Untitled")
            seq_id = issue.get("sequence_id")
            seq_str = f"#{seq_id} " if seq_id else ""
            iid = issue.get("id", "")
            priority = (issue.get("priority") or "none").capitalize()
            lines.append(f"- **{seq_str}{name}** | Status: `{st_name}` | Priority: `{priority}` | ID: `{iid}`")

        return "\n".join(lines)


async def get_plane_issue_details(ctx: Any = None, workspace: str = "", project_id: str = "", issue_id: str = "", token: Optional[str] = None) -> str:
    """
    Get complete details of a specific task: description, status, priority, assignees, dates, and comments.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace or not project_id or not issue_id:
        return "Error: workspace, project_id, and issue_id are all required."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        # Resolve issue ID if user passed sequence number (e.g. "12" or "PROJ-12")
        clean_target = issue_id.strip()
        actual_uuid = clean_target

        if not re.match(r'^[0-9a-fA-F-]{32,36}$', clean_target):
            # List tasks to resolve sequence number or prefix
            url_list = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        url_item = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/{actual_uuid}/"
        url_item_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/{actual_uuid}/"

        status, issue = await _plane_get(client, url_item, headers, fallback_url=url_item_fb)
        if status != 200 or not isinstance(issue, dict):
            return f"Error: Task '{issue_id}' not found in project '{project_id}' (HTTP {status})."

        # States map
        states = await _fetch_project_states(client, base, workspace, project_id, headers)
        state_map = {s.get("id"): s.get("name", "") for s in states}

        st_id = issue.get("state") or issue.get("state_id") or ""
        st_name = state_map.get(st_id) or "Unknown"

        name = issue.get("name", "Untitled")
        seq_id = issue.get("sequence_id", "")
        seq_str = f"#{seq_id} " if seq_id else ""
        priority = (issue.get("priority") or "none").capitalize()
        created_at = (issue.get("created_at") or "")[:10]
        target_date = issue.get("target_date") or "None"

        # Raw or HTML description
        desc_html = issue.get("description_html") or issue.get("description") or "No description provided."
        clean_desc = _strip_html(desc_html)

        # Comments
        comments_url = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/{actual_uuid}/comments/"
        comments_url_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/{actual_uuid}/comments/"
        c_status, c_data = await _plane_get(client, comments_url, headers, fallback_url=comments_url_fb)
        comments_list = c_data.get("results", []) if isinstance(c_data, dict) else (c_data if isinstance(c_data, list) else [])

        comments_text = ""
        if comments_list:
            comments_text = "\n\n**Recent Comments:**\n"
            for c in comments_list[-5:]:
                c_author = (c.get("actor_detail") or {}).get("display_name") or "User"
                c_body = _strip_html(c.get("comment_html") or c.get("comment_json") or "")
                c_date = (c.get("created_at") or "")[:16].replace("T", " ")
                comments_text += f"- *{c_author} ({c_date}):* {c_body}\n"

        return (
            f"### {seq_str}{name}\n"
            f"- **Status:** `{st_name}`\n"
            f"- **Priority:** `{priority}`\n"
            f"- **Target Date:** `{target_date}` | Created: `{created_at}`\n"
            f"- **ID:** `{actual_uuid}`\n\n"
            f"**Description:**\n{clean_desc}"
            f"{comments_text}"
        )


async def create_plane_issue(ctx: Any = None, workspace: str = "", project_id: str = "", name: str = "", description: str = "", priority: str = "none", state_name: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    Create a new task in a Plane project.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace or not project_id or not name:
        return "Error: workspace, project_id, and task name are required."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        payload: Dict[str, Any] = {
            "name": name.strip(),
            "priority": priority.strip().lower() if priority else "none",
        }

        if description:
            clean_desc = html.escape(description.strip()).replace("\n", "<br/>")
            payload["description_html"] = f"<p>{clean_desc}</p>"

        # Optional initial state
        if state_name:
            states = await _fetch_project_states(client, base, workspace, project_id, headers)
            target_st = next((s for s in states if s.get("name", "").lower() == state_name.strip().lower()), None)
            if target_st:
                payload["state"] = target_st["id"]

        url_items = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/"
        url_legacy = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"

        status, created = await _plane_post(client, url_items, headers, payload, fallback_url=url_legacy)
        if status not in (200, 201) or not isinstance(created, dict):
            return f"Error creating task in Plane (HTTP {status}): {created}"

        seq_id = created.get("sequence_id")
        seq_str = f"#{seq_id} " if seq_id else ""
        iid = created.get("id", "")
        return f"Successfully created task **{seq_str}{name}** (ID: `{iid}`)."


async def update_plane_issue_state(ctx: Any = None, workspace: str = "", project_id: str = "", issue_id: str = "", state_name: str = "", token: Optional[str] = None) -> str:
    """
    Update the workflow status of an issue (e.g. 'In Progress', 'In Review', 'Done', 'Backlog', 'Todo').
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace or not project_id or not issue_id or not state_name:
        return "Error: workspace, project_id, issue_id, and state_name are required."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        # 1. Fetch available states
        states = await _fetch_project_states(client, base, workspace, project_id, headers)
        if not states:
            return f"Error: Could not retrieve workflow states for project '{project_id}'."

        norm_target = state_name.strip().lower()
        # Match by state name or state group
        matched_state = next(
            (s for s in states if s.get("name", "").lower() == norm_target or s.get("group", "").lower() == norm_target),
            None
        )
        if not matched_state:
            avail = ", ".join(f"'{s.get('name')}'" for s in states)
            return f"Status '{state_name}' not found. Available statuses in this project: {avail}"

        # 2. Resolve issue UUID if needed
        clean_target = issue_id.strip()
        actual_uuid = clean_target
        if not re.match(r'^[0-9a-fA-F-]{32,36}$', clean_target):
            url_list = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        # 3. Patch state
        url_item = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/{actual_uuid}/"
        url_item_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/{actual_uuid}/"

        patch_payload = {"state": matched_state["id"]}
        status, updated = await _plane_patch(client, url_item, headers, patch_payload, fallback_url=url_item_fb)
        if status not in (200, 201):
            return f"Error updating task status (HTTP {status}): {updated}"

        return f"Successfully updated task `{actual_uuid}` state to **{matched_state.get('name')}**."


async def add_plane_issue_comment(ctx: Any = None, workspace: str = "", project_id: str = "", issue_id: str = "", comment: str = "", token: Optional[str] = None) -> str:
    """
    Add a comment, progress update, or completion report to a Plane task.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    if not workspace or not project_id or not issue_id or not comment:
        return "Error: workspace, project_id, issue_id, and comment text are required."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        # Resolve issue UUID if needed
        clean_target = issue_id.strip()
        actual_uuid = clean_target
        if not re.match(r'^[0-9a-fA-F-]{32,36}$', clean_target):
            url_list = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        comments_url = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/work-items/{actual_uuid}/comments/"
        comments_url_fb = f"{base}/api/v1/workspaces/{workspace}/projects/{project_id}/issues/{actual_uuid}/comments/"

        clean_comment = html.escape(comment.strip()).replace("\n", "<br/>")
        payload = {"comment_html": f"<p>{clean_comment}</p>"}

        status, created = await _plane_post(client, comments_url, headers, payload, fallback_url=comments_url_fb)
        if status not in (200, 201):
            return f"Error posting comment to Plane task (HTTP {status}): {created}"

        return f"Successfully added comment to task `{actual_uuid}`."
