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
    # Normalize Plane cloud web frontend SPA URL (app.plane.so) to API endpoint (api.plane.so)
    if "app.plane.so" in url or url in ("https://plane.so", "http://plane.so"):
        return DEFAULT_PLANE_BASE_URL
    if url.endswith("/api/v1") or url.endswith("/api/v2"):
        url = url.rsplit("/api/", 1)[0]
    return url


def resolve_plane_workspace(ctx: Any = None, workspace: Optional[str] = None) -> str:
    """
    Resolve workspace slug:
    1. Explicit parameter (if URL, extracts slug)
    2. ctx.mcp_keys['plane_workspace']
    3. Slug extracted from ctx.mcp_keys['plane_url'] if present (e.g. https://app.plane.so/my-workspace)
    4. Environment variable PLANE_WORKSPACE
    5. Default 'onmydisk'
    """
    if workspace and isinstance(workspace, str) and workspace.strip():
        ws = workspace.strip()
        if "plane.so/" in ws:
            ws = ws.split("plane.so/")[1].split("/")[0]
        return ws.strip("/")
    if ctx and hasattr(ctx, "mcp_keys") and isinstance(ctx.mcp_keys, dict):
        ws = ctx.mcp_keys.get("plane_workspace")
        if ws and str(ws).strip():
            return str(ws).strip()
        url = ctx.mcp_keys.get("plane_url") or ""
        if "plane.so/" in url:
            parts = url.split("plane.so/")[1].split("/")
            if parts and parts[0] and parts[0] not in ("api", "auth", "spaces", "instances"):
                return parts[0].strip("/")
    env_ws = os.environ.get("PLANE_WORKSPACE")
    if env_ws and env_ws.strip():
        return env_ws.strip()
    return "onmydisk"


def _make_headers(token: str) -> Dict[str, str]:
    return {
        "X-API-Key": token,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "OnMyChat-MCP/1.0",
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
        
        # Check if response is HTML instead of JSON
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type or resp.text.strip().startswith("<!DOCTYPE") or resp.text.strip().startswith("<html"):
            return 502, "Error: Plane returned HTML instead of JSON. Ensure your API URL is https://api.plane.so (not app.plane.so)."

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
        
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type or resp.text.strip().startswith("<!DOCTYPE") or resp.text.strip().startswith("<html"):
            return 502, "Error: Plane returned HTML instead of JSON. Ensure your API URL is https://api.plane.so (not app.plane.so)."

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
        
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type or resp.text.strip().startswith("<!DOCTYPE") or resp.text.strip().startswith("<html"):
            return 502, "Error: Plane returned HTML instead of JSON. Ensure your API URL is https://api.plane.so (not app.plane.so)."

        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return resp.status_code, data
    except Exception as e:
        logger.warning(f"[Plane API] PATCH {url} failed: {e}")
        return 500, str(e)


async def _fetch_workspace_projects(client: httpx.AsyncClient, base: str, workspace: str, headers: Dict[str, str]) -> List[dict]:
    url = f"{base}/api/v1/workspaces/{workspace}/projects/"
    st, data = await _plane_get(client, url, headers)
    if st == 200:
        if isinstance(data, dict):
            return data.get("results", [])
        if isinstance(data, list):
            return data
    return []


async def _resolve_project_id(client: httpx.AsyncClient, base: str, workspace: str, project_id: Optional[str], headers: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve project_id (UUID). If empty or given as name/identifier, fetch projects and resolve.
    Returns (resolved_uuid, error_message).
    """
    target = (project_id or "").strip()
    if target and re.match(r'^[0-9a-fA-F-]{32,36}$', target):
        return target, None

    projects = await _fetch_workspace_projects(client, base, workspace, headers)
    if not projects:
        return None, f"No projects found in workspace '{workspace}'."

    if not target:
        if len(projects) == 1:
            return projects[0].get("id"), None
        names = ", ".join([f"**{p.get('name')}** (ID: `{p.get('id')}`)" for p in projects])
        return None, f"Multiple projects found in workspace '{workspace}'. Please specify `project_id`:\n{names}"

    # Search by identifier or name (case-insensitive)
    t_lower = target.lower()
    for p in projects:
        if p.get("identifier", "").lower() == t_lower or p.get("name", "").lower() == t_lower:
            return p.get("id"), None

    names = ", ".join([f"**{p.get('name')}** [{p.get('identifier')}]" for p in projects])
    return None, f"Project '{target}' not found in workspace '{workspace}'. Available projects: {names}"


async def list_plane_workspaces(ctx: Any = None, token: Optional[str] = None) -> str:
    """
    List all workspaces available to the user in Plane.
    Returns workspace names, slugs, and projects.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    # In Plane REST API, Personal Access Tokens (X-API-Key) operate with workspace-scoped endpoints.
    # We probe candidate slugs: configured workspace, URL slug, known defaults, and user profile.
    candidates: List[str] = []

    # 1. From ctx.mcp_keys
    if ctx and hasattr(ctx, "mcp_keys") and isinstance(ctx.mcp_keys, dict):
        ws = ctx.mcp_keys.get("plane_workspace")
        if ws and str(ws).strip():
            candidates.append(str(ws).strip())
        url = ctx.mcp_keys.get("plane_url") or ""
        if "plane.so/" in url:
            part = url.split("plane.so/")[1].split("/")[0]
            if part and part not in ("api", "auth", "spaces", "instances") and part not in candidates:
                candidates.append(part)

    # 2. Known default slugs
    for default_slug in ["onmydisk", "kvik", "omd"]:
        if default_slug not in candidates:
            candidates.append(default_slug)

    verified: List[Tuple[str, List[dict]]] = []
    async with httpx.AsyncClient() as client:
        # Check user profile
        user_status, user_data = await _plane_get(client, f"{base}/api/v1/users/me/", headers)
        if user_status == 200 and isinstance(user_data, dict):
            disp = (user_data.get("display_name") or user_data.get("first_name") or "").lower().strip()
            if disp and disp not in candidates:
                candidates.append(disp)

        for slug in candidates:
            projects = await _fetch_workspace_projects(client, base, slug, headers)
            if projects:
                verified.append((slug, projects))
            else:
                m_st, m_data = await _plane_get(client, f"{base}/api/v1/workspaces/{slug}/members/", headers)
                if m_st == 200 and isinstance(m_data, list):
                    verified.append((slug, []))

    if not verified:
        return (
            "Plane API requires specifying the workspace slug (from your URL: `https://app.plane.so/{workspace_slug}`).\n"
            "None of the default workspace slugs could be verified. "
            "Please provide your workspace slug (e.g. `list_plane_projects(workspace='...')`)."
        )

    lines = ["**Your Plane Workspaces:**\n"]
    for slug, projs in verified:
        if projs:
            p_desc = ", ".join([f"**{p.get('name')}** (ID: `{p.get('id')}`)" for p in projs])
            lines.append(f"- **{slug}** (slug: `{slug}`) — {len(projs)} project(s): {p_desc}")
        else:
            lines.append(f"- **{slug}** (slug: `{slug}`) — 0 projects")

    lines.append("\n*Note: Plane Personal Access Tokens require workspace slugs. "
                 "If you have another workspace, specify its slug (e.g. `workspace='your-other-workspace'`).*")
    return "\n".join(lines)


async def list_plane_projects(ctx: Any = None, workspace: str = "", token: Optional[str] = None) -> str:
    """
    List all projects in a specific Plane workspace.
    """
    tok = resolve_plane_token(ctx, token)
    if not tok:
        return "Error: Plane API key not configured. Please save your Plane key in Settings -> Security & Keys."

    ws = resolve_plane_workspace(ctx, workspace)
    if not ws:
        return "Error: workspace slug is required. Call `list_plane_workspaces` first if unknown."

    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        results = await _fetch_workspace_projects(client, base, ws, headers)
        if not results:
            return f"No projects found in workspace '{ws}'."

        lines = [f"**Projects in Plane workspace '{ws}':**\n"]
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

    ws = resolve_plane_workspace(ctx, workspace)
    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        pid, err = await _resolve_project_id(client, base, ws, project_id, headers)
        if err or not pid:
            return err or f"Error: Could not resolve project_id in workspace '{ws}'."

        # Fetch states map in parallel/beforehand for human-readable state names
        states = await _fetch_project_states(client, base, ws, pid, headers)
        state_map = {s.get("id"): s.get("name", "") for s in states}

        url_items = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/"
        url_legacy = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/"

        status, data = await _plane_get(client, url_items, headers, fallback_url=url_legacy)
        if status != 200:
            return f"Error retrieving tasks from Plane project '{pid}' (HTTP {status}): {data}"

        results = data.get("results", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not results:
            return f"No tasks found in project '{pid}'."

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
            return f"No tasks found matching status filter '{state}' in project '{pid}'."

        lines = [f"**Tasks in Plane project '{pid}' (Total: {len(filtered)}):**\n"]
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

    if not issue_id:
        return "Error: issue_id is required."

    ws = resolve_plane_workspace(ctx, workspace)
    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        pid, err = await _resolve_project_id(client, base, ws, project_id, headers)
        if err or not pid:
            return err or f"Error: Could not resolve project_id in workspace '{ws}'."

        clean_target = issue_id.strip()
        actual_uuid = clean_target

        if not re.match(r'^[0-9a-fA-F-]{32,36}$', clean_target):
            # List tasks to resolve sequence number or prefix
            url_list = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        url_item = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/{actual_uuid}/"
        url_item_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/{actual_uuid}/"

        status, issue = await _plane_get(client, url_item, headers, fallback_url=url_item_fb)
        if status != 200 or not isinstance(issue, dict):
            return f"Error: Task '{issue_id}' not found in project '{pid}' (HTTP {status})."

        # States map
        states = await _fetch_project_states(client, base, ws, pid, headers)
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
        comments_url = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/{actual_uuid}/comments/"
        comments_url_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/{actual_uuid}/comments/"
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

    if not name:
        return "Error: task name is required."

    ws = resolve_plane_workspace(ctx, workspace)
    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        pid, err = await _resolve_project_id(client, base, ws, project_id, headers)
        if err or not pid:
            return err or f"Error: Could not resolve project_id in workspace '{ws}'."

        payload: Dict[str, Any] = {
            "name": name.strip(),
            "priority": priority.strip().lower() if priority else "none",
        }

        if description:
            clean_desc = html.escape(description.strip()).replace("\n", "<br/>")
            payload["description_html"] = f"<p>{clean_desc}</p>"

        # Optional initial state
        if state_name:
            states = await _fetch_project_states(client, base, ws, pid, headers)
            target_st = next((s for s in states if s.get("name", "").lower() == state_name.strip().lower()), None)
            if target_st:
                payload["state"] = target_st["id"]

        url_items = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/"
        url_legacy = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/"

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

    if not issue_id or not state_name:
        return "Error: issue_id and state_name are required."

    ws = resolve_plane_workspace(ctx, workspace)
    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        pid, err = await _resolve_project_id(client, base, ws, project_id, headers)
        if err or not pid:
            return err or f"Error: Could not resolve project_id in workspace '{ws}'."

        # 1. Fetch available states
        states = await _fetch_project_states(client, base, ws, pid, headers)
        if not states:
            return f"Error: Could not retrieve workflow states for project '{pid}'."

        norm_target = state_name.strip().lower()
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
            url_list = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        # 3. Patch state
        url_item = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/{actual_uuid}/"
        url_item_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/{actual_uuid}/"

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

    if not issue_id or not comment:
        return "Error: issue_id and comment text are required."

    ws = resolve_plane_workspace(ctx, workspace)
    base = resolve_plane_base_url(ctx)
    headers = _make_headers(tok)

    async with httpx.AsyncClient() as client:
        pid, err = await _resolve_project_id(client, base, ws, project_id, headers)
        if err or not pid:
            return err or f"Error: Could not resolve project_id in workspace '{ws}'."

        clean_target = issue_id.strip()
        actual_uuid = clean_target
        if not re.match(r'^[0-9a-fA-F-]{32,36}$', clean_target):
            url_list = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/"
            url_list_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/"
            st_code, list_data = await _plane_get(client, url_list, headers, fallback_url=url_list_fb)
            if st_code == 200:
                items = list_data.get("results", []) if isinstance(list_data, dict) else (list_data if isinstance(list_data, list) else [])
                seq_target = clean_target.split("-")[-1]
                for it in items:
                    if str(it.get("sequence_id")) == seq_target or it.get("id", "").startswith(clean_target):
                        actual_uuid = it["id"]
                        break

        comments_url = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/work-items/{actual_uuid}/comments/"
        comments_url_fb = f"{base}/api/v1/workspaces/{ws}/projects/{pid}/issues/{actual_uuid}/comments/"

        clean_comment = html.escape(comment.strip()).replace("\n", "<br/>")
        payload = {"comment_html": f"<p>{clean_comment}</p>"}

        status, created = await _plane_post(client, comments_url, headers, payload, fallback_url=comments_url_fb)
        if status not in (200, 201):
            return f"Error posting comment to Plane task (HTTP {status}): {created}"

        return f"Successfully added comment to task `{actual_uuid}`."
