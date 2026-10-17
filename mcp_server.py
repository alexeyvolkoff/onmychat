import os
import sys
import asyncio
import logging
from typing import Optional
import requests

# Setup minimal logging to stderr since stdout is used for MCP stdio
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

from mcp.server.fastmcp import FastMCP

import core_service
import user_context

# Create the FastMCP server
mcp = FastMCP("OnMyDisk MCP Server")

def get_current_user_context() -> user_context.UserContext:
    """
    Construct UserContext from environment variables.
    For STDIO connections (like Hermes local MCP), we rely on OMD_KEY.
    """
    omd_key = os.environ.get("OMD_KEY")
    storage = os.environ.get("OMD_STORAGE", "")
    
    if not omd_key:
        raise ValueError("OMD_KEY environment variable is required to authenticate with OnMyDisk.")
        
    return user_context.get_context_by_account(omd_key, storage)


@mcp.tool()
async def list_omd_files(path: str) -> str:
    """
    List files in a directory. Results show METADATA (size, date) and are SORTED by date (most recent first). 
    Size is in BYTES. Use read_omd_file to see content.
    
    Args:
        path: The absolute path EXACTLY as written by user, e.g. /Linux-desktop/Private/Data
    """
    ctx = get_current_user_context()
    return await core_service.list_omd_files(ctx, path)


@mcp.tool()
async def read_omd_file(path: str) -> str:
    """
    Read the content of a file. Supports .txt, .md, .pdf, .docx, .odt, .csv. 
    For PDFs and documents, the system automatically converts them to text.
    
    Args:
        path: The EXACT absolute path of the file to read, e.g. /Linux-desktop/Private/Data/file.txt
    """
    ctx = get_current_user_context()
    return await core_service.read_omd_file(ctx, path)


@mcp.tool()
async def write_omd_file(path: str, content: str) -> str:
    """
    Write data to a file. 
    
    Args:
        path: The absolute path INCLUDING FILENAME, e.g. /Linux-desktop/Private/Data/invoice.txt
        content: The text content to write to the file
    """
    ctx = get_current_user_context()
    return await core_service.write_omd_file(ctx, path, content)


@mcp.tool()
async def find_omd_file(root_directory: str, condition: str) -> str:
    """
    Find a single file in a directory based on natural language criteria.
    
    Args:
        root_directory: The directory to search within
        condition: Natural language condition (e.g., 'most recent invoice', 'largest pdf')
    """
    ctx = get_current_user_context()
    return await core_service.find_omd_file(ctx, root_directory, condition)


@mcp.tool()
async def search_memory(query: str) -> str:
    """
    Search the internal knowledge base and indexed files for facts or information.
    You MUST use this tool FIRST for any general questions before falling back to the web search.
    
    Args:
        query: The search query
    """
    ctx = get_current_user_context()
    return await core_service.search_memory_tool(ctx, query)


@mcp.tool()
async def read_odt_placeholders(template_path: str) -> str:
    """
    Read an ODT file and extract all unique placeholders of the form {{placeholder}}.
    This is useful for discovering what fields need to be replaced in a template.
    
    Args:
        template_path: The absolute path of the ODT template file on OMD, e.g. /Linux-desktop/Private/Templates/contract.odt
    """
    ctx = get_current_user_context()
    return await core_service.read_odt_placeholders(ctx, template_path)


@mcp.tool()
async def modify_odt_file(template_path: str, output_path: str, replacements: dict) -> str:
    """
    Create a new ODT document by copying a template ODT document and replacing specified placeholders or strings.
    All replacement values are automatically XML-escaped to ensure the document remains valid.
    
    Args:
        template_path: The absolute path of the source template ODT file, e.g. /Linux-desktop/Private/Templates/contract.odt
        output_path: The absolute path of the new ODT file to be written, e.g. /Linux-desktop/Private/Contracts/contract_new.odt
        replacements: A dictionary of key-value pairs to replace in the document. Keys are target strings/placeholders, and values are the new texts.
    """
    ctx = get_current_user_context()
    return await core_service.modify_odt_file(ctx, template_path, output_path, replacements)


def _make_plane_headers(token: str) -> dict:
    return {
        "X-API-Key": token,
        "Content-Type": "application/json"
    }


def _make_plane_urls(workspace: str, project_id: str) -> tuple:
    base_url = "https://api.plane.so/api/v1"
    return f"{base_url}/workspaces/{workspace}/projects/{project_id}/work-items/", f"{base_url}/workspaces/{workspace}/projects/{project_id}/states/"

@mcp.tool()
async def list_plane_workspaces(token: str) -> str:
    """
    List all workspaces available for the user in Plane.
    Use this FIRST when you need to work with Plane but don't know the workspace slug.
    After getting the list, use the workspace slug to list projects.

    Args:
        token: Plane API key (Personal Access Token)
    """
    headers = _make_plane_headers(token)
    url = "https://api.plane.so/api/v1/workspaces/"
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        return f"Error getting workspaces from Plane: {response.status_code}"
    
    workspaces = response.json().get("results", [])
    if not workspaces:
        return "You have no workspaces in Plane."
    
    return "Your workspaces in Plane:\n\n" + "\n".join(
        f"{ws['name']} (slug: {ws['slug']})" for ws in workspaces
    )


@mcp.tool()
async def list_plane_projects(token: str, workspace: str) -> str:
    """
    List all projects in a specific Plane workspace.
    Use this when you know the workspace slug but need to choose a project.
    After getting the list, use the project ID to list issues.

    Args:
        token: Plane API key (Personal Access Token)
        workspace: The workspace slug
    """
    headers = _make_plane_headers(token)
    url = f"https://api.plane.so/api/v1/workspaces/{workspace}/projects/"
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code != 200:
        return f"Error getting projects from Plane: {response.status_code}"
    
    projects = response.json().get("results", [])
    if not projects:
        return f"You have no projects in workspace '{workspace}'."
    
    return f"Your projects in '{workspace}':\n\n" + "\n".join(
        f"{project['name']} (ID: {project['id']})" for project in projects
    )

@mcp.tool()
async def list_plane_issues(token: str, workspace: str, project_id: str) -> str:
    """
    List all tasks from a Plane project.
    Use this when the user asks about their tasks or what to do next.

    Args:
        token: Plane API key (Personal Access Token)
        workspace: The workspace slug
        project_id: The project ID
    """
    headers = _make_plane_headers(token)
    issues_url, _ = _make_plane_urls(workspace, project_id)
    response = requests.get(issues_url, headers=headers, timeout=10)
    if response.status_code != 200:
        return f"Error getting tasks from Plane: {response.status_code}"
    
    issues = response.json().get("results", [])
    if not issues:
        return "You have no tasks in this Plane project."
    
    return "Your tasks in Plane:\n" + "\n".join(
        f"{issue['name']} (ID: {issue['id']})" for issue in issues
    )

@mcp.tool()
async def report_and_close_plane_issue(token: str, workspace: str, project_id: str, issue_id: str) -> str:
    """
    Generate a report for a Plane task and then close it (move to Done).
    Use this when the user asks to write a report for a task, complete it, or close it.

    Args:
        token: Plane API key (Personal Access Token)
        workspace: The workspace slug
        project_id: The project ID
        issue_id: The ID of the task, e.g. "4bb74be5"
    """
    headers = _make_plane_headers(token)
    issues_url, states_url = _make_plane_urls(workspace, project_id)
    issues_response = requests.get(issues_url, headers=headers, timeout=10)
    if issues_response.status_code != 200:
        return f"Error getting tasks from Plane: {issues_response.status_code}"
    issues = issues_response.json().get("results", [])

    issue = next((i for i in issues if i["id"].startswith(issue_id)), None)
    if issue is None:
        return f"Task with code '{issue_id}' not found."

    states_response = requests.get(states_url, headers=headers, timeout=10)
    if states_response.status_code != 200:
        return f"Error getting statuses: {states_response.status_code}"
    states = states_response.json().get("results", [])

    done_state = next((s for s in states if s["group"] == "completed"), None)
    if done_state is None:
        return "Done status not found, cannot close the task."

    close_response = requests.patch(f"{issues_url}{issue['id']}/", headers=headers, json={"state": done_state["id"]}, timeout=10)
    if close_response.status_code not in (200, 201):
        return f"Error closing the task: {close_response.status_code}"

    report = (
        f"Name: {issue.get('name', 'No name')}\n"
        f"ID: {issue.get('id')}\n"
        "Result: the task has been closed (moved to Done)."
    )
    return report


# --- Craft editors: real .docx / .xlsx / .pptx for the agent (WordCraft / GridCraft / DeckCraft) ---

async def _craft_get_data(ctx, path: str, local: str | None) -> bytes:
    """Bytes of the document: straight from the node's disk when it is local,
    over the gateway/P2P stack when it lives on another node."""
    if local:
        with open(local, "rb") as f:
            return f.read()
    return await core_service.download_omd_file(ctx, path)


async def _craft_put_data(ctx, path: str, local: str | None, data: bytes) -> str:
    if local:
        with open(local, "wb") as f:
            f.write(data)
        return f"Saved locally: {path} ({len(data)} bytes)"
    return await core_service.upload_omd_file_binary(ctx, path, data)


def _craft_ext(filename: str) -> str:
    import os as _os
    return _os.path.splitext(filename or "")[1].lower()


@mcp.tool()
async def read_document(path: str, mode: str = "text") -> str:
    """
    Read a real office document with WordCraft/GridCraft (full fidelity, better than plain text).
    mode="text" — plain text; mode="markdown" — Markdown (text documents only); mode="structure" — JSON.
    Supported: .docx .odt .rtf .doc .md .html .txt (and .xlsx .csv as text).
    
    Args:
        path: The absolute path of the document on OMD, e.g. /Linux-desktop/Private/reports/q3.docx
        mode: "text" (default), "markdown" or "structure"
    """
    ctx = get_current_user_context()
    import craft_service as cs
    name = path.rsplit("/", 1)[-1]
    kind = cs.kind_for_filename(name)
    if not kind:
        return f"Unsupported document type: {name!r} (supported: docx, odt, rtf, doc, md, html, txt, xlsx, csv)"
    local = cs.resolve_local_path(ctx, path)
    data = await _craft_get_data(ctx, path, local)
    if mode == "markdown":
        return cs.read_document_markdown(kind, data, name)
    if mode == "structure":
        return cs.inspect_document(kind, data, name)
    text = cs.read_document_text(kind, data, name)
    if len(text) > 200_000:
        text = text[:200_000] + "\n… [truncated, " + str(len(text)) + " chars total]"
    return text


@mcp.tool()
async def create_document(path: str, content: str) -> str:
    """
    Create a REAL office document on OMD from text the agent wrote.
    The file extension decides the format: .docx/.odt/.rtf → content is MARKDOWN
    (# headings, **bold**, lists, tables all survive); .xlsx → content is CSV;
    .pptx → content is an outline (one line = one slide title, indented lines are bullets).
    The result opens in the OMD editor (WordCraft/GridCraft/DeckCraft) with full formatting.
    
    Args:
        path: The absolute target path on OMD INCLUDING the file name, e.g. /Linux-desktop/Private/reports/q3-report.docx
        content: Markdown / CSV / outline text as described above
    """
    ctx = get_current_user_context()
    import craft_service as cs
    name = path.rsplit("/", 1)[-1]
    kind = cs.kind_for_filename(name)
    if not kind:
        return f"Unsupported target type: {name!r} (use .docx/.odt/.rtf for Markdown, .xlsx for CSV, .pptx for an outline)"
    local = cs.resolve_local_path(ctx, path)
    data = cs.create_document(kind, name, content)
    return await _craft_put_data(ctx, path, local, data)


@mcp.tool()
async def edit_document(path: str, commands: list, save_as: str = "", author: str = "OMD Agent") -> str:
    """
    Edit a REAL office document in place with engine commands, then save it back to OMD.
    commands: a list of strings "id={json params}", e.g. ["caret.docEnd={}", "text.insert={\"text\":\"hi\"}"].
    File-system commands (file.*, anything with a path) are NOT allowed — this tool itself
    downloads and uploads the document. Tracked changes/comments are authored by `author`.
    Use read_document(mode="structure") and list_document_commands to discover what is possible.
    
    Args:
        path: The absolute path of the document on OMD
        commands: list of "id={json}" command strings (max 200)
        save_as: optional alternative absolute path to save under (defaults to the same path)
        author: the name recorded for tracked changes and comments
    """
    ctx = get_current_user_context()
    import craft_service as cs
    name = path.rsplit("/", 1)[-1]
    kind = cs.kind_for_filename(name)
    if not kind:
        return f"Unsupported document type: {name!r}"
    if not isinstance(commands, list) or not commands:
        return "commands must be a non-empty list of 'id={json}' strings"
    local = cs.resolve_local_path(ctx, path)
    data = await _craft_get_data(ctx, path, local)
    out_name = save_as.rsplit("/", 1)[-1] if save_as else name
    edited = cs.edit_document(kind, data, name, [str(c) for c in commands], author=author, out_name=out_name)
    return await _craft_put_data(ctx, save_as or path, local if not save_as else None, edited)


@mcp.tool()
async def convert_document(path: str, out_path: str) -> str:
    """
    Convert a document between formats with the real engines: docx→pdf, md→docx,
    docx→odt, xlsx→csv, csv→xlsx… The output extension decides the format.
    
    Args:
        path: The absolute path of the source document on OMD
        out_path: The absolute target path on OMD (the extension decides the format)
    """
    ctx = get_current_user_context()
    import craft_service as cs
    name = path.rsplit("/", 1)[-1]
    kind = cs.kind_for_filename(name)
    if not kind:
        return f"Unsupported source type: {name!r}"
    local = cs.resolve_local_path(ctx, path)
    data = await _craft_get_data(ctx, path, local)
    local_out = cs.resolve_local_path(ctx, out_path) if out_path.rsplit("/", 1)[0] == path.rsplit("/", 1)[0] else None
    converted = cs.convert_document(kind, data, name, out_path.rsplit("/", 1)[-1])
    return await _craft_put_data(ctx, out_path, local_out, converted)


@mcp.tool()
async def list_document_commands(prefix: str = "") -> str:
    """
    List the engine commands an agent can run in edit_document (422 in WordCraft alone:
    text, formatting, tables, references, review…). Filter by prefix, e.g. "table." or "review.".
    
    Args:
        prefix: optional command prefix filter, e.g. "insert." or "format."
    """
    import craft_service as cs
    import json
    import subprocess
    sections = []
    total = 0
    for kind, label in (("doc", "wordcraft (docx/odt/rtf)"), ("sheet", "gridcraft (xlsx/csv)"), ("deck", "deckcraft (pptx)")):
        cli = cs._cli_path(kind)
        if not cli:
            continue
        try:
            r = subprocess.run([cli, "commands", "--json"], capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                continue
            catalog = json.loads(r.stdout)
            if isinstance(catalog, dict) and "commands" in catalog:
                catalog = catalog["commands"]
            ids = [c.get("id", "") for c in catalog if isinstance(c, dict)]
        except Exception:
            continue
        if prefix:
            ids = [i for i in ids if i.startswith(prefix)]
        if ids:
            total += len(ids)
            sections.append(f"## {label}\n" + "\n".join(ids[:400]))
    if not sections:
        return f"no commands match prefix {prefix!r} (or no craft CLIs installed)"
    return "\n\n".join(sections) + (f"\n… ({total} total)" if total > 400 else "")


if __name__ == "__main__":
    # Start the MCP server using standard IO (suitable for Cursor, Claude Desktop, Hermes Agent)
    # Logging will go to stderr, MCP protocol messages to stdout.
    logging.info("Starting OnMyDisk MCP Server via STDIO...")
    mcp.run(transport='stdio')
