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
import plane_service

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
        
    ctx = user_context.get_context_by_account(omd_key, storage)
    # This is the per-owner stdio server: it is launched by the node owner on
    # this machine, so craft_service.resolve_local_path may touch the local disk.
    # (HTTP contexts get private_mode only from api.trusted_private_access.)
    ctx.private_mode = True
    ctx.trusted_owner = True  # own node, own disk: read/write files directly
    return ctx


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
async def write_omd_file(path: str, content: str = "", template_path: str = "", replacements: dict | None = None) -> str:
    """
    Write data to a file or create documents in OnMyDisk storage.
    
    Args:
        path: The absolute path INCLUDING FILENAME, e.g. /Linux-desktop/Private/Data/invoice.txt
        content: The text or markdown content to write to the file
        template_path: Optional path to template document if copying/modifying an existing document
        replacements: Optional key-value dictionary of string replacements to perform on the template
    """
    ctx = get_current_user_context()
    return await core_service.write_omd_file(ctx, path, content, template_path=template_path, replacements=replacements)


@mcp.tool()
async def create_document_from_template(template_path: str, target_path: str, replacements: dict) -> str:
    """
    Create a new office document (.docx, .odt, .xlsx, .pptx) based on an existing template or document,
    preserving 100% of formatting, tables, styles, logos, and layout. Replaces specified text strings in the template with new values.
    
    Args:
        template_path: The absolute path of the existing template/document to copy from, e.g. /beelink/Documents/Bineon_invoice.docx
        target_path: The absolute path for the new document to create, e.g. /beelink/Documents/Bineon_invoice_10.docx
        replacements: Dictionary of text replacements where keys are the exact original text in the template and values are the new replacement text, e.g. {'014/2026': '015/2026', '05.05.2026': '18.10.2026'}
    """
    ctx = get_current_user_context()
    return await core_service.create_document_from_template(ctx, template_path, target_path, replacements=replacements)


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





# --- Plane.so Tools (Project & Task Management for Developers) ---

@mcp.tool()
async def list_plane_workspaces(token: Optional[str] = None) -> str:
    """
    List all workspaces available to the user in Plane.
    Use this FIRST when you need to work with Plane but don't know the workspace slug.
    After getting the list, use the workspace slug to list projects.

    Args:
        token: Optional Plane API key (Personal Access Token). If omitted, automatically uses the key configured in settings.
    """
    ctx = get_current_user_context()
    return await plane_service.list_plane_workspaces(ctx, token=token)


@mcp.tool()
async def list_plane_projects(workspace: str, token: Optional[str] = None) -> str:
    """
    List all projects in a specific Plane workspace.
    Use this when you know the workspace slug but need to choose a project.
    After getting the list, use the project ID to list tasks.

    Args:
        workspace: The workspace slug
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.list_plane_projects(ctx, workspace=workspace, token=token)


@mcp.tool()
async def list_plane_issues(workspace: str, project_id: str, state: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    List tasks/issues from a Plane project with their sequence ID, name, status, priority, and UUID.
    Can filter by state (e.g. 'todo', 'in progress', 'done', 'backlog').

    Args:
        workspace: The workspace slug
        project_id: The project ID or identifier
        state: Optional state filter (e.g. 'todo', 'in progress', 'done')
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.list_plane_issues(ctx, workspace=workspace, project_id=project_id, state=state, token=token)


@mcp.tool()
async def get_plane_issue_details(workspace: str, project_id: str, issue_id: str, token: Optional[str] = None) -> str:
    """
    Get complete details of a specific Plane task/issue, including full description, status, priority, target dates, assignees, and recent comments.

    Args:
        workspace: The workspace slug
        project_id: The project ID or identifier
        issue_id: The task UUID, short ID, or sequence number (e.g. '12' or 'PROJ-12')
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.get_plane_issue_details(ctx, workspace=workspace, project_id=project_id, issue_id=issue_id, token=token)


@mcp.tool()
async def create_plane_issue(workspace: str, project_id: str, name: str, description: str = "", priority: str = "none", state_name: Optional[str] = None, token: Optional[str] = None) -> str:
    """
    Create a new task/issue in a Plane project.

    Args:
        workspace: The workspace slug
        project_id: The project ID
        name: Title of the task/issue
        description: Optional detailed description of the task (Markdown or plain text)
        priority: Optional priority ('urgent', 'high', 'medium', 'low', 'none')
        state_name: Optional initial state name (e.g. 'Todo', 'Backlog')
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.create_plane_issue(ctx, workspace=workspace, project_id=project_id, name=name, description=description, priority=priority, state_name=state_name, token=token)


@mcp.tool()
async def update_plane_issue_state(workspace: str, project_id: str, issue_id: str, state_name: str, token: Optional[str] = None) -> str:
    """
    Update the workflow status of an issue (e.g. 'In Progress', 'In Review', 'Done', 'Backlog', 'Todo').

    Args:
        workspace: The workspace slug
        project_id: The project ID
        issue_id: The task UUID or sequence number (e.g. '12' or 'PROJ-12')
        state_name: Target state name (e.g. 'In Progress', 'In Review', 'Done', 'Todo', 'Backlog')
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.update_plane_issue_state(ctx, workspace=workspace, project_id=project_id, issue_id=issue_id, state_name=state_name, token=token)


@mcp.tool()
async def add_plane_issue_comment(workspace: str, project_id: str, issue_id: str, comment: str, token: Optional[str] = None) -> str:
    """
    Add a comment, progress update, or completion report to a Plane task/issue.

    Args:
        workspace: The workspace slug
        project_id: The project ID
        issue_id: The task UUID or sequence number (e.g. '12' or 'PROJ-12')
        comment: The comment text or report to attach
        token: Optional Plane API key
    """
    ctx = get_current_user_context()
    return await plane_service.add_plane_issue_comment(ctx, workspace=workspace, project_id=project_id, issue_id=issue_id, comment=comment, token=token)


@mcp.tool()
async def report_and_close_plane_issue(workspace: str, project_id: str, issue_id: str, token: Optional[str] = None) -> str:
    """
    Legacy helper: Mark a Plane task as Done/Closed.
    """
    ctx = get_current_user_context()
    return await plane_service.update_plane_issue_state(ctx, workspace=workspace, project_id=project_id, issue_id=issue_id, state_name="Done", token=token)


# --- Craft editors: real .docx / .xlsx / .pptx for the agent (WordCraft / GridCraft / DeckCraft) ---

async def _craft_get_data(ctx, path: str, local: str | None) -> bytes:
    """Bytes of the document: straight from the node's disk when it is local,
    over the gateway/P2P stack when it lives on another node."""
    if local:
        with open(local, "rb") as f:
            return f.read()
    return await core_service.download_omd_file(ctx, path)


def _write_local_atomic(local: str, path: str, data: bytes) -> str:
    """Replace the target atomically (tmp file in the same dir + os.replace):
    a crash mid-write must not truncate the user's document, and missing parent
    folders are created."""
    parent = os.path.dirname(local)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{local}.{os.getpid()}.omd-tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, local)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return f"Saved locally: {path} ({len(data)} bytes)"


async def _craft_put_data(ctx, path: str, local: str | None, data: bytes) -> str:
    if local:
        return await asyncio.to_thread(_write_local_atomic, local, path, data)
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
    if mode not in ("text", "markdown", "structure"):
        return f"Unknown mode {mode!r}: use \"text\", \"markdown\" or \"structure\""
    local = cs.resolve_local_path(ctx, path)
    data = await _craft_get_data(ctx, path, local)
    # CLI runs block, so they go to a worker thread — never the MCP event loop.
    if mode == "markdown":
        text = await asyncio.to_thread(cs.read_document_markdown, kind, data, name)
    elif mode == "structure":
        text = await asyncio.to_thread(cs.inspect_document, kind, data, name)
    else:
        text = await asyncio.to_thread(cs.read_document_text, kind, data, name)
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
    data = await asyncio.to_thread(cs.create_document, kind, name, content)
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
    edited = await asyncio.to_thread(cs.edit_document, kind, data, name, [str(c) for c in commands],
                                     author=author, out_name=out_name)
    # save_as is resolved like any other target (it may be local or another node)
    local_out = cs.resolve_local_path(ctx, save_as) if save_as else local
    return await _craft_put_data(ctx, save_as or path, local_out, edited)


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
    local_out = cs.resolve_local_path(ctx, out_path)
    converted = await asyncio.to_thread(cs.convert_document, kind, data, name, out_path.rsplit("/", 1)[-1])
    return await _craft_put_data(ctx, out_path, local_out, converted)


def _engine_command_ids(cli: str) -> list:
    """Command ids of one engine: `commands --json` where supported, the plain
    `commands` listing otherwise (deckcraft prints "id — label: params" lines)."""
    import json
    import subprocess
    try:
        r = subprocess.run([cli, "commands", "--json"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            catalog = json.loads(r.stdout)
            if isinstance(catalog, dict) and "commands" in catalog:
                catalog = catalog["commands"]
            if isinstance(catalog, list):
                ids = [c.get("id", "") for c in catalog if isinstance(c, dict)]
                ids = [i for i in ids if i]
                if ids:
                    return ids
    except Exception:
        pass
    try:
        r = subprocess.run([cli, "commands"], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return [ln.split(" ", 1)[0].strip() for ln in r.stdout.splitlines()
                    if ln.strip() and not ln[0].isspace()]
    except Exception:
        pass
    return []


def _list_document_commands_sync(prefix: str) -> str:
    import craft_service as cs
    sections = []
    total = 0
    for kind, label in (("doc", "wordcraft (docx/odt/rtf)"), ("sheet", "gridcraft (xlsx/csv)"), ("deck", "deckcraft (pptx)")):
        cli = cs._cli_path(kind)
        if not cli:
            continue
        ids = _engine_command_ids(cli)
        if prefix:
            ids = [i for i in ids if i.startswith(prefix)]
        if ids:
            total += len(ids)
            sections.append(f"## {label}\n" + "\n".join(ids[:400]))
    if not sections:
        return f"no commands match prefix {prefix!r} (or no craft CLIs installed)"
    return "\n\n".join(sections) + (f"\n… ({total} total)" if total > 400 else "")


@mcp.tool()
async def list_document_commands(prefix: str = "") -> str:
    """
    List the engine commands an agent can run in edit_document (422 in WordCraft alone:
    text, formatting, tables, references, review…). Filter by prefix, e.g. "table." or "review.".
    
    Args:
        prefix: optional command prefix filter, e.g. "insert." or "format."
    """
    return await asyncio.to_thread(_list_document_commands_sync, prefix)


if __name__ == "__main__":
    # Start the MCP server using standard IO (suitable for Cursor, Claude Desktop, Hermes Agent)
    # Logging will go to stderr, MCP protocol messages to stdout.
    logging.info("Starting OnMyDisk MCP Server via STDIO...")
    mcp.run(transport='stdio')
