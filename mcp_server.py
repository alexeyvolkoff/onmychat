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


if __name__ == "__main__":
    # Start the MCP server using standard IO (suitable for Cursor, Claude Desktop, Hermes Agent)
    # Logging will go to stderr, MCP protocol messages to stdout.
    logging.info("Starting OnMyDisk MCP Server via STDIO...")
    mcp.run(transport='stdio')
