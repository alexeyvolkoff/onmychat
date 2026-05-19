import os
import sys
import asyncio
import logging
from typing import Optional

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


if __name__ == "__main__":
    # Start the MCP server using standard IO (suitable for Cursor, Claude Desktop, Hermes Agent)
    # Logging will go to stderr, MCP protocol messages to stdout.
    logging.info("Starting OnMyDisk MCP Server via STDIO...")
    mcp.run(transport='stdio')
