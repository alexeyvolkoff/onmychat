import os
import hashlib
import re
import random
import json
import zipfile
import shutil
import logging
from PIL import Image, PngImagePlugin
import io

import aiohttp
import time
import subprocess

from typing import AsyncGenerator

from config import SETTINGS
from config import USER_DATA_DIR

from utils import clean_response, upload_to_storage, upload_data_to_storage, get_image_from_source


import user_context
from user_context import UserContext
from datetime import datetime, timezone
import re
import uuid

from memory_index import (
    add_memory_card,
    fetch_document_text,
    chunk_and_vectorize_to_file,
    search_memories,
    search_indexed_files
)


DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]

# LLM and RAG settings #
OLLAMA_URL = SETTINGS["OLLAMA_URL"]
CODE_BASE_URL = SETTINGS.get("CODE_BASE_URL", "http://localhost:4096")
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]
SFW_MODEL = SETTINGS.get("SFW_MODEL", DEFAULT_MODEL)
NSFW_MODEL = SETTINGS.get("NSFW_MODEL", DEFAULT_MODEL)
CODE_MODEL = SETTINGS.get("CODE_MODEL", DEFAULT_MODEL)
MCP_MODEL = DEFAULT_MODEL

def get_llm_model(ctx: UserContext) -> str:
    if ctx.settings.get("nsfw", False):
        return NSFW_MODEL
    return SFW_MODEL

# Imaging settings #
COMFY_API_URL = SETTINGS["COMFY_API_URL"]

WORKFLOW_PATH = SETTINGS["WORKFLOW_PATH"]
COMFY_OUTPUT_DIR = SETTINGS["COMFY_OUTPUT_DIR"]
COMFY_INPUT_DIR = SETTINGS["COMFY_INPUT_DIR"]
AVATAR_DIR = SETTINGS["AVATAR_DIR"]
STORAGE_ROOT = SETTINGS["STORAGE_ROOT"]
APP_ROOT_DIR = SETTINGS["APP_ROOT_DIR"]
HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"])
GATEWAY_URL = SETTINGS["GATEWAY_URL"]
REASONONG_SUPPORTED = False

def hash_string(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


# Prompt loading logic
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def get_prompt(filename):
    try:
        with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load prompt {filename}: {e}")
        return ""

def get_json_prompt(filename):
    try:
        with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load json prompt {filename}: {e}")
        return {}

# Default system prompts
BASE_SYSTEM_PROMPT = get_prompt("base_system.txt")
MEMORIZATION_PROMPT = get_prompt("memorization.txt")
SYSTEM_INSTRUCTION_CHARACTER = get_prompt("instruction_character.txt")
SYSTEM_INSTRUCTION_GENERAL = get_prompt("instruction_general.txt")
IMAGE_PROMPT_NSFW = get_prompt("image_nsfw.txt")
RAG_SYSTEM_PROMPT = get_prompt("rag_system.txt")
IMPROVEMENT_PROMPT = get_prompt("improvement.txt")

STYLE_MODELS = {
    "realistic": SETTINGS["REALISTIC_MODEL"],
    "realistic_nsfw": SETTINGS["REALISTIC_MODEL_NSFW"],
    "realistic2": SETTINGS.get("REALISTIC2_MODEL", SETTINGS["REALISTIC_MODEL"]),
    "realistic2_nsfw": SETTINGS.get("REALISTIC2_MODEL_NSFW", SETTINGS["REALISTIC_MODEL_NSFW"]),
    "perfect": SETTINGS["PERFECT_MODEL"],
    "perfect_nsfw": SETTINGS["PERFECT_MODEL_NSFW"],
    "fantasy": SETTINGS["FANTASY_MODEL"],
    "fantasy_nsfw": SETTINGS["FANTASY_MODEL_NSFW"],
    "tooned": SETTINGS["TOONED_MODEL"],
    "tooned_nsfw": SETTINGS["TOONED_MODEL_NSFW"],
    "pleasure": SETTINGS["PLEASURE_MODEL"],
    "pleasure_nsfw": SETTINGS["PLEASURE_MODEL_NSFW"],
}

NEGATIVE_PROMPTS = get_json_prompt("negative_prompts.json")
INTENT_PROMPT = get_prompt("intent.txt")
SAFETY_CHECK_PROMPT = get_prompt("safety_check.txt")
SUMMARY_PROMPT = get_prompt("summary.txt")
NSFW_PREPHASE = get_prompt("nsfw_prephase.txt")
DEFAULT_MCP_INSTRUCTIONS = get_prompt("mcp_instructions.txt")
IMAGE_NEUTRAL_DESC_PROMPT = get_prompt("image_neutral_desc.txt")



# Native Tool Definitions for Ollama
MCP_TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "list_omd_files",
      "description": "List files in a directory. Results show METADATA (size, date) and are SORTED by date (most recent first). Size is in BYTES, not money/values. Use read_omd_file to see content.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The absolute path EXACTLY as written by user, e.g. /Linux-desktop/Private/Data"
          }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_omd_file",
      "description": "Read the content of a file. Supports .txt, .md, .pdf, .docx, .odt, .csv. For PDFs and documents, the system automatically converts them to text.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "The EXACT absolute path of the file to read, e.g. /Linux-desktop/Private/Data/file.txt"
          }
        },
        "required": ["path"]
      }
    }
  },
  {
      "type": "function",
      "function": {
          "name": "write_omd_file",
          "description": "Write data to a file. NEVER use this to report errors or report that files were not found. Only write successfully gathered information.",
          "parameters": {
              "type": "object",
              "properties": {
                  "path": {
                      "type": "string",
                      "description": "The absolute path INCLUDING FILENAME, e.g. /Linux-desktop/Private/Data/invoice.txt"
                  },
                  "content": {
                      "type": "string",
                      "description": "The text content to write to the file"
                  }
              },
              "required": ["path", "content"]
          }
      }
  },

  {
      "type": "function",
      "function": {
          "name": "search_memory",
          "description": "Search the internal knowledge base and indexed files for facts or information. You MUST use this tool FIRST for any general questions before falling back to the web search.",
          "parameters": {
              "type": "object",
              "properties": {
                  "query": {
                      "type": "string",
                      "description": "The search query"
                  }
              },
              "required": ["query"]
          }
      }
  },
  {
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "Search the public internet (DuckDuckGo). ONLY use this if `search_memory` failed to find the answer, or if the user explicitly asks for real-time web data (weather, news).",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query to send to the search engine"
          }
        },
        "required": ["query"]
      }
    }
  },
  {
      "type": "function",
      "function": {
          "name": "find_omd_file",
          "description": "Find a single file in a directory based on natural language criteria (e.g., 'most recent', 'biggest', 'plain text').",
          "parameters": {
              "type": "object",
              "properties": {
                  "root_directory": {
                      "type": "string",
                      "description": "The directory to search within"
                  },
                  "condition": {
                      "type": "string",
                      "description": "Natural language condition (e.g., 'most recent invoice', 'largest pdf')"
                  }
              },
              "required": ["root_directory", "condition"]
          }
      }
  },
  {
      "type": "function",
      "function": {
          "name": "save_user_fact",
          "description": "Save persistent, factual information explicitly stated by the user about themselves (e.g., biographical details, job roles, persistent preferences). DO NOT use for temporary context, emotions, or roleplay events.",
          "parameters": {
              "type": "object",
              "properties": {
                  "fact": {
                      "type": "string",
                      "description": "The exact factual statement to memorize"
                  }
              },
              "required": ["fact"]
          }
      }
  },
  {
    "type": "function",
    "function": {
      "name": "read_odt_placeholders",
      "description": "Read an ODT template file and extract all unique placeholders of the form {{placeholder}}. MUST be called before modify_odt_file to discover actual placeholder names — do NOT guess them.",
      "parameters": {
        "type": "object",
        "properties": {
          "template_path": {
            "type": "string",
            "description": "REQUIRED. The absolute path of the ODT template file starting with the active storage root (e.g. /MyDevice/Documents/my_template.odt). Must end in .odt."
          }
        },
        "required": ["template_path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "modify_odt_file",
      "description": "Create a new ODT document by copying a template ODT and substituting placeholders. Call ONLY after read_odt_placeholders has returned the actual placeholder list. All three parameters are REQUIRED.",
      "parameters": {
        "type": "object",
        "properties": {
          "template_path": {
            "type": "string",
            "description": "REQUIRED. The absolute path of the source ODT template file (same path used in read_odt_placeholders), e.g. /MyDevice/Documents/my_template.odt"
          },
          "output_path": {
            "type": "string",
            "description": "REQUIRED. The absolute path where the resulting ODT file will be saved, e.g. /MyDevice/Documents/output_document.odt. Must include filename with .odt extension."
          },
          "replacements": {
            "type": "object",
            "description": "REQUIRED. A non-empty dictionary mapping each placeholder key (as returned by read_odt_placeholders, e.g. 'recipient_name') to its replacement string value (e.g. 'June'). Keys may be bare names or wrapped in {{}}.",
            "additionalProperties": {
              "type": "string"
            }
          }
        },
        "required": ["template_path", "output_path", "replacements"]
      }
    }
  }
]

# === MCP TOOLS ===
def validate_omd_path(ctx: UserContext, path: str):
    from typing import Optional
    if not ctx or not ctx.storage:
        return None
        
    storage_id = ctx.storage.strip("/")
    if not storage_id:
        return None
        
    storage_root = storage_id.split("/")[0] if "/" in storage_id else storage_id
    storage_root_with_slash = "/" + storage_root
    
    path_str = str(path).strip()
    if not path_str.startswith("/"):
        path_str = "/" + path_str
        
    # Check prefix
    if path_str == storage_root_with_slash or path_str.startswith(storage_root_with_slash + "/"):
        return None
        
    return f"Error: Path '{path}' must start with your active storage root: '{storage_root_with_slash}'."

async def find_omd_file(ctx: UserContext, root_directory: str, condition: str) -> str:
    # 1. List files
    listing = await list_omd_files(ctx, root_directory)
    if listing.startswith("Error") or "Result: Directory" in listing:
        return "[NO FILE FOUND]"
    
    # 2. Use LLM to pick the file
    prompt = (
        "You are a file selection assistant.\n"
        "Given the following directory listing, identify the single file that best matches the condition.\n"
        f"Condition: {condition}\n\n"
        f"Listing:\n{listing}\n\n"
        "Respond ONLY with the absolute path of the chosen file in this format: [FILE]:/absolute/path/to/file\n"
        "If no file matches, respond with: [NO FILE FOUND]\n"
        "IMPORTANT: The path must be taken EXACTLY from the [ABS_PATH] in the listing."
    )
    
    payload = {
        "model": get_llm_model(ctx),
        "messages": [{"role": "system", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    data = await llm_request(payload)
    if not data or "message" not in data:
        return "[NO FILE FOUND]"
    
    return data["message"]["content"].strip()

async def download_omd_file(ctx: UserContext, path: str) -> bytes:
    if not ctx.omd_key or not ctx.storage:
        raise ValueError("OMD storage not linked.")
        
    storage_id = ctx.storage.strip("/")
    base_url = GATEWAY_URL.rstrip("/")
    
    clean_path = path if not path.startswith("/") else path[1:]
    root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id

    if path.startswith("/") or path.startswith(root_folder):
        url_path = path.lstrip("/")
    else:
        url_path = f"{storage_id}/{clean_path}"
        
    url = f"{base_url}/{url_path}"
    # Consistent URL-based token authentication
    separator = "&" if "?" in url else "?"
    url += f"{separator}token={ctx.omd_key}"
    
    logging.info(f"Downloading raw binary file from OMD URL: {url.split('token=')[0]}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=30) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                text = await resp.text()
                raise IOError(f"Failed to download file from OMD: Status {resp.status} - {text}")

async def upload_omd_file_binary(ctx: UserContext, path: str, data: bytes) -> str:
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    storage_id = ctx.storage.strip("/")
    storage_key = ctx.omd_key
    base_url = GATEWAY_URL.rstrip("/")
    
    clean_path = path if not path.startswith("/") else path[1:]
    root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id
    
    if path.startswith("/") or path.startswith(root_folder):
        url_path = path.lstrip("/")
        dest_url = f"{base_url}/{url_path}"
        path = "/" + url_path
    else:
        dest_url = f"{base_url}/{storage_id}/{clean_path}"
        path = f"/{storage_id}/{clean_path}"
        
    # Consistent URL-based token authentication
    separator = "&" if "?" in dest_url else "?"
    dest_url += f"{separator}token={storage_key}"
    
    headers = {
        "Response": "json",
        "Content-Type": "application/vnd.oasis.opendocument.text"
    }
    
    logging.info(f"Uploading raw binary file to OMD URL: {dest_url.split('token=')[0]}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.put(dest_url, data=data, headers=headers) as resp:
            if resp.status in [200, 201, 204]:
                return f"Successfully wrote binary file to {path}"
            else:
                text = await resp.text()
                return f"Error writing file: Status {resp.status} - {text}"

def secure_extract_zip(z: zipfile.ZipFile, extract_dir: str):
    """
    Extracts all files from a ZIP archive securely while validating paths
    against Zip Slip directory traversal vulnerabilities.
    """
    extract_dir_abs = os.path.abspath(extract_dir)
    for member in z.infolist():
        # Resolve absolute destination path
        target_path = os.path.abspath(os.path.join(extract_dir_abs, member.filename))
        # Enforce boundary check to prevent Zip Slip
        if not target_path.startswith(extract_dir_abs + os.sep) and target_path != extract_dir_abs:
            raise PermissionError(f"Security Block: Zip Slip directory traversal path detected: {member.filename}")
        # Explicit validation for double dots or absolute path indicators
        if ".." in member.filename or member.filename.startswith("/") or member.filename.startswith("\\"):
            raise PermissionError(f"Security Block: Traversing or absolute filename in ZIP archive: {member.filename}")
        z.extract(member, extract_dir_abs)

async def read_odt_placeholders(ctx: UserContext, path: str) -> str:
    """
    Inspects an ODT document and extracts all unique placeholders of the form {{placeholder}}.
    """
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    path_err = validate_omd_path(ctx, path)
    if path_err:
        return path_err
        
    try:
        file_bytes = await download_omd_file(ctx, path)
        
        # Safe temp directories inside the workspace
        import uuid
        temp_id = str(uuid.uuid4())
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(workspace_dir, "user_data", f"temp_odt_{temp_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        local_odt_path = os.path.join(temp_dir, "temp.odt")
        with open(local_odt_path, "wb") as f:
            f.write(file_bytes)
            
        placeholders = set()
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(local_odt_path, 'r') as z:
            # Secure unzipping with Zip Slip defense
            secure_extract_zip(z, extract_dir)
            
            # Read content from the unzipped files
            for name in ["content.xml", "styles.xml", "meta.xml"]:
                xml_path = os.path.join(extract_dir, name)
                if os.path.exists(xml_path):
                    with open(xml_path, "r", encoding="utf-8", errors="ignore") as f_xml:
                        content = f_xml.read()
                        found = re.findall(r"\{\{([^}]+)\}\}", content)
                        for item in found:
                            # Strip formatting XML tags inside the placeholder to get a clean name
                            clean_item = re.sub(r'<[^>]+>', '', item).strip()
                            if clean_item:
                                placeholders.add(clean_item)
                            
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        if not placeholders:
            return "No placeholders (e.g. {{placeholder}}) found in this ODT document."
            
        return json.dumps(sorted(list(placeholders)), ensure_ascii=False)
        
    except PermissionError as se:
        logging.error(f"Security error in read_odt_placeholders: {se}")
        return f"Security Error: {se}"
    except zipfile.BadZipFile:
        logging.error(f"read_odt_placeholders: BadZipFile for path '{path}' — server returned non-ODT content (possibly 404 HTML).")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return (
            f"Error: The file at '{path}' could not be opened as an ODT document (it is not a valid ZIP/ODT file). "
            f"This usually means the file does not exist at that path, or the path is wrong. "
            f"Check that the path exactly matches what the user specified and that the file exists in their storage. "
            f"Do NOT retry with a guessed or modified filename."
        )
    except Exception as e:
        logging.error(f"Error in read_odt_placeholders: {e}", exc_info=True)
        return f"Error reading placeholders: {e}"

async def modify_odt_file(ctx: UserContext, template_path: str, output_path: str, replacements: dict) -> str:
    """
    Reads a template ODT document from OMD storage, executes an XML-safe search-and-replace
    using replacements, and uploads the resulting ODT document to output_path.
    """
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    if not replacements:
        return "Error: Replacements dictionary is empty."
        
    path_err1 = validate_omd_path(ctx, template_path)
    if path_err1:
        return path_err1
        
    path_err2 = validate_omd_path(ctx, output_path)
    if path_err2:
        return path_err2
        
    try:
        file_bytes = await download_omd_file(ctx, template_path)
        
        # Safe temp directories inside the workspace
        import uuid
        temp_id = str(uuid.uuid4())
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(workspace_dir, "user_data", f"temp_odt_{temp_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        local_odt_path = os.path.join(temp_dir, "temp_in.odt")
        with open(local_odt_path, "wb") as f:
            f.write(file_bytes)
            
        extract_dir = os.path.join(temp_dir, "extract")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(local_odt_path, 'r') as z:
            # Secure unzipping with Zip Slip defense
            secure_extract_zip(z, extract_dir)
            
        # Validate replacements keys against actual placeholders
        actual_placeholders = set()
        for xml_name in ["content.xml", "styles.xml", "meta.xml"]:
            xml_path = os.path.join(extract_dir, xml_name)
            if os.path.exists(xml_path):
                with open(xml_path, "r", encoding="utf-8", errors="ignore") as f_xml:
                    xml_content = f_xml.read()
                    found = re.findall(r"\{\{([^}]+)\}\}", xml_content)
                    for item in found:
                        clean_item = re.sub(r'<[^>]+>', '', item).strip()
                        if clean_item:
                            actual_placeholders.add(clean_item)
                            
        if actual_placeholders:
            logging.info(f"[ODT] Discovered placeholders in template: {sorted(list(actual_placeholders))}")
            
            invalid_keys = []
            for k in replacements.keys():
                k_str = str(k).strip()
                if k_str.startswith("{{") and k_str.endswith("}}"):
                    clean_k = k_str[2:-2].strip()
                else:
                    clean_k = k_str
                if clean_k not in actual_placeholders:
                    invalid_keys.append(k)
            
            if invalid_keys:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return f"Error: The replacements dictionary contains keys that do not exist in the template placeholders: {sorted(invalid_keys)}. The actual placeholders found in this template are: {sorted(list(actual_placeholders))}. Please only use these exact placeholder names as keys in your replacements dictionary."
            
        # XML-escape helpers to avoid breaking target XML markup (equivalent to invoice.py)
        def escape_xml(s) -> str:
            if s is None:
                return ""
            escaped = (
                str(s).replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;")
                      .replace('"', "&quot;")
                      .replace("'", "&apos;")
            )
            return escaped.replace("\n", "<text:line-break/>")
            
        
        # Perform replacements on main files
        xml_names = ["content.xml", "styles.xml", "meta.xml"]
        for name in xml_names:
            xml_path = os.path.join(extract_dir, name)
            if os.path.exists(xml_path):
                with open(xml_path, "r", encoding="utf-8", errors="ignore") as f_xml:
                    content = f_xml.read()
                    
                modified = False
                for key, val in replacements.items():
                    escaped_val = escape_xml(val)
                    
                    # Clean the key to get the raw identifier (e.g. "customer_name")
                    k_str = str(key).strip()
                    if k_str.startswith("{{") and k_str.endswith("}}"):
                        clean_key = k_str[2:-2].strip()
                    else:
                        clean_key = k_str
                        
                    # Advanced Formatting-Tolerant braced replacement:
                    # We find all occurrences of {{...}} and match their text contents.
                    # We then strip any XML formatting/styling tags inside to check if it matches clean_key.
                    # If it matches, we substitute the entire {{...}} block (including formatting) with our value.
                    def replace_braced(match):
                        raw_block = match.group(0) # e.g. {{<text:span>customer_name</text:span>}}
                        inner_content = match.group(1) # e.g. <text:span>customer_name</text:span>
                        clean_inner = re.sub(r'<[^>]+>', '', inner_content).strip()
                        if clean_inner == clean_key:
                            return escaped_val
                        return raw_block
                        
                    new_content = re.sub(r'\{\{([^}]+)\}\}', replace_braced, content)
                    if new_content != content:
                        content = new_content
                        modified = True
                    else:
                        # Fallback: Literal search-and-replace of the original key string wrapped in braces
                        braced_key = k_str if (k_str.startswith("{{") and k_str.endswith("}}")) else f"{{{{{k_str}}}}}"
                        if braced_key in content:
                            content = content.replace(braced_key, escaped_val)
                            modified = True
                        
                if modified:
                    with open(xml_path, "w", encoding="utf-8") as f_xml:
                        f_xml.write(content)
                        
        # Repack everything securely into a new ODT zip container
        # Note: According to ODF specifications, the 'mimetype' file MUST be the first file in the ZIP archive
        # and MUST be stored uncompressed (ZIP_STORED) to ensure office suites recognize it and load native styling.
        output_local_odt = os.path.join(temp_dir, "temp_out.odt")
        with zipfile.ZipFile(output_local_odt, 'w') as z_out:
            # 1. mimetype MUST be the very first file and stored uncompressed
            mimetype_path = os.path.join(extract_dir, "mimetype")
            if os.path.exists(mimetype_path):
                z_out.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)
                
            # 2. All other files compressed
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    full_filepath = os.path.join(root, file)
                    relative_path = os.path.relpath(full_filepath, extract_dir)
                    if relative_path == "mimetype":
                        continue
                    z_out.write(full_filepath, relative_path, compress_type=zipfile.ZIP_DEFLATED)
                    
        # Read the generated modified ODT raw bytes
        with open(output_local_odt, "rb") as f_out:
            modified_bytes = f_out.read()
            
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        # Upload the final raw binary to OMD storage
        return await upload_omd_file_binary(ctx, output_path, modified_bytes)
        
    except PermissionError as se:
        logging.error(f"Security error in modify_odt_file: {se}")
        return f"Security Error: {se}"
    except Exception as e:
        logging.error(f"Error in modify_odt_file: {e}", exc_info=True)
        return f"Error modifying ODT file: {e}"

async def list_supported_tools(ctx: UserContext) -> str:
    if ctx.settings.get("nsfw", False):
        return "No tools available in NSFW mode."
    # Construct description from MCP_TOOLS to ensure it's always accurate
    output = "Currently supported System Tools:\n"
    for tool in MCP_TOOLS:
        name = tool["function"]["name"]
        desc = tool["function"]["description"]
        output += f"- {name}: {desc}\n"
    return output

async def list_omd_files(ctx: UserContext, path: str) -> str:
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    path_err = validate_omd_path(ctx, path)
    if path_err:
        return path_err
    
    try:
        storage_id = ctx.storage.strip("/")
        storage_key = ctx.omd_key
        base_url = GATEWAY_URL.rstrip("/")
        
        # OMD API for listing: GET /<storage>/<path>?list&token=<key>
        # OMD API for listing: GET /<storage>/<path>?list&token=<key>
        root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id
        # Strip leading slash for processing but keep it for comparison
        clean_path = path if not path.startswith("/") else path[1:]
        root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id
        
        # OMD Path Logic:
        # If path starts with /, it is relative to Gateway root. 
        # If it matches root_folder (device), we use it as is.
        # If not, we still allow it as Gateway-absolute.
        
        # Simplified URL Token Authentication
        if path.startswith("/") or path.startswith(root_folder):
            url_path = path.lstrip("/")
            url = f"{base_url}/{url_path}?list&token={storage_key}"
            path = "/" + url_path
        else:
            url = f"{base_url}/{storage_id}/{clean_path}?list&token={storage_key}"
            path = f"/{storage_id}/{clean_path}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                if resp.status in [200, 206]:
                    text = await resp.text()
                    items = []
                    
                    # Robust parsing of concatenated JSON chunks (OMD Transfer-Encoding: chunked)
                    decoder = json.JSONDecoder()
                    pos = 0
                    text_len = len(text.strip())
                    while pos < text_len:
                        # Skip leading whitespace
                        while pos < len(text) and text[pos].isspace():
                            pos += 1
                        if pos >= len(text):
                            break
                        try:
                            obj, idx = decoder.raw_decode(text, pos)
                            pos = idx
                            if "list" in obj:
                                items.extend(obj["list"])
                            elif "result" in obj:
                                if isinstance(obj["result"], list):
                                    items.extend(obj["result"])
                                elif isinstance(obj["result"], dict) and "list" in obj["result"]:
                                    items.extend(obj["result"]["list"])
                        except json.JSONDecodeError as je:
                            logging.error(f"OMD chunk JSON decode error at pos {pos}: {je}")
                            # Fallback if it is a single valid JSON block
                            try:
                                single_obj = json.loads(text)
                                if "list" in single_obj:
                                    items = single_obj["list"]
                                elif "result" in single_obj:
                                    items = single_obj["result"] if isinstance(single_obj["result"], list) else []
                            except Exception:
                                pass
                            break
                    
                    if not items:
                        return f"Result: Directory {path} is empty or does not exist."
                    
                    # Sort items by date descending (most recent first)
                    try:
                        items.sort(key=lambda x: x.get("date", ""), reverse=True)
                    except Exception as e:
                        logging.warning(f"Failed to sort OMD items: {e}")

                    result_str = f"Files in {path} (most recent first):\n"
                    base_dir = path.rstrip("/")
                    for item in items:
                        name = item.get("name", "")
                        type_ = item.get("type", "file")
                        size = item.get("size", "0")
                        date = item.get("date", "")
                        
                        abs_path = f"{base_dir}/{name}"
                        if type_ in ["dir", "directory"]:
                            result_str += f"- [DIRECTORY] [ABS_PATH]: {abs_path}\n"
                        else:
                            result_str += f"- [FILE] [ABS_PATH]: {abs_path} (Size: {size}, Last Modified: {date})\n"
                    return result_str
                else:
                    return f"Error: Could not list directory {path} (Status {resp.status})"
    except Exception as e:
        return f"Exception listing files: {e}"

async def read_omd_file(ctx: UserContext, path: str) -> str:
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    path_err = validate_omd_path(ctx, path)
    if path_err:
        return path_err

    try:
        storage_id = ctx.storage.strip("/")
        base_url = GATEWAY_URL.rstrip("/")
        
        clean_path = path if not path.startswith("/") else path[1:]
        root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id

        if path.startswith("/") or path.startswith(root_folder):
            url_path = path.lstrip("/")
            url = f"{base_url}/{url_path}"
        else:
            url = f"{base_url}/{storage_id}/{clean_path}"
        
        # Use fetch_document_text which handles PDFs via ?totext parameter
        return await fetch_document_text(url, ctx.omd_key)

    except Exception as e:
        return f"Exception reading file: {e}"

async def search_memory_tool(ctx: UserContext, query: str) -> str:
    try:
        collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
        
        # 1. Search semantic memory/knowledge base
        mem_results = search_memories(ctx, query, collection=collection, top_k=3)
        
        # 2. Search indexed files
        file_results = search_indexed_files(ctx, query, top_k=3)
        
        all_results = mem_results + file_results
        
        # Collect sources for the frontend widget (cached on UserContext)
        sources = []
        for res in all_results:
             doc_id = res.get('document_id', '')
             title = res.get('title', '')
             owner = res.get('owner', 'alexey')
             if doc_id or title:
                 # Deduplicate sources
                 filename = title or doc_id.split("/")[-1]
                 if not any(s['title'] == filename and s['owner'] == owner for s in sources):
                     sources.append({
                         "title": filename,
                         "owner": owner,
                         "clickable": False
                     })
        ctx.temp_sources = sources
        
        if not all_results:
            return "No relevant knowledge or files found."
            
        output = f"Knowledge & Indexed Files Search Results for '{query}':\n"
        for i, res in enumerate(all_results, 1):
             text = res.get('text', '')
             source = res.get('source', 'unknown')
             doc_id = res.get('document_id', '')
             title = res.get('title', '')
             
             header = f"Source: {source}"
             if doc_id or title:
                 header += f" ({title or doc_id})"
                 
             output += f"{i}. [{header}]\n{text}\n\n"
             
        return output.strip()
    except Exception as e:
        return f"Error searching memory/files: {e}"

async def write_omd_file(ctx: UserContext, path: str, content: str) -> str:
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."
        
    path_err = validate_omd_path(ctx, path)
    if path_err:
        return path_err

    try:
        from utils import upload_data_to_storage
        # Using the sync util function for now, but wrapping might be better. 
        # Given it's a "tool" that might take time, we accept it might block slightly 
        # or we could rely on OMD being fast. For proper async, should rewrite upload in aiohttp.
        # But to be safe and consistent with existing upload logic including headers construction:
        
        # We need to construct 'dest'. upload_data_to_storage takes (omd_key, dest, filename, data)
        # It constructs url as GATEWAY_URL/dest/filename
        # So we need to split path.
        
        storage_id = ctx.storage.strip("/")
        full_path = f"{storage_id}/{path}" # e.g. storage/user/docs/file.txt
        
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        
        # upload_data_to_storage(omd_key, dest, filename, data)
        # It does PUT {GATEWAY_URL}/{dest}/{filename}
        
        # CAUTION: core_service is async, upload_data_to_storage is sync and uses requests.
        # If this blocks too long, it's bad. But for now, let's use it directly to reuse logic.
        
        # Need to separate storage_id from the rest for 'dest' argument if needed?
        # upload_data_to_storage doc says: dest — полный путь (например "storage/user123/history/chat1.json")
        # And constructs: url = f"{GATEWAY_URL}/{dest}/{filename}?jsonResponse=true"
        # So 'dest' should be the directory path relative to gateway root (which includes storage id)
        
        # Wait, if I pass directory as dest, it appends filename.
        # directory e.g. "storage/user123/docs"
        
        # We can implement a simple async put here to avoid blocking.
        storage_key = ctx.omd_key
        base_url = GATEWAY_URL.rstrip("/")
        if not storage_key or not storage_id:
             return "Error: OMD storage not linked."
             
        root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id
        clean_path = path if not path.startswith("/") else path[1:]
        root_folder = storage_id.split("/")[0] if "/" in storage_id else storage_id
        
        if path.startswith("/") or path.startswith(root_folder):
            url_path = path.lstrip("/")
            dest_url = f"{base_url}/{url_path}"
            path = "/" + url_path
        else:
            dest_url = f"{base_url}/{storage_id}/{clean_path}"
            path = f"/{storage_id}/{clean_path}"
             
        # Fallback if path looks like a directory
        if not os.path.splitext(path)[1]:
             dest_url = dest_url.rstrip("/") + "/new_file.txt"
             path = path.rstrip("/") + "/new_file.txt"
             logging.info(f"[write_omd_file] No extension found, appending default filename: {path}")
        
        # Consistent URL-based token authentication
        separator = "&" if "?" in dest_url else "?"
        dest_url += f"{separator}token={storage_key}"
             
        headers = {
            "Response": "json",
            "Content-Type": "text/plain; charset=utf-8"
        }
        
        async with aiohttp.ClientSession() as session:
             async with session.put(dest_url, data=content.encode("utf-8"), headers=headers) as resp:
                 if resp.status in [200, 201, 204]:
                     return f"Successfully wrote to {path}"
                 else:
                     text = await resp.text()
                     return f"Error writing file: Status {resp.status} - {text}"

    except Exception as e:
        return f"Exception writing file: {e}"

def extract_plan_steps(text: str) -> list:
    steps = []
    lines = text.split('\n')
    in_plan = False
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
        # Skip tags and formatting wrapper lines
        if any(tag in line_strip.upper() for tag in ["[/PLAN]", "[THOUGHT]", "[/THOUGHT]", "<THOUGHT>", "</THOUGHT>", "[CHECKLIST]", "[/CHECKLIST]"]):
            continue
        if "[PLAN]" in line_strip.upper() or "PLAN:" in line_strip.upper() or "CHECKLIST:" in line_strip.upper():
            in_plan = True
            continue
        if in_plan:
            # If we hit a new section like "Tool Call:" or "Agent Reasoning" or something empty/different
            if line_strip.startswith("[") and not line_strip.upper().startswith("[PLAN"):
                break
            # Match plan items
            # Supports: "Step 1: ...", "1. ...", "- [ ] ...", "- ...", "* ..."
            match = re.match(r'^(?:-\s*\[\s*\]\s*|-\s*\[\s*x\s*\]\s*|-\s*|\*\s*|Step\s*\d+\s*:\s*|\d+\.\s*)(.*)', line_strip, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    steps.append(content)
            elif line_strip.startswith("Step") or (line_strip and line_strip[0].isdigit()):
                # Fallback for plain "Step 1 ..." or "1 ..."
                content = re.sub(r'^(?:Step\s*\d+\s*|\d+\s+)', '', line_strip).strip()
                if content:
                    steps.append(content)
    # If no steps found with explicit [PLAN] header, try a global scan of the whole text for "Step X:" or "1. " patterns
    if not steps:
        for line in lines:
            line_strip = line.strip()
            # Skip tags and formatting wrapper lines
            if any(tag in line_strip.upper() for tag in ["[/PLAN]", "[THOUGHT]", "[/THOUGHT]", "<THOUGHT>", "</THOUGHT>", "[CHECKLIST]", "[/CHECKLIST]"]):
                continue
            match = re.match(r'^(?:Step\s*\d+\s*:\s*|\d+\.\s+)(.*)', line_strip, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    steps.append(content)
    return steps

def get_non_plan_reasoning(text: str) -> str:
    if not text:
        return ""
    lines = text.split('\n')
    non_plan_lines = []
    in_plan = False
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            non_plan_lines.append("")
            continue
        # Skip tags and formatting wrapper lines entirely from reasoning
        if any(tag in line_strip.upper() for tag in ["[PLAN]", "[/PLAN]", "[THOUGHT]", "[/THOUGHT]", "<THOUGHT>", "</THOUGHT>", "[CHECKLIST]", "[/CHECKLIST]"]):
            continue
        if "[PLAN]" in line_strip.upper() or "PLAN:" in line_strip.upper() or "CHECKLIST:" in line_strip.upper():
            in_plan = True
            continue
        
        is_step = False
        # Matches: "- [ ]", "- [x]", "- [X]", "* Step 1:", "Step 1:", "1.", "- ", "* "
        if re.match(r'^(?:-\s*\[\s*[ xX]?\s*\]\s*|-\s*|\*\s*|Step\s*\d+\s*:\s*|\d+\.\s+)(.*)', line_strip, re.IGNORECASE):
            is_step = True
        elif line_strip.lower().startswith("step") or (line_strip and line_strip[0].isdigit() and len(line_strip) > 1 and line_strip[1:2] in ['.', ' ', ':']):
            is_step = True
            
        if in_plan:
            if line_strip.startswith("[") and not line_strip.upper().startswith("[PLAN"):
                in_plan = False
            else:
                is_step = True
                
        if is_step:
            continue
            
        non_plan_lines.append(line)
        
    # Reconstruct while stripping consecutive empty lines
    cleaned = []
    last_empty = False
    for line in non_plan_lines:
        if not line.strip():
            if not last_empty:
                cleaned.append("")
                last_empty = True
        else:
            cleaned.append(line)
            last_empty = False
            
    return "\n".join(cleaned).strip()

def clean_final_content(text: str) -> str:
    if not text:
        return ""
    # Remove any debug/agent labels
    text = re.sub(r'^(?:Agent\s+Reasoning\s*\(Turn\s*\d+\)\s*:|Agent\s+Conclusion\s*:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n(?:Agent\s+Reasoning\s*\(Turn\s*\d+\)\s*:|Agent\s+Conclusion\s*:)\s*', '\n', text, flags=re.IGNORECASE)
    
    # Strip plan blocks
    text = get_non_plan_reasoning(text)
    
    return text.strip()

def clean_mimetype(filename: str, raw_mimetype: str = None) -> str:
    """
    Cleans and formats mimetype to be strictly compatible with the OnMyDisk frontend flat-file icon assets.
    Replaces '/' with '-' and applies correct custom mappings for common formats.
    """
    import mimetypes
    
    # If raw_mimetype is a valid specific type, normalize it
    if raw_mimetype and raw_mimetype not in ['unknown', 'application/octet-stream', 'application/x-zerosize', '']:
        return str(raw_mimetype).replace('/', '-')
        
    ext = os.path.splitext(filename)[1].lstrip('.').lower()
    
    # Try guessing with Python's library first
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        # Fallback map for common file types that might not be registered in system's mimetypes
        fallback_map = {
            'odt': 'application-vnd.oasis.opendocument.text',
            'ods': 'application-vnd.oasis.opendocument.spreadsheet',
            'odp': 'application-vnd.oasis.opendocument.presentation',
            'md': 'text-x-markdown',
            'sql': 'application-sql',
            'xml': 'application-xml',
            'json': 'application-json',
            'php': 'application-x-php',
            'py': 'application-x-python',
            'js': 'application-x-javascript',
            'css': 'text-css',
            'txt': 'text-plain',
            'pdf': 'application-pdf',
            'docx': 'application-vnd.openxmlformats-officedocument.wordprocessingml.document',
            'xlsx': 'application-vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'pptx': 'application-vnd.openxmlformats-officedocument.presentationml.presentation',
        }
        mime_type = fallback_map.get(ext, 'unknown')
        
    return mime_type.replace('/', '-') if mime_type else 'unknown'

def clean_assistant_response(text: str) -> str:
    """
    Cleans final assistant message without stripping numbered lists, bullet points, or steps.
    Strips the specific [PLAN] checklists to prevent showing them to the user.
    """
    if not text:
        return ""
    # Remove any debug/agent labels
    text = re.sub(r'^(?:Agent\s+Reasoning\s*\(Turn\s*\d+\)\s*:|Agent\s+Conclusion\s*:)\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n(?:Agent\s+Reasoning\s*\(Turn\s*\d+\)\s*:|Agent\s+Conclusion\s*:)\s*', '\n', text, flags=re.IGNORECASE)
    
    # Strip the specific [PLAN] block from the final output, preserving other content
    lines = text.split('\n')
    cleaned_lines = []
    in_plan_block = False
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            cleaned_lines.append(line)
            continue
            
        # Detect start of plan block
        if "[PLAN]" in line_strip.upper() or "PLAN:" in line_strip.upper() or "CHECKLIST:" in line_strip.upper():
            in_plan_block = True
            continue
            
        if in_plan_block:
            # Plan lines usually look like checkboxes: - [ ] or - [x] or * Step X: etc.
            is_checkbox = re.match(r'^(?:-\s*\[\s*[ xX]?\s*\]\s*|Step\s*\d+|Completed in previous turn|\(\s*Completed in previous turn\s*\))', line_strip, re.IGNORECASE)
            # If we hit a non-plan line, terminate the plan block suppression
            if not is_checkbox and not line_strip.startswith(("-", "*", "•")) and len(line_strip) > 0:
                in_plan_block = False
            else:
                # Suppress the plan checkbox line
                continue
                
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines).strip()

async def check_and_execute_mcp(ctx: UserContext, message: str, provided_history: list|None = None) -> AsyncGenerator[dict, None]:
    if ctx.settings.get("nsfw", False):
        return
        
    # Clean the routing prefix / slash command from the beginning of the message to prevent LLM confusion
    clean_message = message.strip()
    for cmd in ["/doc", "/mcp", "/generate", "/sign", "/help", "/forget", "/forget_all", "/chat", "/code", "/import"]:
        if clean_message.lower().startswith(cmd) and (len(clean_message) == len(cmd) or clean_message[len(cmd)].isspace()):
            clean_message = clean_message[len(cmd):].strip()
            break
            
    todos = []
    has_initialized_todos = False
    changed_files = []
    success = True
    tool_error_msg = ""
    # 1. Path Extraction Heuristic (Help the model find the path)
    raw_paths = re.findall(r"(\/[\w\-\.\/]+)", clean_message)
    potential_paths = []
    
    # List of known slash commands to exclude from path detection
    slash_commands = {
        "/doc", "/mcp", "/generate", "/sign", "/help", "/forget", "/forget_all", 
        "/chat", "/code", "/import", "/show", "/view", "/imagine", "/learn", 
        "/recognize", "/detect", "/think", "/explain", "/search", "/nsfw", "/tools"
    }
    
    for p in raw_paths:
        p_clean = p.strip()
        if p_clean.lower() in slash_commands:
            continue
        if p_clean.count("/") == 1 and clean_message.strip().startswith(p_clean):
            continue
        potential_paths.append(p_clean)
        
    path_hint = ""
    
    if potential_paths:
        # Check if any path looks like a specific file with an extension
        files = [p for p in potential_paths if any(p.lower().endswith(ext) for ext in [".odt", ".pdf", ".txt", ".docx", ".csv", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".html", ".css", ".js", ".cpp", ".py"])]
        if files:
            odt_files = [f for f in files if f.lower().endswith(".odt")]
            other_files = [f for f in files if not f.lower().endswith(".odt")]
            
            if odt_files:
                # ODT paths are MANDATORY — model must not invent or alter them
                mandatory_block = (
                    f"\n\nMANDATORY PATH LOCK: The user explicitly provided the following ODT file path(s) in their request: "
                    f"{', '.join(odt_files)}. "
                    f"You MUST use EXACTLY these path(s) character-by-character. "
                    f"Do NOT invent, guess, abbreviate, or modify the filename in any way. "
                    f"Passing any other path to read_odt_placeholders or modify_odt_file is strictly forbidden."
                )
                path_hint += mandatory_block
            
            if other_files:
                path_hint += f"\nSYSTEM HINT: Detected exact file paths: {', '.join(other_files)}. Access/read these files directly using appropriate tools. Do NOT use `list_omd_files` if you already know the exact file path."
        else:
            path_hint = f"\nSYSTEM HINT: Detected paths: {', '.join(potential_paths)}. \nSTRATEGY: You MUST use `list_omd_files` first to see what's inside before trying to read."

    # 2. Native Tool Call Loop (Multi-Turn)
    system_instruction = DEFAULT_MCP_INSTRUCTIONS
    
    # Inject username and current date
    username = ctx.settings.get("name") or ctx.settings.get("username", "User") if ctx else "User"
    current_date = datetime.now().strftime("%d.%m.%Y")
    current_date_iso = datetime.now().strftime("%Y-%m-%d")
    
    system_instruction += f"\n\n- USERNAME: '{username}'"
    system_instruction += f"\n- CURRENT DATE: {current_date} (ISO format: {current_date_iso})"
    
    if ctx and ctx.storage:
        storage_id = ctx.storage.strip("/")
        if storage_id:
            storage_root = "/" + storage_id.split("/")[0] if "/" in storage_id else "/" + storage_id
            system_instruction += f"\n\n- ACTIVE SESSION STORAGE ROOT: '{storage_root}'. All absolute paths you access, list, search inside, or write to MUST start with this root directory. Do NOT guess or use fake directories like '/OnMyDisk' or '/Linux-desktop'."

    # Initial Turn: Mandatory [PLAN] Phase
    # We use a more permissive system prompt for the planning turn.
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"User Request: {clean_message}\n{path_hint}"}
    ]
    
    all_tool_results = ""
    max_turns = 8 # Increased for complex autonomous tasks
    listed_paths = set()
    known_files = set() # Strict cache of verified files
    call_history = set() # Prevent repeated failed attempts
    
    has_read_file = False
    last_listing = "" # For re-injection on errors
    
    # The agent is now autonomous and reasoning-based. 
    # It will manage its own data extraction logic via the [PLAN] mandate.

    
    for turn in range(max_turns):
        # Keep status as 'performing' throughout the turns
        yield {"type": "status", "content": "performing"}
        
        # Determine if reading is required dynamically based on the agent's own planned checklist steps (fully multilingual)
        requires_read = any("read_omd_file" in str(t.get("content", "")) or "read_odt_placeholders" in str(t.get("content", "")) for t in todos) if todos else False

        # Check if read_odt_placeholders was successfully called and extract its output
        discovered_placeholders_str = ""
        for m in reversed(messages):
            if m.get("role") == "tool":
                content_str = m.get("content", "").strip()
                if content_str.startswith("[") and content_str.endswith("]"):
                    try:
                        parsed = json.loads(content_str)
                        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                            discovered_placeholders_str = ", ".join(f"'{x}'" for x in parsed)
                            break
                    except Exception:
                        pass

        # [TURN-SPECIFIC GUIDANCE]
        guidance = "You are an autonomous agent. "
        if discovered_placeholders_str:
             guidance += f"CRITICAL ODT TEMPLATE REMINDER: The template you are modifying has EXACTLY these placeholders: [{discovered_placeholders_str}]. Your `replacements` dictionary for `modify_odt_file` MUST ONLY use keys from this list. Do NOT use any other keys under any circumstances. Map all required document content specifically into these placeholders! "
        # [PROACTIVE DISCOVERY GATE]
        # If Turn 0 and we have a path that looks like a directory, only allow list_omd_files
        available_tools = MCP_TOOLS
        if turn == 0:
             guidance += "MANDATORY: Analyze the request and create a detailed `[PLAN]` checklist. Your plan MUST cover every part of the user's request (e.g., if they ask to 'extract' or 'find amount', your plan MUST include `read_omd_file`). You MUST also call the first tool in your plan if possible."
             # [PROACTIVE DISCOVERY GATE]
             if potential_paths and not known_files:
                  is_dir_like = any(p.endswith("/") or not os.path.splitext(p)[1] for p in potential_paths)
                  if is_dir_like:
                       available_tools = [t for t in MCP_TOOLS if t["function"]["name"] == "find_omd_file" or t["function"]["name"] == "list_omd_files"]
                       logging.info(f"[MCP] Enforcing Discovery Phase for: {potential_paths}")
                       guidance += "\nDISCOVERY PHASE: You MUST use `find_omd_file` or `list_omd_files` first to find the file name."
             
             # Remove strict "Tool Only" constraint for turn 0
             current_messages = [
                 {"role": "system", "content": messages[0]["content"] + f"\n\nCURRENT GUIDANCE: {guidance}"}
             ] + messages[1:]
        else:
             if known_files:
                  guidance = "Proceed executing your `[PLAN]`. Use the absolute file paths already discovered."
             else:
                  guidance += "Continue executing your `[PLAN]`. Mark steps as [x] once done."
             
             # Turn 1+ is strict tool calling
             current_messages = [
                 {
                     "role": "system", 
                     "content": (
                         messages[0]["content"] + 
                         "\nCRITICAL: Respond ONLY with tool calls. Do NOT explain or output conversational text in your response. "
                         "You are 100% allowed and expected to write full natural language content, letters, and text inside the JSON arguments of your tool calls (e.g. inside `replacements` or `content` fields).\n\n"
                         f"CURRENT GUIDANCE: {guidance}"
                     )
                 }
             ] + messages[1:]
        
        # General removal of search_memory if paths were provided
        if potential_paths:
             available_tools = [tool for tool in available_tools if tool["function"]["name"] != "search_memory"]
            
        payload = {
            "model": MCP_MODEL,
            "messages": current_messages,
            "stream": False,
            "tools": available_tools,
            "options": {
                "temperature": 0.0 # Deterministic
            }
        }
        
        response_data = await llm_request(payload)
        if not response_data or "message" not in response_data:
            break
            
        msg = response_data["message"]
        tool_calls = msg.get("tool_calls", [])
        
        if len(tool_calls) > 1:
             # Force pipelining: Only take the first tool call to avoid parallel execution conflicts
             tool_calls = tool_calls[:1]
             msg["tool_calls"] = tool_calls
        
        messages.append(msg)
         
        # [ODT WORKFLOW NUDGE]
        # If we have run read_odt_placeholders but have not yet run modify_odt_file,
        # we MUST ensure the model proceeds to modify the document and does not prematurely stop or output text.
        # This check is placed outside agent_text block so it catches empty/failed tool-calling responses.
        has_read_placeholders = "read_odt_placeholders" in all_tool_results
        has_modified_odt = "modify_odt_file" in all_tool_results
        is_modifying_now = any(tc.get("function", {}).get("name") == "modify_odt_file" for tc in tool_calls) if tool_calls else False
        
        if has_read_placeholders and not has_modified_odt and not is_modifying_now:
              messages.append({
                  "role": "user", 
                  "content": (
                      "Wait! You have successfully read the ODT placeholders, but you have NOT modified the document yet! "
                      "You MUST now calculate the required replacement values and execute `modify_odt_file` to generate the final document. "
                      "Do NOT output markdown invoice templates, text representations, or explanations. You MUST call `modify_odt_file` to save the document."
                  )
              })
              logging.warning(f"[MCP][Turn {turn+1}] Placeholders read but modify_odt_file not called. Injecting ODT Workflow Nudge.")
              tool_calls = []
              msg["tool_calls"] = []
              continue
        
        # [REASONING CAPTURE]
        # Ensure that plans, thoughts, and checklists are preserved and visible to the main assistant.
        agent_text = msg.get("content", "").strip()
        
        # [REPETITIVE GENERATION GUARD]
        # Detect token-looping issues in local LLM output (e.g., endlessly repeating templates/ or words)
        is_looping = False
        if len(agent_text) > 100:
             # Look for consecutive repetitions of substrings of length 3 to 40
             for length in range(3, 40):
                  for i in range(len(agent_text) - length * 6):
                       sub = agent_text[i:i+length]
                       # If the substring repeats 6 or more times consecutively
                       if agent_text[i:].startswith(sub * 6):
                            logging.warning(f"[MCP LOOP DETECTED] LLM repeating '{sub}' infinitely. Breaking stream.")
                            agent_text = agent_text[:i] + f"\n\n[System Alert: Generative loop cut off at '{sub}'].\n"
                            msg["content"] = agent_text
                            is_looping = True
                            break
                  if is_looping:
                       break
        
        if is_looping:
             all_tool_results += f"\n{agent_text}\n"
             break

        logging.info(f"[MCP] Agent Reasoning (Turn {turn+1}):\n{agent_text}")
        if agent_text:
             # [HALLUCINATION FILTER]
             # Prevent the agent from "mimicking" system alerts in its reasoning.
             filtered_lines = [
                 line for line in agent_text.split("\n") 
                 if not any(prefix in line.upper() for prefix in ["SYSTEM ALERT:", "SYSTEM NOTICE:", "SYSTEM ERROR:"])
             ]
             clean_agent_text = "\n".join(filtered_lines).strip()
             
             if clean_agent_text:
                  all_tool_results += f"{clean_agent_text}\n\n"
                  
                  # Show thought block only if there's non-plan reasoning
                  non_plan_reasoning = get_non_plan_reasoning(clean_agent_text)
                  if non_plan_reasoning:
                       yield {
                           "id": f"thought_turn_{turn}",
                           "type": "reasoning",
                           "delta": non_plan_reasoning
                       }
                  
                  # Parse and initialize checklist todos on turn 0
                  if turn == 0 and not has_initialized_todos:
                       steps = extract_plan_steps(clean_agent_text)
                       if steps:
                            todos = [{"content": step, "status": "pending"} for step in steps]
                            has_initialized_todos = True
                            yield {
                                "id": "mcp_todo",
                                "type": "tool",
                                "state": {
                                    "status": "running",
                                    "tool": "todo",
                                    "input": {
                                        "todos": todos
                                    }
                                },
                                "action": "part_update"
                            }
                  
                  # [PLAN LOGGING]
                  if turn == 0:
                       logging.info(f"\n{'='*20} [MCP] GENERATED PLAN {'='*20}\n{clean_agent_text}\n{'='*61}")
                  else:
                       logging.info(f"[MCP][Turn {turn+1}] Agent Reasoning/Plan: {clean_agent_text}")
             
             # [PLAN-BASED LOGGING]
             # We log whether the agent's plan mentions reading, but we do NOT override requires_read.
             # The user's intent (set at the start) is the source of truth.
             if turn == 0:
                  # [PLANNING GATE]
                  # If Turn 0 has a tool call but MISSES the [PLAN] checklist, we intercept.
                  if tool_calls and ("[PLAN]" not in agent_text and "checklist" not in agent_text.lower()):
                       logging.warning("[MCP] Turn 0: Model skipped [PLAN]. Intercepting tool call.")
                       messages.append({"role": "user", "content": "Wait! You MUST provide a detailed `[PLAN]` checklist in and call your first tool Turn 0. Redo your response with a [PLAN]."})
                       # Forget the tool calls for this turn
                       tool_calls = []
                       msg["tool_calls"] = []
                  else:
                       # Log whether the agent's plan includes read tools (for debugging)
                       data_tools = ["read_omd_file", "search_memory", "search_web"]
                       if any(tool in agent_text for tool in data_tools):
                            logging.info("[MCP] Agent PLAN includes data tool.")
                       else:
                            logging.info("[MCP] Agent PLAN does NOT include data tool (user intent will override if needed).")

             # [ODT WORKFLOW NUDGE] (Moved above)
             pass

             # [CONTINUITY NUDGE]
             # Fire if discovery tools were used but read wasn't, and USER'S request implies reading.
             has_discovered = "list_omd_files" in all_tool_results or "find_omd_file" in all_tool_results
             if turn > 0 and not has_read_file and has_discovered and requires_read:
                  # If we discovered a file but haven't read it, force the agent to read.
                  messages.append({"role": "user", "content": "You identified a file. The user's request requires reading/extraction. You MUST call `read_omd_file` using the absolute path provided. Do NOT conclude yet."})
                  logging.warning(f"[MCP][Turn {turn+1}] File discovered but not read (User Intent: READ). Injecting Continuity Nudge.")
                  continue

             # RECORD CONCLUSION ONLY IF NO TOOL CALLS
             if not tool_calls and agent_text and agent_text != "NO_TOOL":
                  # [STRICT STOP CHECK]
                  # Block if user's request implied reading but we haven't read yet.
                  if known_files and not has_read_file and requires_read:
                       messages.append({"role": "user", "content": "Wait! The user asked to READ or EXTRACT data. You haven't read the file yet. You MUST call `read_omd_file` before finishing."})
                       logging.warning(f"[MCP][Turn {turn+1}] Agent attempted to stop without reading (User Intent: READ). Blocking.")
                       continue
                  
                  all_tool_results += f"\n{agent_text}\n"
                  break

            
        # If there are no tool calls and no agent_text (e.g., just an empty message or "NO_TOOL" without content)
        # or if the agent_text was just a plan on turn 0, handle it here.
        if not tool_calls:
            # Plan/Thought turn detection
            content = msg.get("content", "").strip() # Redefine content for this block if agent_text was empty
            if turn == 0 and ("[PLAN]" in content or "checklist" in content.lower()):
                 logging.info(f"[MCP][Turn {turn+1}] First turn was for PLANNING. Continuing to execution.")
                 # Add a system nudge to actually start tool calls if they haven't yet.
                 messages.append({"role": "user", "content": "Plan received. Proceed with the first tool call now."})
                 continue
            # [STUCK/EMPTY RESPONSE RECOVERY FOR ODT]
            # If we successfully read placeholders but have not called modify_odt_file yet, 
            # and the model is stuck/returned an empty response, we force it to proceed.
            has_read_placeholders = "Tool Output (read_odt_placeholders):" in all_tool_results
            has_modified_odt = "Tool Output (modify_odt_file):" in all_tool_results
            
            if has_read_placeholders and not has_modified_odt and turn < max_turns - 1:
                p_list_str = discovered_placeholders_str if discovered_placeholders_str else "content, title"
                logging.warning(f"[MCP][Turn {turn+1}] ODT Stuck Recovery: placeholders read but modify_odt_file not called. Injecting Stuck Nudge.")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Wait! You have successfully read the ODT placeholders, but you have NOT modified the document yet!\n"
                        f"Discovered placeholders: [{p_list_str}].\n\n"
                        f"You MUST now:\n"
                        f"1. First perform a 'Data Calculation & Mapping' analysis in your reasoning block to map the user's request details specifically into these placeholders.\n"
                        f"2. Immediately execute the `modify_odt_file` tool to generate and save the final document.\n\n"
                        f"Respond with both your analysis and the tool call to `modify_odt_file`."
                    )
                })
                continue

            # If we reach here and there are no tool calls, and it wasn't a plan on turn 0,
            # then the agent is done or stuck. Break the loop.
            break
            
        # Execute tool calls
        found_new_info = False
        
        # [SEQUENTIAL ENFORCEMENT]
        # We only take the FIRST tool call. This forces the agent to wait for results 
        # before making the next logical jump (preventing hallucinations).
        tool = tool_calls[0]
        func = tool.get("function", {})
        name = func.get("name")
        args = func.get("arguments", {})
        call_id = tool.get("id")

        # [STATUS YIELD]
        # Map tool name to human-readable status
        status_map = {
            "list_omd_files": "listing",
            "read_omd_file": "reading",
            "find_omd_file": "searching",
            "write_omd_file": "writing",
            "search_web": "searching",
            "search_memory": "searching",
            "save_user_fact": "learning",
            "read_odt_placeholders": "reading",
            "modify_odt_file": "writing"
        }
        status_msg = status_map.get(name, "executing")
        
        # Extract arguments for localization
        status_args = {}
        if name == "list_omd_files":
             status_args["path"] = os.path.basename(args.get('path', 'directory').rstrip('/'))
        elif name == "read_omd_file":
             status_args["path"] = os.path.basename(args.get('path', 'file'))
        elif name == "read_odt_placeholders":
             status_args["path"] = os.path.basename(args.get('path', 'file'))
        elif name == "modify_odt_file":
             status_args["path"] = os.path.basename(args.get('output_path', 'file'))
        elif name == "find_omd_file":
             status_args["path"] = os.path.basename(args.get('root_directory', 'root').rstrip('/'))
        elif name == "write_omd_file":
             status_args["path"] = os.path.basename(args.get('path', 'file'))
        elif name == "search_web":
             status_args["path"] = args.get('query', '')

        # Yield status event
        yield {"type": "status", "content": status_msg, "args": status_args}

        # Update checklist todos
        if todos:
             matched = False
             for t_item in todos:
                 if t_item["status"] == "pending" and (name in t_item["content"] or status_msg in t_item["content"].lower()):
                     t_item["status"] = "completed"
                     matched = True
                     break
             if not matched:
                 for t_item in todos:
                     if t_item["status"] == "pending":
                         t_item["status"] = "completed"
                         break
             yield {
                 "id": "mcp_todo",
                 "type": "tool",
                 "state": {
                     "status": "running",
                     "tool": "todo",
                     "input": {
                         "todos": todos
                     }
                 },
                 "action": "part_update"
             }

        # Yield tool start
        yield {
            "id": f"tool_turn_{turn}",
            "type": "tool",
            "state": {
                "status": "running",
                "tool": name,
                "input": args
            },
            "action": "part_update"
        }

        # [DEBUG OUTPUT]
        # User requested to see what is going on with tool calls
        debug_info = f"[MCP][Turn {turn+1}] {name}({args})"
        logging.info(f"--- TOOL CALL: {debug_info}")
        # Also print to stdout for visibility in the terminal
        print(f"\n>>> TURN {turn+1} TOOL CALL: {debug_info}\n")
        
        # [REPETITION BLOCK]
        # Hash the call to check for duplication in this session
        call_id_str = f"{name}:{json.dumps(args, sort_keys=True)}"
        
        if call_id_str in call_history:
             # Model is looping. Inject a forceful mechanical block.
             candidate_paths = sorted(list(known_files))
             res = f"SYSTEM ERROR: You already called {name} with these arguments. DO NOT REPEAT. If you need data, call `read_omd_file` on one of these verified paths: {candidate_paths[:5]}"
             logging.warning(f"[MCP] Blocked repeating tool call: {call_id_str}")
        else:
             call_history.add(call_id_str)
             res = ""
             
             # [NUCLEAR PHASE 4: SEARCH WEB LOOP PREVENTION]
             if name == "search_web":
                  search_count = sum(1 for c in list(call_history) if c.startswith("search_web:"))
                  if search_count >= 2:
                       res = "SYSTEM ERROR: You have searched the web multiple times. Do NOT continue searching. If you cannot find the answer, explain what you found or state that information is missing. DO NOT CALL SEARCH AGAIN."
                       logging.warning("[MCP] Blocked secondary web search to prevent loop.")
             
             if not res and name == "list_omd_files":
                  path_arg = (args.get("path") or args.get("file_path") or args.get("filePath") or "").strip()
                  
                  # [LOOP PREVENTION]
                  # Prevent recursive directory path loops (e.g. /Templates/Templates/Templates)
                  if re.search(r"/([^/]+)/\1/\1", path_arg) or re.search(r"^([^/]+)/\1/\1", path_arg):
                       res = f"ERROR: Recursion loop detected in path '{path_arg}'. Please list a different, non-repeating directory."
                       logging.warning(f"[MCP PATH LOOP] Blocked recursive path: {path_arg}")
                  elif path_arg.rstrip("/") in listed_paths:
                       res = f"Note: You already listed '{path_arg}'. Use the information you already have or list a DIFFERENT directory."
                       logging.info(f"[MCP] Blocked redundant list for {path_arg}")
                  else:
                       # [TURN LIMIT PROTECTION]
                       if turn >= max_turns - 2 and not has_read_file:
                            res = "CRITICAL ERROR: Operation limit reached. You have not successfully completed the data extraction. Please report what you found and state what is missing."
                            logging.error(f"[MCP] Autonomous Agent Turn Limit: {turn+1}")
                       else:
                            res = await list_omd_files(ctx, path_arg)
                            if not res.startswith("Error"):
                                 listed_paths.add(path_arg.rstrip("/"))
                                 last_listing = res # Cache for recovery
                                 
                                 # Populate known_files to prevent hallucinations
                                 # Simple filename extractor from list output
                                 # Extract absolute paths from format: "- [FILE] [ABS_PATH]: /path/to/file"
                                 path_matches = re.findall(r'\[ABS_PATH\]: ([^\s\n|]+)', res)
                                 for f in path_matches:
                                      known_files.add(f.strip())
                                 
                                 # Also track sub-directories
                                 dir_matches = re.findall(r'- \[DIRECTORY\] \[ABS_PATH\]: ([^\s\n|]+)', res)
                                 for d in dir_matches:
                                      listed_paths.add(d.strip())
                                      
                                 # [SYSTEM REDIRECT] 
                                 # Inject a mandatory manifest into the tool result to prevent logic errors
                                 res += f"\n\nSYSTEM NOTICE: You MUST use one of these [ABS_PATH] values exactly for your next tool call. Forbidden: guessing, relative paths, or reading directories."
             
             elif name == "read_omd_file":
                 path_arg = (args.get("path") or args.get("template_path") or args.get("template") or args.get("templatePath") or args.get("file_path") or args.get("filePath") or "").strip()
                 
                 # [ANTI-HALLUCINATION] List before Read
                 # We encourage the model to list first, but if it knows the file, we check our cache
                 path_dir = path_arg.rstrip("/").rsplit("/", 1)[0]
                 
                 # [DIRECTORY BLOCK] - Prevent reading paths confirmed as directories
                 if path_arg.rstrip("/") in listed_paths:
                      # [PATH MISMATCH NUDGE]
                      # Did the agent mention a file in its reasoning but call read on the directory?
                      detected_file = ""
                      for kf in known_files:
                           if os.path.basename(kf) in agent_text or kf in agent_text:
                                detected_file = kf
                                break
                      
                      res = f"ERROR: '{path_arg}' is a DIRECTORY. You MUST use the [ABS_PATH] of a FILE from the listing instead."
                      if detected_file:
                           res = f"CRITICAL MISMATCH: You correctly identified '{detected_file}' in your reasoning, but wrongly called read_omd_file on the DIRECTORY '{path_arg}'.\n\nRETRY NOW: Call `read_omd_file` using the absolute path '{detected_file}' WITHOUT any modification."
                      
                      # Re-inject the listing to help it recover
                      if last_listing:
                           res += f"\n\n--- REFRESHED LISTING ---\n{last_listing}"
                      
                      logging.warning(f"[MCP] Blocked directory read and injected nudge/listing: {path_arg}")
                 elif path_dir and path_dir.rstrip("/") in listed_paths and path_arg.rstrip("/") not in known_files:
                      res = f"ERROR: Path '{path_arg}' not found in current manifest. Reference the [ABS_PATH] value EXACTLY as provided by the tool."
                      logging.warning(f"[MCP] Blocked hallucinated/corrupted read: {path_arg}")
                 else:
                      res = await read_omd_file(ctx, path_arg)
                      if not res.startswith("Error"):
                           has_read_file = True
                      elif "DIRECTORY" in res:
                           # If it failed because it was a directory, we need to mark it
                           listed_paths.add(path_arg.rstrip("/"))
             elif name == "find_omd_file":
                  root_dir = args.get("root_directory", "").strip()
                  cond = args.get("condition", "").strip()
                  res = await find_omd_file(ctx, root_dir, cond)
                  if res.startswith("[FILE]:"):
                       # Extract only the path part, ignoring any SYSTEM NOTICE that might be appended
                       raw_res = res.split("\n\nSYSTEM NOTICE:")[0]
                       file_path = raw_res.replace("[FILE]:", "").strip()
                       known_files.add(file_path)
                       # Inject nudge
                       
             elif name == "write_omd_file":
                 # [TURN 1 SHIELD]
                 # Prevent early writes if source paths are mentioned but not yet processed.
                 if turn == 0 and potential_paths:
                      res = "Error: It is FORBIDDEN to write on Turn 1 when source paths are mentioned. You MUST use list_omd_files or read_omd_file first to gather data and avoid hallucination."
                      logging.warning(f"[MCP] Turn 1 Write Blocked: {args.get('path')}")
                 else:
                     content = args.get("content", "")
                     path_arg = (args.get("path") or args.get("file_path") or args.get("filePath") or "").strip()
                     
                     # [ROBUST WRITE SHIELD]
                     # If the content looks like an error, placeholder, or failure report, reject it.
                     forbidden_patterns = ["[", "...", "Failed to fetch", "Error reading", "No information", "Access denied", "403", "DIRECTORY", "cannot read"]
                     is_invalid = any(pattern.lower() in content.lower() for pattern in forbidden_patterns)
                     
                     # [DIRECTORY WRITE BLOCK]
                     is_dir_write = path_arg.rstrip("/") in listed_paths
                     
                     if is_invalid:
                          res = "Error: Do NOT use `write_omd_file` to save error messages, placeholders, or reports of failure. You must only write if you have successfully extracted the required data."
                     elif is_dir_write:
                          res = f"Error: '{path_arg}' is a confirmed DIRECTORY. You cannot write a file to this path. Use a full filename (e.g., {path_arg.rstrip('/')}/report.txt)."
                     else:
                          res = await write_omd_file(ctx, path_arg, content)
                          if not res.startswith("Error") and not res.startswith("Exception"):
                               changed_files.append(path_arg)
             elif name == "search_web":
                 res = await search_web(ctx, args.get("query", ""))
             elif name == "search_memory":
                 # [TURN 1 SEARCH SHIELD]
                 # Prevent using memory search on Turn 1 if user provided an explicit file path.
                 if turn == 0 and potential_paths:
                      res = "Error: A file path was provided. You MUST use list_omd_files or read_omd_file first. Do NOT use search_memory if you have a path to explore."
                      logging.warning(f"[MCP] Turn 1 Search Blocked (Path present)")
                 else:
                     query = args.get("query", "")
                     if query:
                         res = await search_memory_tool(ctx, query)
             elif name == "save_user_fact":
                 fact = args.get("fact", "")
                 if fact:
                     try:
                         logging.info(f"[MCP] Agent actively memorizing: {fact}")
                         memorize(ctx, fact)
                         res = f"Success. Memorized: {fact}"
                     except Exception as e:
                         res = f"Error saving memory: {e}"
             elif name == "read_odt_placeholders":
                  path_arg = (args.get("template_path") or args.get("odt_file_path") or args.get("path") or args.get("template") or args.get("templatePath") or args.get("file_path") or args.get("filePath") or "").strip()
                  if not path_arg:
                      res = "Error: Missing required parameter 'template_path'. You must specify the absolute path to the ODT template file (e.g., '/MyDevice/Documents/my_template.odt')."
                  else:
                      res = await read_odt_placeholders(ctx, path_arg)
             elif name == "modify_odt_file":
                    template_path = (args.get("template_path") or args.get("path") or args.get("template_file_path") or args.get("templatePath") or args.get("template") or "").strip()
                    
                    # Proactive template path recovery/healing: find verified path from previous read_odt_placeholders call
                    discovered_template = ""
                    for call in list(call_history):
                        if "read_odt_placeholders" in call:
                            try:
                                args_str = call.split(":", 1)[1]
                                args_parsed = json.loads(args_str)
                                path_val = (args_parsed.get("template_path") or args_parsed.get("path") or "").strip()
                                if path_val:
                                    discovered_template = path_val
                                    break
                            except:
                                pass
                                 
                    if discovered_template and (not template_path or template_path != discovered_template or "template.odt" in template_path.lower()):
                        logging.info(f"[MCP HEALER] Auto-corrected template_path from '{template_path}' to verified path '{discovered_template}'")
                        template_path = discovered_template
                         
                    if not template_path and potential_paths:
                        odt_files = [p for p in potential_paths if p.lower().endswith(".odt")]
                        if odt_files:
                            template_path = odt_files[0]
                            logging.info(f"[MCP HEALER] Restored missing template_path from potential prompt paths: '{template_path}'")
                    
                    # --- Fail-fast: required parameter checks before any path validation ---
                    storage_root = f"/{ctx.storage.strip('/').split('/')[0]}" if ctx.storage else "your active storage root"
                    if not template_path:
                        res = (f"Error: Missing required parameter 'template_path' in modify_odt_file call. "
                               f"You must supply the absolute path to the source ODT template "
                               f"(e.g. modify_odt_file(template_path='{storage_root}/Documents/Bineon_template.odt', "
                               f"output_path='{storage_root}/Documents/result.odt', replacements={{...}})). "
                               f"Use the same path you passed to read_odt_placeholders.")
                    else:
                        output_path = (args.get("output_path") or args.get("output_file_path") or args.get("outputPath") or args.get("output") or "").strip()
                        if not output_path:
                            res = (f"Error: Missing required parameter 'output_path' in modify_odt_file call. "
                                   f"You must supply the absolute path for the resulting ODT file "
                                   f"(e.g. output_path='{storage_root}/Documents/result.odt'). "
                                   f"Must include a filename ending in .odt.")
                        else:
                            replacements = args.get("replacements")
                            if not isinstance(replacements, dict):
                                replacements = {}
                                 
                            # Proactive ODT replacements healer: catch flattened custom arguments and move them to replacements
                            excluded_keys = {
                                "template_path", "path", "template_file_path", "templatepath", "template",
                                "output_path", "output_file_path", "outputpath", "output", "replacements"
                            }
                            for k, v in args.items():
                                if k.lower() not in excluded_keys and v is not None:
                                    # Clean the key to map output_content -> content, output_title -> title
                                    clean_k = k
                                    if k.lower().startswith("output_"):
                                        clean_k = k[7:]
                                    elif k.lower().startswith("output"):
                                        clean_k = k[6:]
                                    replacements[clean_k] = v
                                    
                            if replacements and not args.get("replacements"):
                                logging.info(f"[MCP HEALER] Auto-reconstructed replacements from flattened top-level arguments: {replacements}")
                                
                            res = await modify_odt_file(ctx, template_path, output_path, replacements)
                            if not res.startswith("Error") and not res.startswith("Exception"):
                                changed_files.append(output_path)
             else:
                   supported_names = [t["function"]["name"] for t in MCP_TOOLS]
                   res = f"Error: Tool '{name}' is not supported. Supported tools are: {supported_names}. Please only call supported tools."
                   logging.warning(f"[MCP HALLUCINATION GUARD] Blocked hallucinated tool: {name}")

        if res:
            all_tool_results += f"Tool Output ({name}):\n{res}\n\n"
            msg_entry = {"role": "tool", "content": str(res)}
            if call_id:
                 msg_entry["tool_call_id"] = call_id
            messages.append(msg_entry)
            
            # [RESULT DEBUG OUTPUT]
            logging.info(f"--- TOOL RESULT ({name}): {str(res)[:100]}...")
            print(f">>> TURN {turn+1} TOOL RESULT: {str(res)[:200]}...\n")
            
            found_new_info = True
        else:
            msg_entry = {"role": "tool", "content": "Error: Tool returned no result."}
            if call_id:
                 msg_entry["tool_call_id"] = call_id
            messages.append(msg_entry)

        # Yield completion of the tool call part
        res_str = str(res).strip() if res else ""
        is_err = (
            not res_str or
            res_str.lower().startswith(("error", "exception", "failed")) or
            "security error" in res_str.lower() or
            "critical error" in res_str.lower() or
            res_str.startswith("CRITICAL MISMATCH")
        )
        tool_status = "error" if is_err else "success"
        clean_output = res_str if res_str else "Error: Tool returned no result."
        if len(clean_output) > 2000:
            clean_output = clean_output[:2000] + "\n... [Output truncated for length] ..."
            
        yield {
            "id": f"tool_turn_{turn}",
            "type": "tool",
            "state": {
                "status": tool_status,
                "success": (tool_status == "success"),
                "tool": name,
                "input": args,
                "output": clean_output
            },
            "success": (tool_status == "success"),
            "action": "part_update"
        }

        if tool_status == "error":
             success = False
             tool_error_msg = clean_output
             break

        if not found_new_info:
            break
            
        # [TURN RESET]
        # We break the tool list loop here (we only used one tool) and allow the 
        # main 'turn' loop to call the model again with the new knowledge.

    # Hallucination lockdown is now handled by the planning requirement and instructions.
    
    # Log final MCP conclusion
    last_content = messages[-1].get("content", "").strip() if messages and messages[-1].get("role") == "assistant" else ""
    if last_content:
        logging.info(f"[MCP] Agent Conclusion: {last_content[:100]}...")
    
    # If the autonomous agent didn't produce any meaningful output, report failure
    if not all_tool_results.strip():
        all_tool_results = "Tool Output: No files were listed or read. The discovery phase returned no data."
        logging.warning("[MCP] Autonomous loop finished with ZERO results.")
    elif not has_read_file and "find_omd_file" in all_tool_results:
        # Check if we were supposed to read but didn't
        if requires_read:
             all_tool_results += "\nSYSTEM NOTICE: Discovery succeeded, but no file content was read. The requested details (totals/items) are NOT in this tool output."
        
    # Mark all remaining todos completed when finished
    if todos:
         for t_item in todos:
              t_item["status"] = "completed"
         yield {
             "id": "mcp_todo",
             "type": "tool",
             "state": {
                 "status": "success",
                 "tool": "todo",
                 "input": {
                     "todos": todos
                 }
             },
             "action": "part_update"
         }
         
    # Strict validation: if document generation was planned or requested, but no files were actually modified, mark as failure.
    was_write_planned = any("modify_odt_file" in str(t.get("content", "")) or "write_omd_file" in str(t.get("content", "")) for t in todos) if todos else False
    is_doc_cmd = clean_message.strip().startswith("/doc") or "modify_odt_file" in all_tool_results
    if (was_write_planned or is_doc_cmd) and not changed_files and success:
        success = False
        tool_error_msg = "The document was not modified or saved. Please ensure all required parameters are provided."
         
    # Always generate the final human response dynamically via the LLM based on actual execution results,
    # ensuring zero hardcoded templates or concrete conversational Russian phrases are baked into the Python code.
    error_context = f"\nCRITICAL: The operation failed with the following error:\n- {tool_error_msg}" if not success else ""
    files_context = f"\nModified/Created files saved in OnMyDisk storage: {', '.join(changed_files)}" if changed_files else ""
    
    summary_prompt = (
        "You are a helpful file system assistant. The requested file system operations have finished.\n"
        f"User's request was: '{clean_message}'\n"
        f"Execution status: {'SUCCESS' if success else 'FAILED'}\n"
        f"{error_context}\n"
        f"{files_context}\n\n"
        "Write a short, friendly, and natural conversational response in Russian.\n"
        "INSTRUCTIONS:\n"
        "1. If the execution failed (FAILED status), you MUST clearly, politely, and explicitly explain the error to the user in Russian, state exactly what went wrong, and ask how they want to proceed. Do NOT claim success under any circumstances!\n"
        "2. If everything was successful (SUCCESS status), summarize the actions you performed (in detail) and tell the user that everything is successfully saved in their storage.\n"
        "3. NEVER use generic templates or hardcoded phrases. Make the response highly context-specific, warm, and professional."
    )
    payload = {
        "model": MCP_MODEL,
        "messages": [
            {"role": "system", "content": f"{BASE_SYSTEM_PROMPT}\n\n{summary_prompt}"}
        ],
        "stream": False,
        "options": {"temperature": 0.5}
    }
    logging.info("[MCP] Generating dynamic human response via LLM...")
    final_output = ""
    summary_data = await llm_request(payload)
    if summary_data and "message" in summary_data:
        final_output = summary_data["message"]["content"].strip()
        
    # Safe technical fallback (strictly minimal and technical, no hardcoded conversational Russian text!)
    if not final_output:
        if not success:
            final_output = f"Error: {tool_error_msg}"
        elif changed_files:
            final_output = f"Success. Modified files: {changed_files}"
        else:
            final_output = "Completed successfully."

    # Build changed files objects
    changed_objects = []
    for path in changed_files:
        filename = os.path.basename(path)
        mimetype = clean_mimetype(filename)
        
        changed_objects.append({
            "title": filename,
            "owner": ctx.user_id,
            "clickable": True,
            "url": f"https://onmydisk.net{path}",
            "fullPath": path,
            "size": 0,
            "mimetype": mimetype
        })

    yield {
        "type": "result",
        "content": final_output,
        "changedFiles": changed_objects
    }

# === Онбординг успешен - перенос песрональных данных ===
def bind_account(ctx: UserContext, omd_key: str):
    #check if already linked
    #renaming user data folder 
    if omd_key and not ctx.type == "omd" :
        tmp_user_id = ctx.user_id
        old_dir = os.path.join(USER_DATA_DIR, tmp_user_id)
        ctx = user_context.bind(ctx, omd_key)
        new_dir = os.path.join(USER_DATA_DIR, ctx.user_id)

        if os.path.exists(old_dir):
            # если у нового юзера ещё нет директории — просто переименуем
            if not os.path.exists(new_dir):
                os.rename(old_dir, new_dir)
            else:
                # если у нового юзера уже есть данные — можно смержить
                # пока просто оставим старое и не трогаем
                logging.warning(f"[bind_account] WARNING: {new_dir} already exists, skipping rename")

    # === Проверяем storage и переносим данные ===
    storage = ctx.storage
    omd_key = ctx.omd_key
    if ctx.type == "omd" and storage and omd_key:

        # 1. Переносим чаты
        chats_dir = os.path.join(USER_DATA_DIR, ctx.user_id, "chats")
        if os.path.exists(chats_dir):
            for file in os.listdir(chats_dir):
                if not file.endswith(".json"):
                    continue
                local_path = os.path.join(chats_dir, file)
                try:
                    dest = f"{storage}/{ctx.user_id}/chats"
                    upload_to_storage(omd_key, dest, file, local_path)
                    logging.info(f"[bind_account] Chat {file} uploaded to {dest}")
                except Exception as e:
                    logging.error(f"[bind_account] Failed to upload chat {file}: {e}")

        # 2. Персональная память
        mem_path = os.path.join(USER_DATA_DIR, ctx.user_id,  "memory.jsonl")
        if os.path.exists(mem_path):
            try:
                dest = f"{storage}/{ctx.user_id}"
                upload_to_storage(omd_key, dest,"memory.jsonl", mem_path)
                logging.info(f"[bind_account] Memory uploaded to {dest}")
            except Exception as e:
                logging.error(f"[bind_account] Failed to upload memory: {e}")

    return ctx


    

def get_assistant_avatar_path(ctx: UserContext) -> str:
    model = ctx.settings.get("assistant_model", "Domi")
    full_avatar_path = f"{COMFY_INPUT_DIR}/{AVATAR_DIR}/{model}.png"
    avatar_path = full_avatar_path.replace(STORAGE_ROOT, "")
    return avatar_path

def get_model_avatar_path(model_name: str) -> str:
    full_avatar_path = f"{COMFY_INPUT_DIR}/{AVATAR_DIR}/{model_name}.png"
    avatar_path = full_avatar_path.replace(STORAGE_ROOT, "")
    return avatar_path


def get_available_loras(ctx: UserContext = None, nsfw: bool | None = None) -> list:
    lora_file = os.path.join(os.path.dirname(__file__), "analog_character_lora.json")
    if os.path.exists(lora_file):
        try:
            with open(lora_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Find Power Lora Loader node (ID 103 based on file inspection)
            # Or search for class_type "Power Lora Loader (rgthree)"
            loras = []
            for key, node in data.items():
                if node.get("class_type") == "Power Lora Loader (rgthree)":
                    inputs = node.get("inputs", {})
                    for input_key, input_val in inputs.items():
                        if input_key.startswith("lora_") and isinstance(input_val, dict):
                            if "name" in input_val:
                                # Prioritize explicit nsfw flag, then fallback to ctx
                                if nsfw is not None:
                                    user_nsfw = nsfw
                                elif ctx:
                                    user_nsfw = ctx.settings.get("nsfw", False)
                                else:
                                    user_nsfw = False
                                    
                                # Filter based on NSFW setting
                                lora_nsfw = input_val.get("nsfw", False)
                                # Skip NSFW loras if user has NSFW disabled
                                if lora_nsfw and not user_nsfw:
                                    continue
                                
                                loras.append({
                                    "name": input_val["name"],
                                    "type": input_val.get("type", "character")
                                })
            
            # Sort by name
            loras.sort(key=lambda x: x["name"])
            return loras

        except Exception as e:
            logging.error(f"Error reading LoRA file: {e}")
            return []
    return []


async def get_avatar_version(ctx: UserContext) -> str:
    # 1. Check remote
    if ctx.storage and ctx.omd_key:
        try:
            storage_id = ctx.storage
            storage_key = ctx.omd_key
            
            base_url = GATEWAY_URL.rstrip("/")
            clean_storage_id = storage_id.strip("/")
            # Manual URL construction to ensure token is passed correctly
            # Use GET instead of HEAD as HEAD might be blocked or malformed for this gateway
            timestamp = str(int(time.time()))
            url = f"{base_url}/{clean_storage_id}/avatar.png?token={storage_key}&_t={timestamp}"
            
            #logging.info(f"[get_avatar_version] Checking remote (GET): {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    #logging.info(f"[get_avatar_version] Response status: {resp.status}")
                    if resp.status == 200:
                        # Use ETag or Last-Modified
                        etag = resp.headers.get("ETag")
                        last_modified = resp.headers.get("Last-Modified")
                        #logging.info(f"[get_avatar_version] ETag: {etag}, Last-Modified: {last_modified}")
                        if etag:
                            return etag.strip('"')
                        if last_modified:
                            # Simple hash of last modified string
                            return str(hash(last_modified))
        except Exception as e:
            logging.warning(f"[get_avatar_version] Failed to get remote avatar version: {e}")

    # 2. Fallback to local
    try:
        storage_path = get_assistant_avatar_path(ctx)
        if storage_path.startswith("/"):
            storage_path = storage_path[1:]
        
        full_path = os.path.join(STORAGE_ROOT, storage_path)
        if os.path.exists(full_path):
            mtime = os.path.getmtime(full_path)
            #logging.info(f"[get_avatar_version] Local avatar local version: {mtime}")

            return str(int(mtime))
    except Exception as e:
        logging.warning(f"Failed to get local avatar version: {e}")
        return str(int(time.time()))

async def get_generated_avatars(ctx: UserContext) -> list:
    avatars = []
    # 1. Remote
    if ctx.storage and ctx.omd_key:
        try:
            storage_id = ctx.storage
            storage_key = ctx.omd_key
            
            base_url = GATEWAY_URL.rstrip("/")
            clean_storage_id = storage_id.strip("/")
            
            # List generated/avatars folder
            url = f"{base_url}/{clean_storage_id}/generated/avatars?list&token={storage_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200 or resp.status == 206:
                        try:
                            data = await resp.json()
                        except Exception as e:
                             text = await resp.text()
                             try:
                                 data = json.loads(text)
                             except json.JSONDecodeError as je:
                                 if "Extra data" in str(je):
                                     data = json.loads(text[:je.pos])
                                 else:
                                     raise je
                        
                        items = []
                        if "list" in data:
                            items = data["list"]
                        elif "result" in data:
                            items = data["result"]
                            
                        for item in items:
                            name = item.get("name", "")
                            item_type = item.get("type", "")
                            
                            if item_type != "dir" and name.startswith("avatar"):
                                    # Construct direct URL
                                    file_url = f"{base_url}/{clean_storage_id}/generated/{name}?token={storage_key}"
                                    avatars.append({
                                        "name": name,
                                        "url": file_url,
                                        "date": item.get("date"),
                                        "size": item.get("size")
                                    })
            return avatars
        except Exception as e:
            logging.warning(f"Failed to list remote avatars: {e}")
            # Fallthrough to local if needed or just return empty?
            # If storage is configured, we assume remote is source of truth.
            return []

    # 2. Local fallback
    try:
        # Assuming generated is adjacent to avatar.png or in user root?
        # Standard: STORAGE_ROOT / user_id / generated
        # Need to verify path. core_service has upload_to_storage logic.
        # It uses 'generated/' as path.
        # We need absolute path.
        # upload_to_storage uses: os.path.join(STORAGE_ROOT, user_id, path)
        
        user_storage_path = os.path.join(STORAGE_ROOT, ctx.user_id)
        if ctx.storage and not ctx.omd_key: # Local storage mount
             # Logic for local mounts is complex, assume standard structure for now
             user_storage_path = os.path.join(STORAGE_ROOT, ctx.storage.strip("/"))
        
        generated_dir = os.path.join(user_storage_path, "generated", "avatars")
        
        if os.path.exists(generated_dir):
            for filename in os.listdir(generated_dir):
                if filename.startswith("avatar"):
                    full_path = os.path.join(generated_dir, filename)
                    if os.path.isfile(full_path):
                         # Simple local URL? Or we need api to serve it?
                         # For now, return filename. Frontend likely needs full URL.
                         # But if local, we might need an endpoint to serve it.
                         # Since remote is priority, basic implementation:
                         avatars.append({
                             "name": filename,
                             "url": "", # Frontend might fallback
                             "date": os.path.getmtime(full_path)
                         })
        
        # Sort local by date?
        avatars.sort(key=lambda x: x["date"], reverse=True)
        return avatars

    except Exception as e:
        logging.warning(f"Failed to list local avatars: {e}")
        return []


async def generate_image_workflow(workflow) -> bytes:
    client_id = str(uuid.uuid4())
    ws_url = f"{COMFY_API_URL.replace('http', 'ws')}/ws?clientId={client_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Send prompt
                payload = {
                    "prompt": workflow,
                    "client_id": client_id
                }
                async with session.post(f"{COMFY_API_URL}/prompt", json=payload) as resp:
                    resp_data = await resp.json()
                    prompt_id = resp_data.get("prompt_id")
                    logging.info(f"Prompt ID: {prompt_id}")

                # Listen for messages
                final_image_data = None
                final_filename = None
                
                # Timeout safety
                loop_start = time.time()

                async for msg in ws:
                    if time.time() - loop_start > 120:
                        logging.error("Timeout waiting for image generation")
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "executing":
                            data_content = data["data"]
                            # Log heartbeat or node executing
                            if data_content["node"] is None and data_content["prompt_id"] == prompt_id:
                                logging.info(f"Execution finished for prompt {prompt_id}")
                                # Determine if we should break or wait for pending fetches
                                # If we have an image, great. If not, maybe we missed it or it is coming?
                                # Usually 'executed' comes before 'executing' (finished).
                                break
                                
                        elif msg_type == "executed":
                            data_content = data["data"]
                            if data_content["prompt_id"] == prompt_id:
                                outputs = data_content.get("output", {})
                                #logging.info(f"Outputs received: {outputs.keys()}")
                                
                                # outputs is directly the dictionary of outputs for the executed node
                                if "images" in outputs:
                                    images = outputs["images"]
                                    #logging.info(f"Images found: {images}")
                                    for image in images:
                                        filename = image.get("filename")
                                        subfolder = image.get("subfolder", "")
                                        img_type = image.get("type", "output")
                                        
                                        #logging.info(f"Fetching image: {filename} [{img_type}]")
                                        
                                        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
                                        async with session.get(f"{COMFY_API_URL}/view", params=params) as img_resp:
                                            if img_resp.status == 200:
                                                final_image_data = await img_resp.read()
                                                final_filename = filename
                                                logging.info(f"Image fetched: {len(final_image_data)} bytes")
                                            else:
                                                logging.error(f"Failed to fetch image: {img_resp.status}")
                        
                        # Log other messages just in case
                        # else:
                        #     logging.debug(f"WS Message: {msg_type}")

                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # Preview image handling
                        # First 4 bytes: integer type (1 = JPEG preview)
                        if len(msg.data) > 4:
                            event_type = int.from_bytes(msg.data[:4], 'big')
                            if event_type == 1:
                                logging.info("Received binary preview image")
                                # Only use binary preview if we don't have a high-res one yet, or as fallback
                                if not final_image_data:
                                    # Skip first 8 bytes? ComfyUI source:
                                    # const view_metadata = new DataView(event.data.slice(0, 8)); ... 
                                    # Actually python int.from_bytes is safe. 
                                    # Standard preview is JPEG.
                                    final_image_data = msg.data[8:] 
                                    final_filename = f"preview_{uuid.uuid4()}.jpg"
                                    logging.info(f"Captured preview image: {len(final_image_data)} bytes")
                            else:
                                logging.info(f"Received binary event type: {event_type}")

                if not final_image_data:
                     logging.warning("No image data captured during execution.")

                return final_image_data, final_filename

    except Exception as e:
        logging.error(f"Error in generate_image_workflow: {e}")
        return None, None

    except Exception as e:
        logging.error(f"Error in generate_image_workflow: {e}")



# === RAG ===
def get_file_icon(filename: str) -> str:
    ext = filename.split(".")[-1].lower() if "." in filename else ""
    icon_map = {
        "pdf": "file-pdf",
        "doc": "file-word", "docx": "file-word", "odt": "file-word",
        "xls": "file-excel", "xlsx": "file-excel", "ods": "file-excel",
        "ppt": "file-powerpoint", "pptx": "file-powerpoint",
        "jpg": "file-image", "jpeg": "file-image", "png": "file-image", "gif": "file-image", "svg": "file-image",
        "mp4": "file-video", "mkv": "file-video", "avi": "file-video",
        "mp3": "file-audio", "wav": "file-audio",
        "zip": "file-archive", "rar": "file-archive", "7z": "file-archive", "tar": "file-archive", "gz": "file-archive",
        "md": "file-markdown", "txt": "file-text",
        "html": "file-code", "js": "file-code", "css": "file-code", "py": "file-code", "cpp": "file-code", "h": "file-code"
    }
    return icon_map.get(ext, "file")

async def get_source_metadata(ctx: UserContext, owner: str, path: str) -> dict:
    """Запрашивает метаданные источника с учетом прав доступа."""
    import aiohttp
    gateway_url = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net").rstrip("/")
    item_path = path.lstrip("/")
    
    CRAWLER_TOKEN = "mypears-4204"
    
    metadata = {
        "mimetype": None,
        "url": None,
        "clickable": False,
        "fullPath": f"/{owner}/{item_path}" # Default full path
    }
    
    async with aiohttp.ClientSession() as session:
        # 1. Пробуем получить атрибуты с ключом текущего пользователя (User Mode)
        if ctx.omd_key:
            user_headers = {
                "Authorization": f"token:{ctx.omd_key}",
                "Content-Type": "application/json"
            }
            try:
                user_target_url = f"{gateway_url}/{item_path}"
                full_item_path = "/" + item_path
                attr_payload = {"action": "getAttributes", "item": full_item_path}
                async with session.post(user_target_url, json=attr_payload, headers=user_headers, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        result = data.get("result", {})
                        success = data.get("success") or result.get("success")
                        mimetype = result.get("mimetype") or data.get("mimetype")
                        
                        if success and mimetype:
                            metadata["mimetype"] = clean_mimetype(item_path, mimetype)
                            metadata["clickable"] = True
                            return metadata
            except Exception as e:
                logging.error(f"User access check failed for {path}: {e}")
        
        # 2. Если не удалось или нет ключа, пробуем через мастер-токен (Master Mode)
        master_target_url = f"{gateway_url}/{owner}/{item_path}"
        master_token = f"{CRAWLER_TOKEN}:{owner}"
        master_headers = {
            "Authorization": f"token:{master_token}",
            "Content-Type": "application/json"
        }
        try:
            full_item_path = "/" + item_path
            attr_payload = {"action": "getAttributes", "item": full_item_path}
            async with session.post(master_target_url, json=attr_payload, headers=master_headers, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    result = data.get("result", {})
                    success = data.get("success") or result.get("success")
                    mimetype = result.get("mimetype") or data.get("mimetype")
                    
                    if success and mimetype:
                        metadata["mimetype"] = clean_mimetype(item_path, mimetype)
                        metadata["clickable"] = True
                        return metadata
        except Exception as e:
            logging.error(f"Master access check failed for {path}: {e}")

    return metadata

async def inject_facts(ctx: UserContext, query: str, collection: str = "", mem_id="", provided_knowledge: list|None = None, skip_db: bool = False) -> tuple[list[str], list[dict]]:
    logging.info(f"[memory] inject_facts for user_id: {ctx.user_id}")
    facts = []
    document_ids = []
    
    # Load RAG_TOP_K from SETTINGS
    rag_top_k = int(SETTINGS.get("RAG_TOP_K", "5"))

    # Force skip_db if NSFW is enabled to exclude working knowledge base search and only use frontend-provided facts
    if ctx.settings.get("nsfw", False):
        logging.info("[memory] NSFW mode enabled: skipping working database and indexed files search, using provided knowledge only.")
        skip_db = True

    # Личные воспоминания (только если предоставлены фронтендом)
    if provided_knowledge is not None:
        # If knowledge is provided via frontend, it fully replaces EXISTING personal memory access.
        for m in provided_knowledge:
            text = m.get("text", "") if isinstance(m, dict) else str(m)
            if text:
                facts.append(f"• {text}")
                doc_id = m.get("document_id") if isinstance(m, dict) else None
                if doc_id:
                    document_ids.append(doc_id)

    # Общие знания — если есть collection
    sources_map = {} # To avoid duplicates
    if not skip_db:
        if collection:
            shared = search_memories(ctx, query, collection=collection, mem_id=mem_id, top_k=rag_top_k)
            for m in shared:
                facts.append(f"• {m['text']}")
                doc_id = m.get("document_id")
                owner = m.get("owner", "alexey")
                if doc_id:
                    key = f"{owner}:{doc_id}"
                    if key not in sources_map:
                        filename = doc_id.split("/")[-1]
                        sources_map[key] = {
                            "title": filename,
                            "owner": owner,
                            "clickable": False,
                        }

        # Личные проиндексированные файлы (omd_search)
        user_files = search_indexed_files(ctx, query, owner=ctx.user_id, top_k=rag_top_k)
        for f in user_files:
            facts.append(f"• [From file {f['title']}]: {f['text']}")
            key = f"file:{f['document_id']}"
            if key not in sources_map:
                sources_map[key] = {
                    "title": f['title'],
                    "owner": f['owner'],
                    "clickable": True,
                    "url": f"https://onmydisk.net{f['document_id']}",
                    "fullPath": f['document_id']
                }


    # Fetch metadata for all sources
    for key, source in sources_map.items():
        if not source.get("clickable"): # Already has metadata if clickable
             owner = source.get("owner", "alexey")
             doc_id = key.split(":", 1)[1] if ":" in key else key
             meta = await get_source_metadata(ctx, owner, doc_id)
             source.update(meta)

    return facts, list(sources_map.values())

# === Ollama запрос ===
async def llm_request_stream(payload: dict, headers: dict = None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data
                    except Exception as e:
                        logging.error(f"Stream parse error: {e}")
    except Exception as e:
        logging.error(f"LLM error: {e}")


async def llm_request(payload: dict, headers: dict = None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                if "application/json" in resp.headers.get("Content-Type", "").lower():
                    return await resp.json()
                else:
                    text = await resp.text()
                    logging.error(f"LLM error: Unexpected content type {resp.headers.get('Content-Type')}. Body: {text[:200]}")
                    return None
    except Exception as e:
        logging.error(f"LLM request exception: {e}")
        return None






# === Чат ===
async def _perform_prompt_gen(ctx: UserContext,
                         instruction: str,
                         message: str,
                         is_rag: bool=False,
                         chat: str = "default",
                         mem_id: str = None,
                         img_source: str = None,
                         stream: bool = False,
                         intent: str = "chat",
                         event: str = None,
                         provided_history: list = None,
                         provided_knowledge: list = None) -> AsyncGenerator:

    model = get_llm_model(ctx)
    nsfw_enabled = ctx.settings.get("nsfw", False)
    b64_image = None
    
    # Internal flags
    is_rag = intent in ["explain", "think"]

    # History is derived from provided_history or managed via frontend OrbitDB sync.
    if chat == "default":
        history = []
    
    history = provided_history or []
    
    system_prompt = ""
    # === ВСПОМНИМ ФАКТЫ ===
    strict_fact = ""
    facts_text = ""
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    logging.debug(f"Loading facts: {collection} {is_rag}")
    # === Facts injection ===
    if intent == "search":
        # Search results are already packed into 'instruction' inside api.py
        # Skipping duplicate Chroma DB search, but restoring sources from the cache
        facts = []
        sources = getattr(ctx, "temp_sources", [])
        if hasattr(ctx, "temp_sources"):
            try:
                del ctx.temp_sources
            except AttributeError:
                pass
    elif intent == "show":
        facts, sources = await inject_facts(ctx, message, collection, mem_id, provided_knowledge=provided_knowledge, skip_db=True)
    else:
        facts, sources = await inject_facts(ctx, message, collection, mem_id, provided_knowledge=provided_knowledge)
    
    # Yield sources immediately for the frontend widget
    if sources:
        yield {"sources": sources, "done": False}

    if facts:
        facts_text += "\n\n*Known facts:*\n" + "\n".join(facts)
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        logging.info(f"RAG request: {collection}")
        prep_prompt = (
            "You are a fact-checking assistant. Based on *Known facts* only, respond to the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt

        if facts_text:
            rag_system_prompt += facts_text

        prep_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": message}
        ]
        prep_payload = {
            "messages": prep_messages,
            "model": model,
            "stream": False,
            "options": {
               "temperature": 0.1,
            }
        }
        data = await llm_request(prep_payload)
        if not data:
            yield {"error": "⚠️ RAG query failed.", "done": True}
            return
        
        rag_resp = data["message"]["content"].strip()
        if rag_resp and not rag_resp.startswith("No information"):
            strict_fact = rag_resp

        # Инжект фактов и источников в system prompt
        if strict_fact:
            facts_text += f"\n\n*Strict facts:*\n{strict_fact}"
        
    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{BASE_SYSTEM_PROMPT}"
    else:  
        system_prompt = f"{BASE_SYSTEM_PROMPT}"


    # Check if the user is new or recurrent one, and prepare chat info
    chat_info = await ensure_chat(ctx, chat, message)
    chat_name = chat_info.get("name", chat or "default")

    # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
    chat_name = chat_info.get("name", chat or "default")

    # Персонализация
    username = ctx.settings.get("name") or ctx.settings.get("username", "User")
    if ctx.settings.get("newUser", False):
        system_prompt += user_context.DEFAULT_UNONBOARDED_PROMPT
        system_prompt += "\n\n*Attention*:* You are communicating with the new user!\n"
    else:        
        system_prompt += f"\n\n*Personality overrides*: {ctx.settings.get("system_prompt", user_context.DEFAULT_USER_PROMPT)}\n"
        system_prompt += f"\n\n*Attention*:* You are communicating with existing user. User name: {username}.\n"

    system_prompt += "\n\n*Appearance overrides:*\n" + ctx.settings.get("assistant_appearance", user_context.DEFAULT_ASSISTANT_APPEARANCE)
    logging.info(f"Model: {model}\nNSFW: {nsfw_enabled}\nUser: {username}")

    # Факты
    if facts_text:
        logging.info(facts_text)
        system_prompt += facts_text

    # Инструкция
    instruction_prompt = "*Instruction:*\n" + instruction

    user_lang = ctx.settings.get("language")
    if user_lang:
        instruction_prompt += f"\nPrefer user's language ({user_lang}) for generating responses."

    if not nsfw_enabled:
        instruction_prompt += "\nThe conversation is continuous; avoid redundancy. Use the provided tool results as the absolute source of truth."
        
        # [HALLUCINATION SHIELD]
        instruction_prompt += "\nCRITICAL: NEVER generate fake tool results. Do NOT use `tool_code` or `*System Tool Output:\n*` blocks yourself. Only the system provides tool results. If tools find nothing, state that it was not found. Do NOT invent content. Do NOT pretend to run actions like saving files by inventing system logs."

    if nsfw_enabled:
        instruction_prompt += "\n\n*Hint:*\nYou are allowed and welcome to respond in more relaxed, NSFW mode"

    # === Gemma 4 Thinking Mode ===
    is_simple_chat = not chat.startswith("/code/")
    thinking_intents = ["search", "explain", "think"]
    is_gemma4 = "gemma4" in model.lower() or "gemma-4" in model.lower()

    if is_gemma4:
        if is_simple_chat and intent in thinking_intents:
            logging.info(f"Enabling Gemma 4 Thinking Mode for intent: {intent}")
            # Ollama API handles the reasoning start logic automatically when 'think': True
        else:
            logging.info(f"Disabling Gemma 4 Thinking Mode for intent: {intent}")


    # === ОСНОВНОЙ ЗАПРОС ===
    system_prompt +=  f"\nCurrent local date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    messages = [{"role": "system", "content": system_prompt}] + history[-HISTORY_LIMIT:]

    # Добавляем новый запрос
    user_message = {
        "role": "user",
        "content": message,
    }

    instruction_message = {
        "role": "system",
        "content": instruction_prompt,
    }

    if img_source:
        b64_image = await get_image_from_source(ctx, img_source)    

    if b64_image:
        user_message["images"] = [b64_image]
        model = get_llm_model(ctx)
        if img_source.startswith("/"):
            user_message["image"] = {"path": img_source}

    if mem_id:
       user_message["mem_id"] = mem_id

    # logging.info(f"Starting main request:{model} {think} {img_source} {mem_id}")

    # Добавляем инструкцию
    messages.append(instruction_message)
    # Добавляем пользовательский промпт
    messages.append(user_message)

    main_payload = {
        "messages": messages,
        "model": model,
        "stream": stream,
        "options": {
            "temperature": 0.85,          # немного выше для разнообразия
            "top_p": 0.9,                 # ограничивает вероятность, убирая “хвост”
            "frequency_penalty": 0.6,     # штраф за частое повторение слов
            "presence_penalty": 0.5,      # штраф за повторение идей/тем
        }
    }

    if is_gemma4:
        if is_simple_chat and intent in thinking_intents:
            main_payload["think"] = True
        else:
            main_payload["think"] = False

    #post-processing of response
    async def process_response(data) -> dict: 
        logging.info(data)
        llm_response = data["message"]["content"]
        llm_response = clean_response(llm_response)
        llm_think_response = None
        if data["message"].get("thinking"):
            llm_think_response = data["message"]["thinking"]

        # result object
        response = {}
        response["content"] = llm_response.strip()
        if sources:  # добавляем блок только если есть источники
            response["sources"] = sources
        if strict_fact:    
            response["facts"] = strict_fact
        if llm_think_response:    
            response["thinking"] = llm_think_response
        # History entries are now managed via frontend OrbitDB sync.
        # We still return the assistant response for the client to process.
        if True: # Always process response, skip_history is redundant
            # Re-load history to get the latest (including the user message we just saved + any parallel ones)
            try:
                history = provided_history or []
            except Exception as e:
                logging.error(f"Failed to reload history before saving assistant response: {e}")
                # We can't yield here as we are in process_response, but we should at least not save if loading failed.
                # Actually, process_response is called from sync and async contexts.
                # Let's re-raise and handle in the caller.
                raise e

            history_entry = {
                "role": "assistant", 
                "content": llm_response
            }     
            if strict_fact:    
                history_entry["facts"] = strict_fact
            if sources:
                history_entry["sources"] = sources
            if llm_think_response:    
                history_entry["thinking"] = llm_think_response

            history.append(history_entry)

            # [LEGACY HISTORY] Save history removed - handled by frontend/OrbitDB
            pass
            
        response["chatinfo"] = chat_info
        return response


    if stream:
        async def gen():
            if strict_fact:
                yield {"facts": strict_fact, "done": False, "event": event}
            accumulated_response = ""
            accumulated_thinking = ""
            thinking = False
            logging.info(f"Requesting LLM {model}")
            async for data in llm_request_stream(main_payload):
                if data.get("done"):  
                    # финал: собираем response на основе всего текста
                    full_data = {
                        "message": {
                            "role": "assistant",
                            "content": accumulated_response
                        }
                    }
                    if accumulated_thinking:
                        full_data["message"]["thinking"] = accumulated_thinking
                    response = await process_response(full_data)
                    response["done"] = True
                    response["event"] = event
                    yield response
                elif data.get("message"):   
                    if data["message"].get("thinking"):
                        delta = data["message"]["thinking"]
                        accumulated_thinking += delta
                        yield {"thought_delta": delta, "done": False}
                        continue

                    delta = data["message"].get("content", "")
                    if not delta:
                        continue

                    # Gemma 4 Tag detection
                    OPEN_TAG = "<|channel>thought\n"
                    CLOSE_TAG = "<channel|>"

                    if OPEN_TAG in delta:
                        parts = delta.split(OPEN_TAG, 1)
                        if parts[0]:
                            yield {"delta": parts[0], "done": False}
                            accumulated_response += parts[0]
                        thinking = True
                        delta = parts[1]

                    if thinking and CLOSE_TAG in delta:
                        parts = delta.split(CLOSE_TAG, 1)
                        if parts[0]:
                            yield {"thought_delta": parts[0], "done": False}
                            accumulated_thinking += parts[0]
                        thinking = False
                        delta = parts[1]

                    if delta:
                        if thinking:
                            yield {"thought_delta": delta, "done": False}
                            accumulated_thinking += delta
                        else:
                            yield {"delta": delta, "done": False}
                            accumulated_response += delta
                        
                        # Only check hallucination if NOT in thinking mode
                        if not thinking and re.search(r'(?i)\*?System Tool Output', accumulated_response):
                             logging.warning("Hallucinated Tool block encountered in stream, suppressing remainder.")
                             break
                elif data.get("error"):    
                    logging.warning(f"{data['error']}")
                    yield {"error": data["error"], "done": True, "event": event}
                else:
                    logging.warning("Empty response")
                    yield {"error": "Empty response", "done": True, "event": event}
        async for item in gen():
            yield item
        return    
    else:
        logging.info(f"Requesting LLM {model}")
        data = await llm_request(main_payload)
        if not data:
            yield {"error": "⚠️ Request to LLM failed.", "done": True}
            return
        response = await process_response(data)
        yield response
        yield response
        return

async def perform_prompt(
    ctx: UserContext,
    instruction: str,
    message: str,
    chat: str="default", 
    intent: str|None=None,
    mem_id: str|None=None,
    img_source: str|None=None,
    event: str|None=None,
    stream: bool=False,
    provided_history: list|None=None,
    provided_knowledge: list|None=None
) -> str | AsyncGenerator:
    """Wrapper for _perform_prompt_gen to maintain backward compatibility."""
    
    gen = _perform_prompt_gen(
        ctx=ctx,
        instruction=instruction,
        message=message,
        chat=chat,
        intent=intent,
        mem_id=mem_id,
        img_source=img_source,
        event=event,
        stream=stream,
        provided_history=provided_history,
        provided_knowledge=provided_knowledge
    )
    
    if stream:
        return gen
    else:
        # Consume the generator and return the final result
        final_result = None
        async for item in gen:
            # We skip 'status' updates in non-streaming mode
            if item.get("status"):
                continue
            
            if item.get("error"):
                 return item["error"]
            
            # The final yield in non-stream mode is the response dict
            final_result = item
            
        if final_result:
            return final_result
        return "⚠️ Unknown error (empty response)"

# === Генерация картинок ===

# === Chats naming ==== #
async def generate_chat_title(message: str, model: str) -> str:
    """
    Спросить у LLM короткое имя для чата.
    """
    prompt = (
        "You are asked to generate a short (2–3 words) title for a chat conversation "
        "based on the following first message. Title should start with a suitable emoji separated by space. "
        "Return ONLY the title, no explanations.\n\n"
        f"Message: {message}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a naming assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": model, 
        "stream": False,
        "options": {"temperature": 0.3}
    }

    data = await llm_request(payload)
    if not data:
        return "New chat"
    return data["message"]["content"].strip() or "New chat"


def slugify(text):
    # Lowercase, remove special characters, replace spaces with underscores
    text = text.lower()
    # Remove emoji and most non-ascii characters (basic)
    text = re.sub(r'[^\w\s]', '', text)
    # Replace spaces and multiple underscores with single underscore
    text = re.sub(r'[\s_]+', '_', text)
    return text.strip('_')

async def ensure_chat(ctx: UserContext, chat: str, first_message: str = None) -> dict:
    # [LEGACY HISTORY] Backend-side chats index removed - handled by frontend/OrbitDB
    if not chat or chat in ["default", "newchat"]:
        # 1. Try to generate a nice title from first message first
        chat_title = "New chat"
        if first_message:
            try:
                model = get_llm_model(ctx)
                chat_title = await generate_chat_title(first_message, model)
            except Exception as e:
                logging.warning(f"Failed to generate chat title: {e}")
                chat_title = first_message[:30] + "..." if len(first_message) > 30 else first_message
                
        # 2. Generate new chat name (ID) using slugified title
        title_slug = slugify(chat_title)
        if not title_slug or title_slug == "new_chat":
             title_slug = "chat"
             
        # Use a shorter unique suffix instead of a long timestamp if needed, 
        # but let's try to keep it as clean as possible.
        unique_suffix = str(uuid.uuid4())[:4]
        chat_name = f"{title_slug}_{unique_suffix}"
        
        return {
            "name": chat_name,
            "title": chat_title,
            "file": f"{chat_name}.json",
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat()
        }
        
    return {
        "name": chat,
        "title": chat,
        "file": f"{chat}.json",
        "created": datetime.now(timezone.utc).isoformat(),
        "updated": datetime.now(timezone.utc).isoformat()
    }


# === Intent ===
async def classify_user_intent(ctx: UserContext, prompt: str, chat: str = "default", provided_history: list|None = None) -> str:
    chat = chat or "default"
    system_prompt = (
        f"{INTENT_PROMPT}\n\n"
        "CRITICAL: Return ONLY the classification (and path if needed) followed by a short reason. "
        "Do NOT repeat the instructions or the system prompt itself."
    )
    
    # Get last 4 messages from history
    try:
        if provided_history is not None:
             history = provided_history
        else:
             history = []
        
        last_messages = history[-2:] if history else []
        if last_messages:
            history_text = "\nChat History (last 2 messages):\n"
            for msg in last_messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    history_text += f"{role}: {content}\n"
            
            system_prompt += f"\n\n{history_text}\n"
    except Exception as e:
        logging.warning(f"Failed to load history for intent classification: {e}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.0, 
        }
    }
    
    data = await llm_request(request_payload)
    
    if data and "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    
    logging.warning(f"Classification failed, response: {data}")
    return "chat\nFallback" 


async def check_prompt_safety(ctx: UserContext, prompt: str) -> str:
    system_prompt = SAFETY_CHECK_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }
    
    data = await llm_request(request_payload)
    
    if data and "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    
    return "SAFE" # Default fallback

async def generate_image_prompt(ctx: UserContext, instruction: str, prompt: str, chat = "default", history: list = None) -> str:
    chat = chat or "default"
    appearance = ctx.settings.get("assistant_appearance") or user_context.DEFAULT_ASSISTANT_APPEARANCE
    
    # Extract tags (like <dream>, <Minerva>) from prompt and appearance so LLM tokenizer/special tokens are not confused
    tags_in_prompt = re.findall(r"<[^>]+>", prompt)
    tags_in_appearance = re.findall(r"<[^>]+>", appearance)
    all_tags = list(dict.fromkeys(tags_in_prompt + tags_in_appearance)) # Deduplicate preserving order
    
    # Clean tags from the text sent to the LLM
    clean_prompt_text = re.sub(r"<[^>]+>", "", prompt).strip()
    clean_appearance_text = re.sub(r"<[^>]+>", "", appearance).strip()

    user_prompt =  "*Personality and behaviour:*\n" + ctx.settings.get("system_prompt", "") + "\n\n*Appearance:*\n" + clean_appearance_text
    nsfw_enabled = ctx.settings.get("nsfw", False)

    # Clean prompt from slash commands
    clean_prompt = re.sub(r'^/(?:show|view|imagine|generate|recognize|detect|think|explain|search|import|learn)\s*', '', clean_prompt_text).strip()
    if not clean_prompt:
        clean_prompt = clean_prompt_text

    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{user_prompt}"
        image_instruction = f"{IMAGE_PROMPT_NSFW}\n{instruction.format(prompt=clean_prompt, appearance=clean_appearance_text)}"
    else:  
        system_prompt =  user_prompt
        image_instruction = instruction.format(prompt=clean_prompt, appearance=clean_appearance_text)



    # Format history as a structured text block instead of active chat messages to prevent model roleplay priming
    history_text = ""
    if history:
        history_text = "\n\n=== RECENT CONVERSATION HISTORY (for context: location, outfit, scene) ===\n"
        for msg in history[-15:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                 history_text += f"{role.upper()}: {content}\n"
        history_text += "========================================================\n"

    # Сборка запроса: ровно 2 сообщения, чтобы LLM не переключалась в режим диалога/списков
    messages = [
        {"role": "system", "content": system_prompt + history_text},
        {"role": "user", "content": image_instruction}
    ]


    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.5,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    # [LEGACY HISTORY] Save history removed - handled by frontend/OrbitDB

    # Deterministic post-processing to inject style, model, and appearance into the final prompt
    final_prompt = response.strip()

    # 1. Prepend the clean physical description at the beginning of the prompt (only for assistant/character image)
    if instruction == SYSTEM_INSTRUCTION_CHARACTER:
        if "Image:" in final_prompt:
            parts = final_prompt.split("Image:", 1)
            prompt_content = parts[1].strip()
            
            # Ensure "1girl, solo" prefix exists, then add clean appearance text
            prefix = "1girl, "
            if "solo" not in prompt_content.lower()[:25]:
                prefix += "solo, "
                
            final_prompt = f"{parts[0]}Image: {prefix}{clean_appearance_text}, {prompt_content}"
        else:
            final_prompt = f"1girl, solo, {clean_appearance_text}, {final_prompt}"

    # 2. Collect all style and model/LoRA tags to append to the end of the prompt
    tags_to_add = []
    
    # Add style tag (for all prompt types)
    style = ctx.settings.get("style", "realistic")
    if style:
        style_tag = f"<{style}>"
        if style_tag not in tags_to_add:
            tags_to_add.append(style_tag)
            
    # Add model/LoRA tags (only for assistant/character image)
    if instruction == SYSTEM_INSTRUCTION_CHARACTER:
        assistant_model = ctx.settings.get("assistant_model", "")
        if assistant_model:
            model_tag = assistant_model if assistant_model.startswith("<") else f"<{assistant_model}>"
            if model_tag not in tags_to_add:
                tags_to_add.append(model_tag)
                
        character_lora = ctx.settings.get("character_lora", "")
        if character_lora:
            lora_tag = character_lora if character_lora.startswith("<") else f"<{character_lora}>"
            if lora_tag not in tags_to_add:
                tags_to_add.append(lora_tag)

    # Merge in any raw tags extracted from the prompt/settings
    for tag in all_tags:
        if tag not in tags_to_add:
            tags_to_add.append(tag)

    # 3. Append the tags to the prompt
    if tags_to_add:
        tag_suffix = " " + " ".join(tags_to_add)
        if "Image:" in final_prompt:
             lines = final_prompt.split("\n")
             for idx, line in enumerate(lines):
                  if line.strip().startswith("Image:"):
                       lines[idx] = line.rstrip() + tag_suffix
                       break
             final_prompt = "\n".join(lines)
        else:
             final_prompt += tag_suffix

    logging.info(f"[image_prompt] Final Prompt with Tags & Appearance: {final_prompt}")
    return final_prompt


# Generate character image, returns full path for further sending or conversion
async def generate_character_image_prompt(ctx: UserContext, prompt, chat="default", history: list = None) -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx,  SYSTEM_INSTRUCTION_CHARACTER, prompt, chat, history)


# Generate general image, returns full path for further sending or conversion
async def generate_general_image_prompt(ctx: UserContext, prompt, chat="default", history: list = None) -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx, SYSTEM_INSTRUCTION_GENERAL, prompt, chat, history)


def extract_title_and_prompt(response: str) -> tuple[str, str]:
    """Extract title and image prompt from LLM response.
    Expected format: 'Title: title\nImage: prompt'
    Returns (prompt, title). If title not found, generates one from first words.
    """
    lines = response.strip().split('\n')
    title = ""
    img_prompt = ""
    
    # Try to find "Image:" marker and take everything after it
    if "Image:" in response:
        parts = response.split("Image:", 1)
        # Search for title in the first part
        title_part = parts[0].strip()
        img_prompt = parts[1].strip()
        
        for line in title_part.split('\n'):
             if line.strip().startswith("Title:"):
                 title = line.strip()[6:].strip()
                 break
    else:
        # Fallback loop for other formats or if Image: not found above (though check catches it)
        for line in lines:
            line = line.strip()
            if line.startswith("Title:"):
                title = line[6:].strip()
            elif line.startswith("Image:"):
                img_prompt = line[6:].strip()
    
    # Fallback: if no Title found, use image prompt or generate from prompt
    if not title and img_prompt:
        # Generate title from first few words
        words = img_prompt.replace('<', '').replace('>', '').split()
        clean_words = [w for w in words if not w.startswith('<')][:4]
        title = ' '.join(clean_words[:4])
    
    # Fallback: if no Image found, use entire response
    if not img_prompt:
        img_prompt = response.strip()
        if not title:
            words = img_prompt.replace('<', '').replace('>', '').split()
            title = ' '.join(words[:4])
    
    return img_prompt, title


async def generate_title_from_prompt(ctx: UserContext, prompt: str) -> str:
    """Generate a descriptive title from a raw user prompt.
    For short prompts (≤4 words), returns cleaned prompt.
    For long prompts, uses LLM to generate 3-4 word title.
    """
    # Clean tags from prompt
    clean_prompt = prompt.replace('<', '').replace('>', '')
    words = [w for w in clean_prompt.split() if not w.startswith('<')]
    
    # For short prompts, use the cleaned prompt itself
    if len(words) <= 4:
        return ' '.join(words)
    
    # For long prompts, generate a descriptive title using LLM
    title_prompt = (
        "Create a short descriptive title (3-4 words maximum) for this image generation prompt. "
        "Return ONLY the title, no quotes, no explanations.\n\n"
        f"Prompt: {prompt}"
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a title generation assistant."},
            {"role": "user", "content": title_prompt}
        ],
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    data = await llm_request(payload)
    if not data or "message" not in data:
         return ' '.join(words[:4])
         
    title = data["message"]["content"].strip().replace('"', '')
    return title


async def generate_neutral_description(ctx: UserContext, prompt: str) -> str:
    """Generate a safe, neutral description for the image prompt."""
    
    model = get_llm_model(ctx)
    if ctx.settings.get("nsfw", False):
         # In NSFW mode, we must prime the model to accept the input prompt
         # but still instruct it to output a neutral, safe description.
         instruction = NSFW_PREPHASE + "\n\n" + IMAGE_PROMPT_NSFW + "\n\n" + IMAGE_NEUTRAL_DESC_PROMPT.format(prompt=prompt)
    else:
         instruction = IMAGE_NEUTRAL_DESC_PROMPT.format(prompt=prompt)
    
    payload = {
        "messages": [
            {"role": "user", "content": instruction}
        ],
        "model": model,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    try:
        data = await llm_request(payload)
        if data and "message" in data:
             return data["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Failed to generate neutral description: {e}")
        
    return "Generated image" # Fallback

    try:
        data = await llm_request(payload)
        if data and "message" in data and "content" in data["message"]:
            title = data["message"]["content"].strip()
            logging.info(f"Generated title from raw prompt: {title}")
            return title
        else:
            # Fallback to first few words if LLM fails
            return ' '.join(words[:4])
    except Exception as e:
        logging.warning(f"Title generation failed: {e}")
        # Fallback to first few words
        return ' '.join(words[:4])



async def generate_image(ctx: UserContext, prompt, chat: str = 'default', update_history: bool = True, use_default_lora: bool = True, prompt_id: str | None = None) -> tuple[str, str, str]:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    if not ctx.storage:
        raise Exception("⚠️ Default storage is not available. Please connect your device.")

    user_id = ctx.user_id

    nsfw_enabled = ctx.settings.get("nsfw", False)

    negative_prompt = NEGATIVE_PROMPTS["base"]
    if nsfw_enabled:
        negative_prompt = NEGATIVE_PROMPTS["nsfw"] + "," + negative_prompt


    logging.info(f"Generating image for user: {user_id}")
    logging.info(f"Prompt: {prompt}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Промпт для генерации
    workflow_json["4"]["inputs"]["text"] = negative_prompt
    workflow_json["85"]["inputs"]["text"] =  prompt + ", " + IMPROVEMENT_PROMPT
    
    # Randomize seed
    seed = random.randint(1, 1125899906842624)
    if "5" in workflow_json and "inputs" in workflow_json["5"]:
        workflow_json["5"]["inputs"]["seed"] = seed

    # Выбираем модель в соответствии с режимом
    style = ctx.settings.get("style", "realistic")
    
    # Check for style tags in prompt
    tags = re.findall(r"<([^>]+)>", prompt)
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in STYLE_MODELS:
            style = tag_lower
            # If nsfw is enabled and we picked a base style, switch to nsfw version if available
            if nsfw_enabled and not style.endswith("_nsfw"):
                 if f"{style}_nsfw" in STYLE_MODELS:
                     style = f"{style}_nsfw"
            # If nsfw is disabled and we picked an nsfw style, switch to base version if available
            elif not nsfw_enabled and style.endswith("_nsfw"):
                 base_style = style[:-5]
                 if base_style in STYLE_MODELS:
                     style = base_style
            break

    # Apply NSFW suffix if not already present and using default setting logic (or if tag didn't handle it fully)
    # Actually, let's simplify: if we didn't find a tag, we use settings.
    # If we found a tag, we already tried to adjust it above.
    # But if we are using settings, we need to apply nsfw logic.
    
    # Re-evaluating logic flow:
    # 1. Default style from settings
    # 2. Override with tag if found
    # 3. Apply NSFW modifier based on nsfw_enabled flag
    
    style_from_tag = None
    for tag in tags:
        tag_lower = tag.lower()
        # Check if it is a valid style key (ignoring nsfw suffix for matching purposes if possible, or just match exact keys)
        # Let's match exact keys first, but also base keys.
        if tag_lower in STYLE_MODELS:
            style_from_tag = tag_lower
            break
            
    if style_from_tag:
        style = style_from_tag
        
    # Now ensure style matches nsfw setting
    if nsfw_enabled:
        if not style.endswith("_nsfw") and f"{style}_nsfw" in STYLE_MODELS:
            style = f"{style}_nsfw"
    else:
        if style.endswith("_nsfw"):
             base_style = style[:-5]
             if base_style in STYLE_MODELS:
                 style = base_style

    model = STYLE_MODELS.get(style, STYLE_MODELS["realistic"]) # Fallback just in case
    workflow_json["127"]["inputs"]["ckpt_name"] = model
    #workflow_json["103"]["inputs"][lora]["on"] = True
    
    # Dynamic LoRA activation
    lora_map = {}
    # Build map from name to key
    if "103" in workflow_json and "inputs" in workflow_json["103"]:
        for key, value in workflow_json["103"]["inputs"].items():
            if isinstance(value, dict) and "name" in value:
                lora_map[value["name"].lower()] = key

    # Find tags in prompt
    active_lora_keys = []
    tags = re.findall(r"<([^>]+)>", prompt)
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in lora_map:
            key = lora_map[tag_lower]
            active_lora_keys.append(key)
            logging.info(f"Found LoRA tag: {tag} ({key})")

    # Fallback to assistant_model if no tags found
    if not active_lora_keys and use_default_lora:
        assistant_model = ctx.settings.get("assistant_model", "").lower()
        if assistant_model in lora_map:
            key = lora_map[assistant_model]
            active_lora_keys.append(key)
            logging.info(f"Using default LoRA: {assistant_model} ({key})")

    # Activate LoRAs and set strength
    #lora_count = len(active_lora_keys)
    #target_strength = 1.0
    #if (style.startswith("perfect") or style.startswith("perfection")) and lora_count > 1:
    #    target_strength = 0.6
    
    for key in active_lora_keys:
        workflow_json["103"]["inputs"][key]["on"] = True
        #workflow_json["103"]["inputs"][key]["strength"] = target_strength
        logging.info(f"Activated LoRA {key} with strength {workflow_json["103"]["inputs"][key]["strength"]}")

    logging.info(f"Generating with model: {model}")
    img_data, filename = await generate_image_workflow(workflow_json)
    
    if not img_data:
        raise Exception("Image generation failed")


    # Report usage to console


    # Local user folder (only used if no remote storage)
    user_folder = os.path.join(APP_ROOT_DIR, USER_DATA_DIR, ctx.user_id, "generated")
    
    # Extract title from prompt (prompt may contain "Title: ..." from LLM)
    img_prompt, img_title = extract_title_and_prompt(prompt)
    
    # Format for markdown file
    formatted_prompt = f"#{img_title}\n\n{img_prompt}"
    
    # Generate neutral description for public Readme and history
    neutral_description = await generate_neutral_description(ctx, img_prompt)
    formatted_readme = f"#{img_title}\n\n{neutral_description}"
    
    # Create unique filename by appending timestamp to index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try to extract index from ComfyUI temp filename if present
    # Pattern: ComfyUI_temp_..._00005_.png
    match = re.search(r"_(\d{5})_\.png$", filename)
    if match:
        index = match.group(1)
        filename = f"IMG_{index}_{timestamp}.png"
    else:
        # Fallback for other ComfyUI temp patterns or unexpected names
        if filename.startswith("ComfyUI_temp"):
             filename = f"IMG_{timestamp}.png"

    if ctx.storage and ctx.omd_key:
        # Копируем файл юзеру на устройство
        dest = f"{ctx.storage}/generated"
        logging.info(f"Uploading to storage: {dest}/{filename}")
        
        # Since we have bytes, we use upload_data_to_storage (or similar, but upload_data_to_storage handles generic data? check implementation)
        # utils.upload_data_to_storage handles str or bytes.
        try:
            upload_data_to_storage(ctx.omd_key, dest, filename, img_data, "image/png")
            
            # Save prompt as description (Readme.md)
            readme_filename = os.path.splitext(filename)[0] + ".Readme.md"
            upload_data_to_storage(ctx.omd_key, dest, readme_filename, formatted_readme, "text/markdown")
            logging.info("Upload completed successfully.")
        except Exception as e:
            logging.error(f"Upload to storage failed: {e}")
            # Fallback to local? Or just fail? 
            # If upload fails, maybe we should try local save as backup?
            # For now just log.
            raise e

    else:    
        logging.info("No storage/key found, saving locally.")
        # Копируем файл в user_data (local)
        os.makedirs(user_folder, exist_ok=True)
        dest_path = os.path.join(user_folder, filename)
        with open(dest_path, "wb") as f:
            f.write(img_data)
        # Save prompt as description (Readme.md)
        readme_filename = os.path.splitext(filename)[0] + ".Readme.md"
        readme_path = os.path.join(user_folder, readme_filename)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(formatted_readme)
    return filename, img_title, neutral_description

async def generate_character_image(ctx: UserContext, prompt, chat: str = 'default', update_history: bool = True, prompt_id: str | None = None) -> tuple[str, str, str]:
    return await generate_image(ctx, prompt, chat, update_history=update_history, prompt_id=prompt_id)

# Generate general image, returns full path for further sending or conversion
async def generate_general_image(ctx: UserContext, prompt, chat: str = 'default', prompt_id: str | None = None) -> tuple[str, str, str]:
    return await generate_image(ctx, prompt, chat, prompt_id=prompt_id)

# img is base64 image #
async def recognize_image(ctx: UserContext, img, prompt="", chat="default"):

    nsfw_enabled = ctx.settings.get("nsfw", False)

    if nsfw_enabled:
        system_prompt = NSFW_PREPHASE + "\n" +  BASE_SYSTEM_PROMPT + "\n" + "Recognize image"
    else:
        system_prompt =  BASE_SYSTEM_PROMPT + "\n" + "Recognize image"


    try:
        history = []
    except Exception as e:
        logging.error(f"Failed to load history in recognize_image: {e}")
        raise e

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]
    # Добавляем историю
    messages.extend(history[-HISTORY_LIMIT:])
    # Добавляем новый запрос с изображением
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [img]
    })

    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.8,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    history.append({"role": "assistant", "content": response})
    # [LEGACY HISTORY] Save history removed

    return response.lower().strip()

# Суммаризация документа

async def summarize_for_memory(ctx: UserContext, raw_text: str, limit: int = 8000) -> str:
    """
    Создаёт 'карточку памяти' документа для дальнейшего поиска.
    :param raw_text: исходный текст документа
    :param limit: максимальное количество символов для передачи модели (по умолчанию ~8000)
    """
    # Усечём текст, если длиннее лимита
    text_to_process = raw_text[:limit]

    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": text_to_process},
    ]

    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    data = await llm_request(request_payload)

    # Универсальное извлечение текста
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            response = data["message"]["content"]
        else:
            response = data.get("content") or str(data)
    else:
        response = str(data)

    logging.info(f"Summary: {response}")    

    return response.strip()

    return response.strip()

# === Web Search Tool ===
async def search_web(ctx: UserContext, query: str) -> str:
    """
    Search the web using DuckDuckGo via Crawl4AI (headless browser) to bypass IP blocks.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        import urllib.parse
        
        logging.info(f"[search] Searching web for: {query}")
        
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(url=url)
            
            if not result or not result.markdown:
                logging.warning(f"[search] No markdown content returned")
                return "No results found."
            
            logging.info(f"[search] Got {len(result.markdown)} chars of markdown")
            
            # DuckDuckGo HTML results contain links and snippets
            # Look for result patterns: links (http) and content lines
            lines = result.markdown.split('\n')
            relevant_content = []
            
            for line in lines:
                line_stripped = line.strip()
                # Skip empty lines and navigation elements
                if not line_stripped:
                    continue
                # Skip short lines that are likely navigation
                if len(line_stripped) < 20:
                    continue
                # Skip lines that look like footer/navigation
                if any(skip in line_stripped.lower() for skip in ['privacy', 'terms', 'settings', 'safe search', 'next page']):
                    continue
                    
                relevant_content.append(line)
                if len(relevant_content) > 40:  # Increased limit for better results
                    break
            
            if not relevant_content:
                # Fallback: just return first chunk of markdown
                logging.warning(f"[search] No relevant content found, using raw markdown")
                return result.markdown[:3000]
            
            search_output = "\n".join(relevant_content)
            logging.info(f"[search] Returning {len(search_output)} chars of results")
            return search_output

    except ImportError:
        return "Error: crawl4ai library not installed. Web search unavailable."
    except Exception as e:
        logging.error(f"[search] Error searching {query}: {e}")
        return f"Error performing search: {e}"

# === Импорт и память ===

async def scrape_with_crawl4ai(url: str) -> str:
    """
    Scrape a URL using Crawl4AI (Playwright-based) to get clean Markdown.
    """
    try:
        # Lazy import to avoid crash if not installed
        from crawl4ai import AsyncWebCrawler
        
        logging.info(f"[crawl4ai] Starting crawl for: {url}")
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)
            
            if result.markdown:
                 logging.info(f"[crawl4ai] Success, length: {len(result.markdown)}")
                 return result.markdown
            else:
                 logging.warning(f"[crawl4ai] No markdown content returned")
                 return ""
                 
    except ImportError:
        logging.error("[crawl4ai] Library not installed. Please run: pip install crawl4ai playwright && playwright install")
        return ""
    except Exception as e:
        logging.error(f"[crawl4ai] Error scraping {url}: {e}")
        return ""

async def import_doc(ctx: UserContext, url_or_path, collection="user"):
    key = ctx.omd_key or ctx.settings.get("omd_key", "")

    logging.info(f"[import] importing: {url_or_path}")
    raw_text = ""
    # Определяем, это OMD или нет
    is_omd = url_or_path.startswith("/") or GATEWAY_URL in url_or_path
    
    if is_omd:
        if url_or_path.startswith("/"):
            url_or_path = f"{GATEWAY_URL}{url_or_path}"
        if not key:
            raise Exception("⚠️ Provide On My Disk account key to access your files:\n`/bind abcdxxxxx...`")
        
        # Check if it is a directory by checking the extension of the path (without query params)
        path_part = url_or_path.split("?")[0]
        ext = os.path.splitext(path_part)[-1].lower().lstrip('.')
        is_dir = not ext or ext == ''
        
        if is_dir:
            list_url = url_or_path
            if "?" in list_url:
                list_url += "&list"
            else:
                list_url += "?list"
            logging.info(f"[import] Directory detected, fetching listing: {list_url}")
            raw_text = await fetch_document_text(list_url, key)
            # If standard directory listing fetch fails or returns error, we fall back to normal fetch
            if raw_text.startswith("Failed to fetch document:"):
                logging.info("[import] Directory listing fetch failed, falling back to standard fetch")
                raw_text = await fetch_document_text(url_or_path, key)
        else:
            raw_text = await fetch_document_text(url_or_path, key)
    else:
        # External URL: Try Crawl4AI first for better parsing
        if url_or_path.startswith("http"):
            raw_text = await scrape_with_crawl4ai(url_or_path)
        
        # Fallback to standard fetch if Crawl4AI failed or returned empty
        if not raw_text:
            if url_or_path.startswith("http"):
                 logging.info("[import] Crawl4AI yielded no result, falling back to standard fetch")
            raw_text = await fetch_document_text(url_or_path)

    if not raw_text or not raw_text.strip():
        # This catch-all ensures we don't proceed with an empty string, 
        # which would trigger a generic frontend error.
        raw_text = "Failed to fetch document: Unsupported file type or document has no selectable text."

    if raw_text.startswith("Failed to fetch document:") or raw_text.startswith("Unsupported file type:"):
        logging.error(f"[import] failed fetch: {raw_text}")    
        return {
            "id": "error",
            "error": True,
            "text": raw_text
        }
    
    # Reject directory listings (JSON results) from being imported as documents
    # and instead recursively import their files.
    if raw_text.strip().startswith('{"list":') or raw_text.strip().startswith('{"result":'):
        logging.info(f"[import] identified as directory listing, recursively importing contents...")
        try:
            import json
            data = json.loads(raw_text)
            items = data.get("list", []) or data.get("result", [])
            imported_count = 0
            errors = []
            
            # Clean trailing slash from base url_or_path
            base_url = url_or_path.split("?")[0].rstrip('/')
            
            for item in items:
                name = item.get("name", "")
                item_type = item.get("type", "")
                if not name:
                    continue
                
                # Build sub-item path
                sub_url = f"{base_url}/{name}"
                # If it's a file, we import it
                if item_type == "file":
                    logging.info(f"[import] Recursively importing sub-file: {sub_url}")
                    try:
                        res = await import_doc(ctx, sub_url, collection=collection)
                        if res and not res.get("error"):
                            if "imported_count" in res:
                                imported_count += res["imported_count"]
                            else:
                                imported_count += 1
                        elif res:
                            errors.append(f"{name}: {res.get('text')}")
                    except Exception as sub_err:
                        logging.error(f"[import] Failed to import sub-file {sub_url}: {sub_err}")
                        errors.append(f"{name}: {sub_err}")
                elif item_type == "dir":
                    # Recursively import sub-directories
                    logging.info(f"[import] Recursively importing sub-directory: {sub_url}")
                    try:
                        res = await import_doc(ctx, sub_url, collection=collection)
                        if res and not res.get("error"):
                            imported_count += int(res.get("imported_count", 0))
                        elif res:
                            errors.append(f"{name}: {res.get('text')}")
                    except Exception as sub_err:
                        logging.error(f"[import] Failed to import sub-dir {sub_url}: {sub_err}")
                        errors.append(f"{name}: {sub_err}")
            
            return {
                "id": "directory",
                "text": f"Successfully imported {imported_count} files from directory '{url_or_path.split('?')[0]}' into collection '{collection}'." + (f" Errors: {', '.join(errors)}" if errors else ""),
                "imported_count": imported_count
            }
        except Exception as e:
            logging.error(f"[import] Error parsing directory listing JSON: {e}")
            return {
                "id": "error",
                "error": True,
                "text": f"Error listing directory: {e}"
            }

    # If it's an external HTML or we just want to ensure it's plain text via pandoc
    if not is_omd or url_or_path.lower().endswith(".html") or url_or_path.lower().endswith(".htm"):
        # We use pandoc to clean up HTML or other formats if needed
        # Note: fetch_document_text for OMD might already return clean text if ?totext was used,
        # but for external URLs it returns raw HTML.
        
        cmd = [
            "pandoc",
            "-f", "html",
            "-t", "plain",
        ]
        try:
            logging.info(f"[import] converting with pandoc (input length: {len(raw_text)})")
            result = subprocess.run(cmd, input=raw_text, capture_output=True, text=True, check=True)
            raw_text = result.stdout
        except Exception as e:
            logging.error(f"[import] pandoc failed: {e}")    
            return {
                "id": "error",
                "error": True,
                "text": f"Error during conversion: {e}"
            }

    # Векторизация и сохранение чанков (только для общих коллекций)
    if collection != "user":
        chunk_and_vectorize_to_file(
            ctx,
            text=raw_text,
            document_id=url_or_path,
            collection=collection
        )
    else:
        logging.info(f"[import] Skipping backend vectorization for user collection.")

    # Добавление краткой аннотации в память (только для общих коллекций)
    card_text = await summarize_for_memory(ctx, raw_text)
    if collection != "user":
        mem_id = add_memory_card(
            ctx,
            text=card_text,
            document_id=url_or_path,
            collection=collection
        )
    else:
        logging.info(f"[import] Skipping backend memory card for user collection. Frontend handles personal knowledge.")
        mem_id = f"user_{url_or_path}"

    mem_card = {
        "id": mem_id,
        "text": card_text,
        "full_text": raw_text
    }
    return mem_card

def memorize(ctx, text):
    # Backend memorization is disabled for "user" collection.
    # New facts are extracted and sent to frontend via 'newFact' event in api.py
    logging.info(f"[memory] Backend memorization skipped for: {text}")
    return f"Fact received (backend memorization is disabled, frontend should handle it)"

async def extract_and_save_memory(ctx: UserContext, message: str) -> str:
    # Deprecated: Background extraction via tools proved too unstable for gemma3:12b without strict guidance.
    # We now extract memories inside classify_user_intent.
    return None


async def generate_avatar(ctx: UserContext, style: str, character_lora: str, prompt: str):
    try:
        # 1. Load Workflow
        if not os.path.exists(WORKFLOW_PATH):
            logging.error("Workflow file not found")
            return None
            
        try:
            with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load workflow: {e}")
            return None
    
        # 2. Determine Style Checkpoint
        style_map = {
            "realistic": "realistic",
            "fantasy": "fantasy", 
            "perfect": "perfect",
            "tooned": "tooned"
        }
        backend_style = style_map.get(style, "realistic")
        
        nsfw = ctx.settings.get("nsfw", False)
        if nsfw:
            backend_style_nsfw = backend_style + "_nsfw"
            if backend_style_nsfw in STYLE_MODELS:
                backend_style = backend_style_nsfw
                
        # STYLE_MODELS values are filenames (strings), not dicts
        ckpt_filename = STYLE_MODELS.get(backend_style, STYLE_MODELS["realistic"])
    
        # 3. Setup Workflow Parameters
        seed = random.randint(1, 9999999999)
        
        # Helper to find node
        def find_nodes_by_class(class_type):
            return [node for node in workflow.values() if node.get("class_type") == class_type]
    
        # Set Seed
        for node in find_nodes_by_class("KSampler"):
            if "inputs" in node and "seed" in node["inputs"]:
                node["inputs"]["seed"] = seed
                
        # Set Resolution (512x512)
        for node in find_nodes_by_class("EmptyLatentImage"):
            if "inputs" in node:
                node["inputs"]["width"] = 512
                node["inputs"]["height"] = 512
            
        # Set Checkpoint
        for node in find_nodes_by_class("CheckpointLoaderSimple"):
            if "inputs" in node:
                node["inputs"]["ckpt_name"] = ckpt_filename
            
            # Set Prompt
        prompt_set = False
        
        # Add appearance to prompt
        negative_prompt = NEGATIVE_PROMPTS["nsfw"] + ", " + NEGATIVE_PROMPTS["base"]
        appearance = ctx.settings.get("assistant_appearance", "")
        full_prompt = f"{prompt} {appearance}"
        logging.info(f"Generating avatar with style={style}, character={character_lora} prompt={full_prompt}")
        
        workflow["4"]["inputs"]["text"] = negative_prompt
        workflow["85"]["inputs"]["text"] = full_prompt
    
        # 4. Inject Character LoRA (if workflow supports it)
        if character_lora:
            # We look for "Power Lora Loader (rgthree)" as used in available_loras
            lora_nodes = find_nodes_by_class("Power Lora Loader (rgthree)")
            if lora_nodes:
                for node in lora_nodes:
                    inputs = node.get("inputs", {})
                    # The loader might have inputs like lora_1, lora_2 etc which are dicts? 
                    # based on previous analysis of available_loras, input val is dict with "name"
                    for key, val in inputs.items():
                        if isinstance(val, dict) and val.get("name") == character_lora:
                            val["on"] = True
                            logging.info(f"Enabled LoRA: {character_lora}")
            else:
                # If standard LoraLoader?
                lora_nodes = find_nodes_by_class("LoraLoader")
                if lora_nodes:
                    # We need mapping from Name -> Filename.
                    # This is tricky without reading analog_character_lora.json or having a map.
                    # For now, if usage implies Power Lora Loader, we stick to that or skip.
                    logging.warning("Character LoRA requested but no compatible LoRA loader found in workflow.")
                        
        # 5. Generate
        # Reuse existing workflow generator which handles websocket and bytes retrieval
        image_data, _ = await generate_image_workflow(workflow)
        
        if image_data:
            # Upload to 'generated' folder in user storage
            filename = f"avatar_{uuid.uuid4()}.png"
            
            if ctx.storage and ctx.omd_key:
                try:
                    dest_path = f"{ctx.storage}/generated/avatars"
                    # upload_data_to_storage(omd_key, dest, filename, data, mime)
                    upload_data_to_storage(ctx.omd_key, dest_path, filename, image_data, "image/png")
                    logging.info(f"Avatar uploaded to {dest_path}/{filename}")
                    
                    # Report usage to console


                    # Construct public URL
                    clean_storage = ctx.storage.strip("/")
                    full_url = f"/{clean_storage}/generated/{filename}"
                    
                    return {"image": filename, "url": full_url}
                except Exception as e:
                    logging.error(f"Failed to upload avatar to storage: {e}")
                    return None
            else:
                 # Fallback for local users (if any, though context implies OMD usage mostly)
                 # But we want to avoid local fs if possible. 
                 # If no storage, we might have to save locally or fail?
                 # Let's save locally as fallback but log warning.
                 output_dir = os.path.join(os.path.dirname(__file__), "generated")
                 if not os.path.exists(output_dir):
                     os.makedirs(output_dir, exist_ok=True)
                 filepath = os.path.join(output_dir, filename)
                 with open(filepath, "wb") as f:
                     f.write(image_data)
                 logging.warning(f"No storage context, saved locally to {filepath}")
                 return {"image": filename}

        else:
            logging.error("No image data received from workflow")
            return None

    except Exception as e:
        logging.error(f"Avatar generation crashed: {e}", exc_info=True)
        return None


