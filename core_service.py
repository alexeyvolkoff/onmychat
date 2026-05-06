import os
import hashlib
import re
import random
import json
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
    search_memories
)


DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]

# LLM and RAG settings #
OLLAMA_URL = SETTINGS["OLLAMA_URL"]
CODE_BASE_URL = SETTINGS.get("CODE_BASE_URL", "http://localhost:4096")
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
NSFW_MODEL = SETTINGS["NSFW_MODEL"]
MCP_MODEL = SETTINGS.get("MCP_MODEL", "google/function-gemma")

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
      "name": "search_web",
      "description": "Search the internet for real-time information. REQUIRED for queries about 'weather', 'news', 'flights', 'stocks', 'events', or any topic not in your training data. Do NOT invent new tools like 'weather_forecast' or 'flights'.",
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
          "name": "search_memory",
          "description": "Search user memory for facts, details or information. Use ONLY if the file system tools fail to provide info.",
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
  }
]

# === MCP TOOLS ===
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
        "model": SFW_MODEL,
        "messages": [{"role": "system", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    data = await llm_request(payload)
    if not data or "message" not in data:
        return "[NO FILE FOUND]"
    
    return data["message"]["content"].strip()

async def list_supported_tools(ctx: UserContext) -> str:
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
                if resp.status == 200:
                    data = await resp.json()
                    # OMD Gateway list format
                    items = data.get("list", [])
                    if not items and "result" in data:
                        items = data["result"]
                    
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
        # Re-using existing sync search_memories but wrapping it if needed.
        # core_service imports search_memories.
        # search_memories(ctx, query, collection=..., mem_id=..., top_k=3)
        # We search both user and generic collection if possible? Or just default kb_id.
        collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
        
        # It's a sync function in memory_index.py? 
        # Imported as: from memory_index import search_memories
        # We should check if it's async. core_service calls it without await in inject_facts? 
        # Line 463: personal = search_memories(...)
        # So it is synchronous. We can run it in executor if it's slow, but for now direct call.
        
        results = search_memories(ctx, query, collection=collection, top_k=5)
        if not results:
            return "No relevant memories found."
            
        output = f"Memory Search Results for '{query}':\n"
        for i, res in enumerate(results, 1):
             text = res.get('text', '')
             output += f"{i}. {text}\n"
        return output
    except Exception as e:
        return f"Error searching memory: {e}"

async def write_omd_file(ctx: UserContext, path: str, content: str) -> str:
    if not ctx.omd_key or not ctx.storage:
        return "Error: OMD storage not linked."

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

async def check_and_execute_mcp(ctx: UserContext, message: str) -> AsyncGenerator[dict, None]:
    # 1. Path Extraction Heuristic (Help the model find the path)
    potential_paths = re.findall(r"(\/[\w\-\.\/]+)", message)
    path_hint = ""
    
    if potential_paths:
        path_hint = f"\nSYSTEM HINT: Detected paths: {', '.join(potential_paths)}. \nSTRATEGY: You MUST use `list_omd_files` first to see what's inside before trying to read."

    # 2. Native Tool Call Loop (Multi-Turn)
    system_instruction = DEFAULT_MCP_INSTRUCTIONS

    # Initial Turn: Mandatory [PLAN] Phase
    # We use a more permissive system prompt for the planning turn.
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"User Request: {message}\n{path_hint}"}
    ]
    
    all_tool_results = ""
    max_turns = 8 # Increased for complex autonomous tasks
    listed_paths = set()
    known_files = set() # Strict cache of verified files
    call_history = set() # Prevent repeated failed attempts
    
    # [USER INTENT DETECTION]
    # Parse USER'S request for action verbs that imply reading/extraction.
    # These are universal command verbs, NOT content keywords.
    read_action_verbs = [
        "read", "extract", "show", "display", "open", "contents", 
        "what is inside", "totals", "amount", "items", "data from", "get"
    ]
    message_lower = message.lower()
    requires_read = any(verb in message_lower for verb in read_action_verbs)
    if requires_read:
        logging.info(f"[MCP] User Intent: READ/EXTRACT detected. Will force read after discovery.")
    else:
        logging.info(f"[MCP] User Intent: FIND ONLY.")
    
    has_read_file = False
    last_listing = "" # For re-injection on errors
    
    # The agent is now autonomous and reasoning-based. 
    # It will manage its own data extraction logic via the [PLAN] mandate.

    
    for turn in range(max_turns):
        # [TURN-SPECIFIC GUIDANCE]
        guidance = "You are an autonomous agent. "
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
                 {"role": "system", "content": messages[0]["content"] + "\nCRITICAL: Respond ONLY with tool calls. Do NOT explain. \n\n" + f"CURRENT GUIDANCE: {guidance}"}
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
        
        # [REASONING CAPTURE]
        # Ensure that plans, thoughts, and checklists are preserved and visible to the main assistant.
        agent_text = msg.get("content", "").strip()
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
                  all_tool_results += f"Agent Reasoning (Turn {turn+1}):\n{clean_agent_text}\n\n"
                  
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
                  
                  all_tool_results += f"\nAgent Conclusion:\n{agent_text}\n"
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
            "save_user_fact": "learning"
        }
        status_msg = status_map.get(name, "executing")
        
        # Extract arguments for localization
        status_args = {}
        if name == "list_omd_files":
             status_args["path"] = os.path.basename(args.get('path', 'directory').rstrip('/'))
        elif name == "read_omd_file":
             status_args["path"] = os.path.basename(args.get('path', 'file'))
        elif name == "find_omd_file":
             status_args["path"] = os.path.basename(args.get('root_directory', 'root').rstrip('/'))
        elif name == "write_omd_file":
             status_args["path"] = os.path.basename(args.get('path', 'file'))
        elif name == "search_web":
             status_args["path"] = args.get('query', '')

        # Yield status event
        yield {"type": "status", "content": status_msg, "args": status_args}

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
                  path_arg = args.get("path", "").strip()
                  
                  # [LOOP PREVENTION]
                  if path_arg.rstrip("/") in listed_paths:
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
                 path_arg = args.get("path", "").strip()
                 
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
                     path_arg = args.get("path", "").strip()
                     
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

        if not found_new_info:
            break
            
        # [TURN RESET]
        # We break the tool list loop here (we only used one tool) and allow the 
        # main 'turn' loop to call the model again with the new knowledge.

    # Hallucination lockdown is now handled by the planning requirement and instructions.
    
    # Suppression filter for final MCP log
    last_content = messages[-1].get("content", "").strip() if messages and messages[-1].get("role") == "assistant" else ""
    refusal_keywords = [
        "i cannot", "i'm sorry", "i am sorry", "cannot assist", 
        "current capabilities are limited", "cannot generate",
        "do not have access", "as an ai model", "just a chatty assistant",
        "cannot help with"
    ]
    if last_content and not any(phrase in last_content.lower() for phrase in refusal_keywords):
        logging.info(f"[MCP] Agent Conclusion: {last_content[:100]}...")
    
    # If the autonomous agent didn't produce any meaningful output, report failure
    if not all_tool_results.strip():
        all_tool_results = "Tool Output: No files were listed or read. The discovery phase returned no data."
        logging.warning("[MCP] Autonomous loop finished with ZERO results.")
    elif not has_read_file and "find_omd_file" in all_tool_results:
        # Check if we were supposed to read but didn't
        if requires_read:
             all_tool_results += "\nSYSTEM NOTICE: Discovery succeeded, but no file content was read. The requested details (totals/items) are NOT in this tool output."
        
    yield {"type": "result", "content": all_tool_results}

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
        return None, None



# === RAG ===
async def inject_facts(ctx: UserContext, query: str, collection: str = "", mem_id="", provided_knowledge: list|None = None) -> tuple[list[str], list[str]]:
    facts = []
    document_ids = []

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
    if collection:
        shared = search_memories(ctx, query, collection=collection, mem_id=mem_id, top_k=3)
        for m in shared:
            facts.append(f"• {m['text']}")
            doc_id = m.get("document_id")
            if doc_id:
                document_ids.append(doc_id)

    return facts, document_ids

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

    nsfw_enabled = ctx.settings.get("nsfw", False)
    #model = get_llm_model(ctx)
    model = DEFAULT_MODEL
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
    doc_ids = []
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    logging.info(f"Loading facts: {collection} {is_rag}")

    # === Facts injection ===
    facts, doc_ids = await inject_facts(ctx, message, collection, mem_id, provided_knowledge=provided_knowledge)
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

        # Добавляем блок с источниками — только в отображаемый ответ
        links = []
        if doc_ids:
            seen = set()
            for doc in doc_ids:
                if doc in seen:
                    continue  # пропускаем дубликаты
                seen.add(doc)
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(doc)
    
        # result object
        response = {}
        response["content"] = llm_response.strip()
        if links:  # добавляем блок только если есть ссылки
            response["sources"] = links
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
            if links:
                history_entry["sources"] = links
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
    user_prompt =  "*Personality and behaviour:*\n" + ctx.settings.get("system_prompt", "") + "\n\n*Appearance:*\n" + ctx.settings.get("assistant_appearance", "")
    nsfw_enabled = ctx.settings.get("nsfw", False)
    #model =  NSFW_MODEL if nsfw_enabled else SFW_MODEL

    # Clean prompt from slash commands
    clean_prompt = re.sub(r'^/(?:show|view|imagine|generate|recognize|detect|think|explain|search|import|learn)\s*', '', prompt).strip()
    if not clean_prompt:
        clean_prompt = prompt

    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{user_prompt}"
        image_instruction = f"{IMAGE_PROMPT_NSFW}\n{instruction.format(prompt=clean_prompt, appearance=ctx.settings.get('assistant_appearance', ''))}"
    else:  
        system_prompt =  user_prompt
        image_instruction = instruction.format(prompt=clean_prompt, appearance=ctx.settings.get('assistant_appearance', ''))


    history = history or []

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю с явным указанием контекста
    if history:
        messages.append({"role": "system", "content": "CONVERSATION CONTEXT (use this to determine current location, character's outfit, and situation):"})
        messages.extend(history[-20:])

    # Добавляем запрос
    messages.append({ "role": "user", "content": image_instruction})

    #model = NSFW_MODEL if nsfw_enabled else SFW_MODEL


    request_payload = {
        "messages": messages,
        "model": get_llm_model(ctx),
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    # [LEGACY HISTORY] Save history removed - handled by frontend/OrbitDB

    return response.strip()


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
    if raw_text.strip().startswith('{"list":') or raw_text.strip().startswith('{"result":'):
        logging.info(f"[import] identified as directory listing, skipping import.")
        return None

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


