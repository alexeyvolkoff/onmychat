# Agent Instructions

# Global Agent Instructions

## TOOL_USE RULES (CRITICAL)
1. **Prevent Empty Output**: Standard tools (grep, ls, find, git) sometimes return nothing. To prevent the agent from hanging:
   - ALWAYS append `|| echo "NO_OUTPUT_RETURNED"` to any command that might return an empty string.
   - Example: `grep "pattern" file.txt || echo "pattern not found"`
2. **No Retries on Empty**: If a command returns "NO_OUTPUT_RETURNED", do not loop. Accept it as a valid result and move to the next logical step.

3. Never try to rebuild or restart Docker containers. It is done by CI
4. Never try to restart or investigate nginx proxy. This project runs on staging environment, not locally.
5. Never rebuild the frontend. It is done by CI
6. Never try to restart the backend. Ask user to do it manually.
7. Never call shell commands that require sudo
8. NEVER read or write files outside the workspace directory. It causes the agent to hang indefinitely.


## Knowledge Management
- Before exploring any topic, first check `.knowledge/` directory in the project root for existing research notes
- If relevant and up-to-date knowledge exists, skip exploration and use what's there
- When explore sub-agent discovers new findings, save them to `.knowledge/<topic>.md` using descriptive filenames
- Always keep `.knowledge/` organized — one file per topic, cross-link when needed

## SYSTEM COMPATIBILITY (Ubuntu 24.04)
- Use standard bash syntax. 
- If a tool call fails with "undefined" error, check if you missed the description field and fix it immediately.

