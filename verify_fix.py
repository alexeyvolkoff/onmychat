import sys
import asyncio
import os
from datetime import datetime, timezone

# Add project root to path
sys.path.append('/home/alexey/projects/onmychat')

# Mock UserContext
class MockContext:
    def __init__(self):
        self.user_id = "test_user"
        self.storage = None
        self.omd_key = None
        self.history = []

async def test_ensure_chat():
    try:
        from core_service import ensure_chat
        print("Successfully imported ensure_chat from core_service")
        
        ctx = MockContext()
        chat_id = "test_chat_restored"
        message = "Hello world"
        
        # Ensure directory exists for local fallback
        os.makedirs(f"/home/alexey/projects/onmychat/data/{ctx.user_id}/chats", exist_ok=True)
        
        # Test 1: Non-default chat (should not trigger LLM)
        print(f"Testing ensure_chat with chat_id='{chat_id}'...")
        chat_info = await ensure_chat(ctx, chat_id, message)
        print(f"ensure_chat returned: {chat_info}")
        
        if chat_info['name'] == chat_id:
            print("Test 1 PASSED")
        else:
            print("Test 1 FAILED: ID mismatch")

        # Test 2: Verify timestamp format
        if "created" in chat_info and "updated" in chat_info:
             print("Test 2 PASSED: Timestamps present")
        else:
             print("Test 2 FAILED: Missing timestamps")

    except ImportError as e:
        print(f"ImportError: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ensure_chat())
