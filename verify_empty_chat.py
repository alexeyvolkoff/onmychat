import sys
import asyncio
import os
from datetime import datetime, timezone

# Add project root to path
sys.path.append('/home/alexey/projects/onmychat')

# Mock UserContext
class MockContext:
    def __init__(self):
        self.user_id = "test_user_empty"
        self.storage = None
        self.omd_key = None
        self.history = []

async def test_empty_chat():
    try:
        from core_service import ensure_chat
        print("Successfully imported ensure_chat from core_service")
        
        ctx = MockContext()
        chat_id = "" # Empty chat ID
        message = "Hello world this is a new chat"
        
        # Ensure directory exists for local fallback
        chats_dir = f"/home/alexey/projects/onmychat/user_data/{ctx.user_id}/chats"
        os.makedirs(chats_dir, exist_ok=True)
        chats_file = os.path.join(chats_dir, "chats.json")
        if os.path.exists(chats_file):
            os.remove(chats_file)
        
        print(f"Testing ensure_chat with chat_id='{chat_id}'...")
        chat_info = await ensure_chat(ctx, chat_id, message)
        print(f"ensure_chat returned: {chat_info}")
        
        if chat_info['name'] == "" or chat_info['title'] == "Chat ":
            print("Issue REPRODUCED: Created chat with empty name/title")
        else:
            print(f"Issue NOT REPRODUCED: Created chat with name '{chat_info['name']}'")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_empty_chat())
