#!/usr/bin/env python3
"""
Telegram Setup Helper

This script helps you set up and troubleshoot your Telegram bot configuration.
It will help you find your chat ID and test your bot connection.
"""

import os
import requests
import json
from typing import Dict, Any, List

try:
    from .utils.config import config
    CONFIG_AVAILABLE = True
except ImportError:
    try:
        # Try relative import for when run as script
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent))
        from utils.config import config
        CONFIG_AVAILABLE = True
    except ImportError:
        CONFIG_AVAILABLE = False

def get_telegram_token() -> str:
    """Get Telegram token from config or environment."""
    if CONFIG_AVAILABLE and config.telegram_token:
        return config.telegram_token
    return os.getenv("TELEGRAM_TOKEN", "")

def test_bot_token(token: str) -> Dict[str, Any]:
    """Test if the bot token is valid."""
    print(f"🤖 Testing bot token: {token[:10]}...")
    
    url = f"https://api.telegram.org/bot{token}/getMe"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("ok"):
            bot_info = data["result"]
            print(f"✅ Bot token is valid!")
            print(f"   Bot name: {bot_info.get('first_name')}")
            print(f"   Bot username: @{bot_info.get('username')}")
            print(f"   Bot ID: {bot_info.get('id')}")
            return {"valid": True, "bot_info": bot_info}
        else:
            print(f"❌ Bot token is invalid: {data.get('description')}")
            return {"valid": False, "error": data.get('description')}
            
    except Exception as e:
        print(f"❌ Error testing bot token: {e}")
        return {"valid": False, "error": str(e)}

def get_chat_updates(token: str) -> List[Dict[str, Any]]:
    """Get recent updates to find chat IDs."""
    print("📱 Getting recent chat updates...")
    
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get("ok"):
            updates = data.get("result", [])
            print(f"📨 Found {len(updates)} recent updates")
            return updates
        else:
            print(f"❌ Failed to get updates: {data.get('description')}")
            return []
            
    except Exception as e:
        print(f"❌ Error getting updates: {e}")
        return []

def extract_chat_ids(updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract unique chat IDs from updates."""
    chats = {}
    
    for update in updates:
        message = update.get("message", {})
        chat = message.get("chat", {})
        
        if chat:
            chat_id = chat.get("id")
            chat_type = chat.get("type")
            chat_title = chat.get("title") or chat.get("first_name") or "Unknown"
            
            if chat_id:
                chats[chat_id] = {
                    "id": chat_id,
                    "type": chat_type,
                    "title": chat_title,
                    "username": chat.get("username")
                }
    
    return list(chats.values())

def test_chat_id(token: str, chat_id: str) -> bool:
    """Test if we can send a message to the chat ID."""
    print(f"💬 Testing chat ID: {chat_id}")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": "🤖 Test message from your trading bot setup helper!"
    }
    
    try:
        response = requests.post(url, data=payload, timeout=10)
        data = response.json()
        
        if data.get("ok"):
            print(f"✅ Successfully sent test message to chat {chat_id}")
            return True
        else:
            print(f"❌ Failed to send message to chat {chat_id}: {data.get('description')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chat {chat_id}: {e}")
        return False

def main():
    """Main setup helper function."""
    print("🚀 Telegram Bot Setup Helper")
    print("=" * 50)
    
    # Get token
    token = get_telegram_token()
    
    if not token or token == "YOUR_TELEGRAM_BOT_TOKEN":
        print("❌ No valid Telegram token found!")
        print("\n📋 To set up your Telegram bot:")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the bot token")
        print("4. Set TELEGRAM_TOKEN environment variable or add to .env file")
        print("\nExample .env entry:")
        print("TELEGRAM_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz")
        return
    
    # Test bot token
    bot_test = test_bot_token(token)
    if not bot_test["valid"]:
        return
    
    print("\n" + "=" * 50)
    
    # Get updates to find chat IDs
    updates = get_chat_updates(token)
    
    if not updates:
        print("\n📱 No recent messages found!")
        print("💡 To find your chat ID:")
        print("1. Send a message to your bot (any message)")
        print("2. Run this script again")
        print("3. Your chat ID will be displayed")
        return
    
    # Extract chat IDs
    chats = extract_chat_ids(updates)
    
    if not chats:
        print("❌ No chats found in updates")
        return
    
    print(f"\n📋 Found {len(chats)} chat(s):")
    for i, chat in enumerate(chats, 1):
        print(f"{i}. Chat ID: {chat['id']}")
        print(f"   Type: {chat['type']}")
        print(f"   Title: {chat['title']}")
        if chat['username']:
            print(f"   Username: @{chat['username']}")
        print()
    
    # Test each chat ID
    print("🧪 Testing chat IDs...")
    working_chats = []
    
    for chat in chats:
        if test_chat_id(token, str(chat['id'])):
            working_chats.append(chat)
    
    if working_chats:
        print(f"\n✅ Found {len(working_chats)} working chat(s)!")
        print("\n📋 Add this to your .env file:")
        
        # Recommend the first working chat (usually the private chat with the user)
        recommended_chat = working_chats[0]
        print(f"TELEGRAM_CHAT_ID={recommended_chat['id']}")
        
        if len(working_chats) > 1:
            print("\n💡 Other working chat IDs:")
            for chat in working_chats[1:]:
                print(f"# TELEGRAM_CHAT_ID={chat['id']}  # {chat['title']}")
    else:
        print("\n❌ No working chat IDs found!")
        print("💡 Make sure you've sent a message to your bot first")

if __name__ == "__main__":
    main()
