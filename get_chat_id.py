#!/usr/bin/env python3
"""Quick script to get your Telegram chat ID"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('TELEGRAM_TOKEN')

print("🔍 Getting recent Telegram updates...")
url = f'https://api.telegram.org/bot{token}/getUpdates'
response = requests.get(url)
data = response.json()

if data.get('ok') and data.get('result'):
    print("📨 Recent chats:")
    for update in data['result']:
        if 'message' in update:
            chat = update['message']['chat']
            print(f"Chat ID: {chat['id']}")
            print(f"Type: {chat['type']}")
            if 'first_name' in chat:
                print(f"Name: {chat['first_name']}")
            if 'username' in chat:
                print(f"Username: @{chat['username']}")
            print("---")
else:
    print("❌ No recent messages found. Send a message to your bot first!")
