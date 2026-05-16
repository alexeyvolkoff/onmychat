import requests
import json

url = "http://localhost:4000/chat"
token = "3a4cf8f0663b69bba03f8c04149df7ed"
headers = {
    "Content-Type": "application/json"
}

payload = {
    "prompt": "What do you know about the project Lukas invoice?",
    "chat": "test_lukas",
    "omd_key": token,
    "storage": "alexey",
    "history": [],
    "settings": {}
}

print(f"Sending request to {url}...")
response = requests.post(url, headers=headers, json=payload)
print(f"Status: {response.status_code}")
try:
    data = response.json()
    content = data.get('content')
    print(f"Response:\n{content}")
    if 'sources' in data:
        print("\nSources:")
        for s in data['sources']:
            print(f"- {s.get('title')} (Owner: {s.get('owner')})")
except Exception as e:
    print(f"Error parsing response: {e}")
    print(response.text)
