import requests

# Your URL
url = "https://7860-01kcgwxzhc7g3w7a2es7k74291.cloudspaces.litng.ai/chat"

# Valid JSON (no literal newlines inside values)
payload = {
    "message": "كيف أتعامل مع طفل التوحد؟"
}

try:
    response = requests.post(url, json=payload)
    print("Status:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)