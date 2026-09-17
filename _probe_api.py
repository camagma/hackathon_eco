import os

import requests
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("NVIDIA_API_KEY")
print("key_len", len(key or ""), "starts_nvapi", (key or "").startswith("nvapi-"))

payload = {
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Reply with OK"}],
    "max_tokens": 8,
}

variants = [
    {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"},
    {"Authorization": key, "Accept": "application/json", "Content-Type": "application/json"},
    {"NVIDIA-API-KEY": key, "Accept": "application/json", "Content-Type": "application/json"},
]

url = "https://integrate.api.nvidia.com/v1/chat/completions"
for i, headers in enumerate(variants):
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("--- variant", i, r.status_code)
    print("www-auth", r.headers.get("www-authenticate"))
    print(r.text[:400].replace("\n", " "))
