import requests
import sys

try:
    res = requests.get("http://localhost:8000/system/llm-status", timeout=10)
    print("Status Check:", res.json())
except Exception as e:
    print("Failed Status Check:", e)

try:
    res2 = requests.post("http://localhost:8000/analyze", json={"query": "What causes flooding in urban India?"}, timeout=30)
    print("Analyze Result:", res2.json())
except Exception as e:
    print("Failed Analyze Check:", e)
