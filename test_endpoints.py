import time
import requests

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:80"

def test_endpoint(method, url, payload=None):
    start = time.time()
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=payload, timeout=30)
        latency = (time.time() - start) * 1000
        print(f"[{method}] {url} - Status: {response.status_code} - Latency: {latency:.2f}ms")
        if response.status_code != 200:
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"[{method}] {url} - FAILED: {str(e)}")

print("Testing Endpoints...")
test_endpoint("GET", f"{FRONTEND_URL}/")
test_endpoint("GET", f"{BASE_URL}/health")
test_endpoint("GET", f"{BASE_URL}/ready")
test_endpoint("GET", f"{BASE_URL}/system/llm-status")

# We will test /analyze and /retrieve with specific payloads later
