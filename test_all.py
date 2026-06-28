import requests
import os

BASE_URL = "http://localhost:8000"
CHIP_1 = "data/raw/sen1floods11/subset/india/India_774689_S1Hand.tif"
CHIP_2 = "data/raw/sen1floods11/subset/india/India_80221_S1Hand.tif"

def main():
    print("Testing Vision")
    r = requests.post(f"{BASE_URL}/vision/infer", json={"chip_path": CHIP_1})
    print(r.status_code)
    print(r.json().get('status'), r.json().get('error_message'))
    
    print("Testing Analyze")
    r = requests.post(f"{BASE_URL}/analyze", json={"query": "Is there flooding?", "chip_path": CHIP_1})
    print(r.status_code)
    print(r.json().get('status'), r.json().get('error_message'))
    
    print("Testing Change Detection")
    r = requests.post(f"{BASE_URL}/vision/change-detection", json={"before_image": CHIP_1, "after_image": CHIP_2})
    print(r.status_code)
    print(r.json().get('status'), r.json().get('error_message'))

if __name__ == "__main__":
    main()
