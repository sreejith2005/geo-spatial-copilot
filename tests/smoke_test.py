import requests
import sys
import os
import glob
import rasterio

BASE_URL = "http://localhost:8000"

def get_valid_chip_path():
    dirs_to_check = [
        "data/raw/sen1floods11/subset/india",
        "data/raw/sen1floods11/subset/spain"
    ]
    
    for d in dirs_to_check:
        search_path = os.path.join(d, "*_S1Hand.tif")
        files = glob.glob(search_path)
        for f in files:
            try:
                # Convert backslashes to forward slashes for cross-platform logic just in case
                f_norm = os.path.normpath(f)
                if not os.path.exists(f_norm):
                    continue
                with rasterio.open(f_norm) as src:
                    # just read meta to ensure it's fully accessible
                    meta = src.meta.copy()
                    if src.count != 2:
                        continue
                    return f_norm, meta
            except Exception:
                continue
    return None, None

def run_test(name, endpoint, payload, expected_status=200, expected_code=None):
    print(f"Running test: {name:.<45} ", end="")
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        
        if response.status_code == expected_status:
            data = response.json()
            if expected_code:
                if data.get("code") == expected_code:
                    print("PASS")
                    return True, data
                else:
                    print(f"FAIL (Expected code {expected_code}, got {data.get('code')})")
                    return False, data
            
            if expected_status == 200 and data.get("status") != "success":
                print(f"FAIL (API status: {data.get('status')} - {data.get('error_message', '')})")
                return False, data
                
            print("PASS")
            return True, data
        else:
            print(f"FAIL (Expected HTTP {expected_status}, got {response.status_code}. Response: {response.text})")
            return False, {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        print("FAIL (Connection Error: Backend not running?)")
        return False, {"error": "Connection Error"}
    except Exception as e:
        print(f"FAIL (Exception: {e})")
        return False, {"error": str(e)}

def main():
    print("=======================================")
    print("   Geospatial Copilot Smoke Tests      ")
    print("=======================================")
    
    # 1. Find a valid chip
    chip_path, chip_meta = get_valid_chip_path()
    if not chip_path:
        print("CRITICAL FAILURE: No valid S1Hand.tif chip found in data/raw/sen1floods11/subset/india or spain.")
        sys.exit(1)
        
    print(f"Selected Chip for Testing: {chip_path}")
            
    tests = [
        {
            "name": "Retrieve Only",
            "endpoint": "/retrieve",
            "payload": {"query": "flood risk procedures", "top_k": 1}
        },
        {
            "name": "Vision Only (Valid)",
            "endpoint": "/vision/infer",
            "payload": {"chip_path": chip_path}
        },
        {
            "name": "Vision Only (Invalid LabelHand)",
            "endpoint": "/vision/infer",
            "payload": {"chip_path": chip_path.replace("_S1Hand.tif", "_LabelHand.tif")},
            "expected_status": 400,
            "expected_code": "INVALID_SENTINEL_INPUT"
        },
        {
            "name": "Document Query (Analyze)",
            "endpoint": "/analyze",
            "payload": {"query": "What are the local flood policies?", "chip_path": None}
        },
        {
            "name": "Image Query (Analyze)",
            "endpoint": "/analyze",
            "payload": {"query": "Is there flooding in this image?", "chip_path": chip_path}
        },
        {
            "name": "Mixed Query (Analyze)",
            "endpoint": "/analyze",
            "payload": {"query": "Based on this image and policy, what is the risk?", "chip_path": chip_path}
        }
    ]
    
    results = []
    all_passed = True
    for test in tests:
        success, data = run_test(
            test["name"], 
            test["endpoint"], 
            test["payload"], 
            expected_status=test.get("expected_status", 200),
            expected_code=test.get("expected_code")
        )
        results.append({"name": test["name"], "status": "PASS" if success else "FAIL"})
        if not success:
            all_passed = False
            
    # Generate final test report
    print("\n=======================================")
    print("         FINAL TEST REPORT             ")
    print("=======================================")
    print(f"Selected Chip Path : {chip_path}")
    print(f"Raster Metadata    :")
    if chip_meta:
        for k, v in chip_meta.items():
            if k != 'transform':
                print(f"  - {k}: {v}")
    
    print("\nEndpoint Status    :")
    for res in results:
        print(f"  - {res['name']:<25}: {res['status']}")
        
    print("=======================================")
    if all_passed:
        print("RESULT: ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("RESULT: SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
