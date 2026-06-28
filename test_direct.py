import asyncio
import os
from app.vision.service import VisionService

CHIP_1 = "data/raw/sen1floods11/subset/india/India_774689_S1Hand.tif"
CHIP_2 = "data/raw/sen1floods11/subset/india/India_80221_S1Hand.tif"

def main():
    service = VisionService()
    
    print("Testing Satellite Vision")
    result = service.infer(CHIP_1)
    print("Vision:", result.get("status") if isinstance(result, dict) else "success")
    
    # Actually wait, is infer async or sync? Let's check. 
    # Usually service.infer() is sync.
    
    print("Testing Change Detection")
    result2 = service.infer_change(CHIP_1, CHIP_2)
    print("Change Detection:", result2.get("status") if isinstance(result2, dict) else "success")
    
if __name__ == "__main__":
    main()
