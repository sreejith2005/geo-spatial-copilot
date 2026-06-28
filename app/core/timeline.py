import os
import pandas as pd
from datetime import datetime

TIMELINE_FILE = "data/interim/risk_timeline.csv"

def _ensure_timeline_exists():
    os.makedirs(os.path.dirname(TIMELINE_FILE), exist_ok=True)
    if not os.path.exists(TIMELINE_FILE):
        df = pd.DataFrame(columns=["timestamp", "chip_path", "severity", "risk_score", "district", "state"])
        df.to_csv(TIMELINE_FILE, index=False)

def log_assessment(chip_path: str, severity: str, risk_score: float, district: str = "", state: str = ""):
    _ensure_timeline_exists()
    new_record = {
        "timestamp": datetime.now().isoformat(),
        "chip_path": chip_path or "Unknown",
        "severity": severity,
        "risk_score": risk_score,
        "district": district or "Unknown",
        "state": state or "Unknown"
    }
    df = pd.DataFrame([new_record])
    df.to_csv(TIMELINE_FILE, mode='a', header=False, index=False)

def get_timeline() -> list:
    _ensure_timeline_exists()
    df = pd.read_csv(TIMELINE_FILE)
    return df.to_dict(orient="records")

def get_trend_analysis(district: str) -> str:
    _ensure_timeline_exists()
    df = pd.read_csv(TIMELINE_FILE)
    if district and district != "Unknown":
        df = df[df["district"] == district]
        
    if len(df) < 2:
        return "Insufficient historical data for trend analysis."
        
    df = df.sort_values(by="timestamp")
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    
    diff = latest["risk_score"] - previous["risk_score"]
    dist_name = district if (district and district != "Unknown") else "the area"
    if diff > 5:
        return f"Risk score has increased significantly (+{diff:.1f}) since the last assessment in {dist_name}."
    elif diff < -5:
        return f"Risk score has decreased significantly ({diff:.1f}) since the last assessment in {dist_name}."
    else:
        return f"Risk score in {dist_name} remains relatively stable."
