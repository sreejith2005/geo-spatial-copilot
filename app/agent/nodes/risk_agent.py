from app.agent.state import AgentState
from app.core.timeline import log_assessment, get_trend_analysis

def analyze_risk(state: AgentState) -> dict:
    """
    Risk Agent node.
    Combines spatial data and vision data to predict risk.
    """
    reasoning = []
    tools_used = ["RiskPredictionModel"]
    
    spatial_result = state.get("spatial_result", [])
    vision_result = state.get("vision_result", {})
    
    reasoning.append("Evaluating risk factors based on spatial and vision inputs.")
    
    score = 50.0
    level = "Medium"
    factors = []
    
    if vision_result and vision_result.get("flood_detected"):
        score += 30.0
        factors.append("Active flood detected in imagery")
        
    if spatial_result:
        # Just stubbing risk based on arbitrary region rules for demo
        # Evaluate country-level risk once based on the primary chip
        primary_sp = spatial_result[0]
        if str(primary_sp.get("country", "")).lower() == "in":
            score += 15.0
            factors.append("Region has historical high susceptibility (India)")
        if str(primary_sp.get("country", "")).lower() == "es":
            score -= 10.0
            factors.append("Region has advanced mitigation infrastructure (Spain)")
                
    if not factors:
        factors.append("Baseline historical risk")
        
    if score > 75:
        level = "High"
    elif score < 40:
        level = "Low"
        
    prediction_result = {
        "risk_score": min(score, 100.0),
        "risk_level": level,
        "affected_area_sqkm": spatial_result[0].get("area_sqm", 0) / 1e6 if spatial_result else 0.0,
        "key_factors": factors
    }
    
    # Log to timeline
    chip_path = state.get("chip_path") or ""
    district = spatial_result[0].get("district", "") if spatial_result else ""
    state_loc = spatial_result[0].get("state", "") if spatial_result else ""
    
    try:
        log_assessment(chip_path, level, prediction_result["risk_score"], district, state_loc)
        trend = get_trend_analysis(district)
        prediction_result["trend_analysis"] = trend
        reasoning.append(f"Trend analysis: {trend}")
    except Exception as e:
        reasoning.append(f"Failed to log assessment to timeline: {e}")
        prediction_result["trend_analysis"] = "Trend analysis unavailable."
    
    reasoning.append(f"Calculated risk level: {level} (Score: {prediction_result['risk_score']})")
    
    return {
        "prediction_result": prediction_result,
        "executed_agents": ["risk_agent"],
        "reasoning_path": reasoning,
        "tools_used": tools_used,
        "trace_log": ["Executed Risk Agent."]
    }

