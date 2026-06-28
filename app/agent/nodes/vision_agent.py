import os
from app.agent.state import AgentState
from app.vision.service import FloodSegmentationService

vision_service = FloodSegmentationService(model_path="models/checkpoints/best_model.pt")

def analyze_vision(state: AgentState) -> dict:
    """
    Vision Agent node.
    """
    chip_path = state.get("chip_path")
    
    # If spatial_result is available, use the first chip if chip_path is missing
    spatial_result = state.get("spatial_result", [])
    if not chip_path and spatial_result and "file_path" in spatial_result[0]:
        chip_path = spatial_result[0]["file_path"]
        
    trace_log = []
    reasoning = []
    tools_used = ["FloodSegmentationService"]
    
    if not chip_path:
        reasoning.append("Vision inference skipped: No chip_path available.")
        return {
            "vision_result": {"error": "No chip_path provided"},
            "executed_agents": ["vision_agent"],
            "reasoning_path": reasoning,
            "tools_used": tools_used,
            "trace_log": ["Vision skipped."]
        }
    
    if os.path.exists(chip_path):
        reasoning.append(f"Executing vision analysis on {chip_path}.")
        try:
            vision_result = vision_service.infer(chip_path)
            reasoning.append(f"Vision inference successful. Flood detected: {vision_result.get('flood_detected', False)}")
        except Exception as e:
            vision_result = {"error": str(e), "chip_path": chip_path}
            reasoning.append(f"Vision inference failed: {e}")
    else:
        vision_result = {"error": "Chip file not found", "chip_path": chip_path}
        reasoning.append(f"Vision inference aborted: File not found ({chip_path}).")
    
    return {
        "vision_result": vision_result,
        "executed_agents": ["vision_agent"],
        "reasoning_path": reasoning,
        "tools_used": tools_used,
        "trace_log": ["Executed Vision Agent."]
    }
