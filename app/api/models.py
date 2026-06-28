from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="The user's input query.")
    chip_path: Optional[str] = Field(default=None, description="Path to the satellite image chip (optional).")
    mode: Optional[str] = Field(default=None, description="Optional forced mode for routing (e.g., 'vision', 'rag', 'predict').")
    conversation_id: str = Field(default="default", description="Identifier for conversational memory.")

class AnalyzeResponse(BaseModel):
    answer: Optional[str] = Field(default=None, description="The final answer text.")
    summary: Optional[str] = Field(default=None, description="Synthesized final answer or report.")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Formatted list of sources for UI.")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of cited documents.")
    vision_result: Optional[Dict[str, Any]] = Field(default=None, description="Results from vision analysis.")
    prediction_result: Optional[Dict[str, Any]] = Field(default=None, description="Results from risk prediction.")
    executed_agents: List[str] = Field(default_factory=list, description="Agents executed during reasoning.")
    reasoning_path: List[str] = Field(default_factory=list, description="Log of reasoning path.")
    tools_used: List[str] = Field(default_factory=list, description="Tools used during execution.")
    trace_log: List[str] = Field(default_factory=list, description="Trace of the agent's execution path.")
    execution_time: Optional[float] = Field(default=None, description="Total execution time in seconds.")
    status: str = Field(..., description="Status of the request (e.g., 'success', 'error').")
    error_message: Optional[str] = Field(default=None, description="Detailed error message if status is 'error'.")

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="The search query.")

class RetrieveResponse(BaseModel):
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="List of retrieved chunks and scores.")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata of retrieved chunks.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")

class VisionInferRequest(BaseModel):
    chip_path: str = Field(..., description="Path to the satellite image chip.")

class VisionInferResponse(BaseModel):
    vision_result: Optional[Dict[str, Any]] = Field(default=None, description="Structured inference results.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service alive status.")

class ReadyResponse(BaseModel):
    status: str = Field(..., description="Overall readiness status.")
    qdrant_status: str = Field(..., description="Status of Qdrant connection.")
    vision_status: str = Field(..., description="Status of the vision service.")
    graph_status: str = Field(..., description="Status of the LangGraph app.")

class ChipsResponse(BaseModel):
    chips: List[Dict[str, Any]] = Field(default_factory=list, description="List of available chips with metadata.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")

class ChipDetailsResponse(BaseModel):
    chip_id: str = Field(..., description="ID of the chip.")
    metadata: Dict[str, Any] = Field(..., description="Chip metadata.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")
    district: Optional[str] = Field(default=None, description="District name.")
    state: Optional[str] = Field(default=None, description="State name.")
    country: Optional[str] = Field(default=None, description="Country code.")
    area_km2: Optional[float] = Field(default=None, description="Area in square kilometers.")
    centroid: Optional[List[float]] = Field(default=None, description="Centroid [lon, lat].")
    bounds: Optional[List[float]] = Field(default=None, description="Bounding box [minx, miny, maxx, maxy].")

class VisionDebugRequest(BaseModel):
    chip_path: str = Field(..., description="Path to the satellite image chip.")

class VisionDebugResponse(BaseModel):
    vision_result: Optional[Dict[str, Any]] = Field(default=None, description="Structured inference results.")
    explainability_outputs: Optional[Dict[str, str]] = Field(default=None, description="Paths to generated explainability artifacts.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")

class ChangeDetectionRequest(BaseModel):
    before_image: str = Field(..., description="Path to the before satellite image chip.")
    after_image: str = Field(..., description="Path to the after satellite image chip.")

class ChangeDetectionResponse(BaseModel):
    changed_area: Optional[float] = Field(default=None, description="Percentage of area that changed.")
    flood_expansion: Optional[float] = Field(default=None, description="Percentage of flood expansion.")
    flood_reduction: Optional[float] = Field(default=None, description="Percentage of flood reduction.")
    change_percentage: Optional[float] = Field(default=None, description="Percentage of total change.")
    change_mask_path: Optional[str] = Field(default=None, description="Path to the generated change mask.")
    status: str = Field(..., description="Status of the request.")
    error_message: Optional[str] = Field(default=None, description="Error message if applicable.")

class ClearMemoryResponse(BaseModel):
    status: str = Field(..., description="Status of the memory clear request.")
    message: str = Field(..., description="Message detailing the result.")
