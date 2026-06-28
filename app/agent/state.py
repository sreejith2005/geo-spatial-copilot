from typing import TypedDict, List, Dict, Any, Optional
import operator
from typing import Annotated

class AgentState(TypedDict):
    user_query: str
    query_type: Optional[str]
    intent_analysis: Optional[Dict[str, str]]
    chip_path: Optional[str]
    retrieved_chunks: List[str]
    vision_result: Optional[Dict[str, Any]]
    prediction_result: Optional[Dict[str, Any]]
    spatial_result: Optional[List[Dict[str, Any]]]
    citations: List[Dict[str, Any]]
    final_answer: Optional[str]
    errors: List[str]
    memory: Optional[Dict[str, Any]]
    conversation_id: Optional[str]
    evidence_confidence: Optional[Dict[str, Any]]
    plan: List[str]
    executed_agents: Annotated[List[str], operator.add]
    reasoning_path: Annotated[List[str], operator.add]
    tools_used: Annotated[List[str], operator.add]
    trace_log: Annotated[List[str], operator.add]
