import os
from app.agent.state import AgentState
from app.rag.indexer import QdrantIndexer

_indexer = None

def get_indexer() -> QdrantIndexer:
    global _indexer
    if _indexer is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        try:
            _indexer = QdrantIndexer(qdrant_url=qdrant_url)
        except Exception as e:
            print(f"Failed to init indexer: {e}")
    return _indexer

def analyze_policy(state: AgentState) -> dict:
    """
    Policy Agent node.
    """
    query = state.get("user_query", "")
    vision_result = state.get("vision_result")
    prediction_result = state.get("prediction_result")
    indexer = get_indexer()
    
    retrieved_chunks = []
    citations = []
    reasoning = []
    tools_used = ["QdrantIndexer"]
    
    queries_to_run = [query] if query else []
    
    if prediction_result:
        risk_level = prediction_result.get("risk_level", "Low")
        if risk_level in ["High", "Medium"]:
            reasoning.append(f"Risk level is {risk_level}. Automatically retrieving NDMA, National Disaster Management Plan, and World Bank guidance.")
            queries_to_run.append("NDMA guidance")
            queries_to_run.append("National Disaster Management Plan")
            queries_to_run.append("World Bank resilience guidance")
    elif vision_result and vision_result.get("flood_detected"):
        reasoning.append("Flooding detected by Vision Agent. Automatically retrieving NDMA and World Bank guidance.")
        queries_to_run.append("NDMA flood response guidance")
        queries_to_run.append("World Bank flood mitigation guidance")
    
    if indexer:
        try:
            for q in queries_to_run:
                if not q: continue
                reasoning.append(f"Searching policy documents for: '{q}'")
                results = indexer.search(q, limit=2)
                for res in results:
                    chunk_text = res.get("text", "")
                    if chunk_text not in retrieved_chunks:
                        retrieved_chunks.append(chunk_text)
                        citations.append({
                            "source": res.get("metadata", {}).get("source", "Unknown"),
                            "page": res.get("metadata", {}).get("page", "Unknown"),
                            "score": res.get("score", 0)
                        })
            reasoning.append(f"Found {len(retrieved_chunks)} relevant policy chunks.")
        except Exception as e:
            reasoning.append(f"Policy search failed: {e}")
    else:
        reasoning.append("Policy agent unavailable: QdrantIndexer not initialized.")
        
    return {
        "retrieved_chunks": retrieved_chunks,
        "citations": citations,
        "executed_agents": ["policy_agent"],
        "reasoning_path": reasoning,
        "tools_used": tools_used,
        "trace_log": ["Executed Policy Agent."]
    }
