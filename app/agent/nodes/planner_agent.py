import os
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent.state import AgentState
from app.llm.provider import get_llm

def generate_plan(state: AgentState) -> dict:
    """
    Planner Agent node.
    Analyzes the user query and determines which specialized agents need to be invoked.
    """
    query = state.get("user_query", "").lower()
    
    plan = []
    intent_analysis = {}
    
    in_scope_keywords = [
        "flood", "disaster", "climate", "resilience", "geospatial", 
        "satellite", "water", "risk", "mitigation", "vision", "imagery",
        "spatial", "where", "country", "area"
    ]
    
    policy_keywords = ["document", "flood guidance", "policy", "reports", "guidelines", "mitigation planning"]
    vision_keywords = ["satellite imagery", "land cover", "flood detection", "image", "vision", "analyze imagery"]
    risk_keywords = ["risk", "suitability", "forecast", "prediction", "flood-prone"]

    routes = []

    memory = state.get("memory", {})
    memory_context = ""
    if memory:
        memory_context = f"\n\nConversational Memory:\n- History: {memory.get('history', [])}\n- Current Objective: {memory.get('current_objective', 'N/A')}\n- Industry: {memory.get('industry', 'N/A')}\n- Current Region: {memory.get('current_region', 'N/A')}"

    # Attempt to use LLM for semantic routing
    llm = get_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=f"""You are a LangGraph router for a Geospatial Risk Copilot.
Determine the appropriate routes for this query. Use the Conversational Memory if the query lacks context (e.g., "What about Gujarat?" implicitly relates to the Current Objective).
Valid routes: UNIFIED, OUT_OF_SCOPE.

If the query and memory are UNRELATED to floods, disaster management, climate resilience, geospatial risk, or satellite analysis (e.g. "clothing factory", "general knowledge"), you MUST return ONLY "OUT_OF_SCOPE".

Otherwise, return ONLY "UNIFIED".{memory_context}""")
            response = llm.invoke([sys_msg, HumanMessage(content=query)])
            llm_routes = [r.strip() for r in response.content.split(",")]
            for r in llm_routes:
                if r in ["UNIFIED", "OUT_OF_SCOPE"]:
                    routes.append(r)
        except Exception:
            pass
            
        # Intent Extraction
        try:
            intent_sys_msg = SystemMessage(content=f"""You are an Intent Extraction agent.
Analyze the user query and the conversational memory. Extract the user intent, integrating memory if the query is a follow-up.
Extract:
1. primary_intent (e.g., Factory location selection)
2. secondary_intent (e.g., Flood avoidance)
3. user_goal (e.g., To find a safe and suitable location for building a factory)
4. current_objective (e.g., Build a textile factory)
5. industry (e.g., Textile)
6. current_region (e.g., Gujarat)
Return the output strictly as a JSON object with keys: primary_intent, secondary_intent, user_goal, current_objective, industry, current_region. Do not include markdown blocks.{memory_context}""")
            intent_response = llm.invoke([intent_sys_msg, HumanMessage(content=query)])
            import json
            import re
            match = re.search(r"\{.*\}", intent_response.content, re.DOTALL)
            if match:
                intent_analysis = json.loads(match.group(0))
        except Exception:
            pass

    # Fallback to heuristic
    if not routes:
        is_in_scope = any(kw in query for kw in in_scope_keywords) or any(kw in query for kw in policy_keywords + vision_keywords + risk_keywords) or state.get("chip_path")
        
        if not is_in_scope:
            routes.append("OUT_OF_SCOPE")
        else:
            routes.append("UNIFIED")

    if "OUT_OF_SCOPE" in routes:
        plan.append("OUT_OF_SCOPE")
        return {
            "plan": plan,
            "query_type": "OUT_OF_SCOPE",
            "executed_agents": ["planner_agent"],
            "reasoning_path": ["Routed to OUT_OF_SCOPE. Query is unrelated to geospatial risk."],
            "tools_used": ["LangGraph Routing"],
            "final_answer": "I'm sorry, but I can only answer questions related to flood intelligence, disaster management, climate resilience, geospatial risk, and satellite analysis. Your query appears to be outside my current knowledge base.",
            "intent_analysis": intent_analysis
        }

    plan.extend(routes)
    return {
        "plan": plan,
        "executed_agents": ["planner_agent"],
        "reasoning_path": [f"Selected routes: {routes}"],
        "tools_used": ["LangGraph Routing"],
        "intent_analysis": intent_analysis
    }

def route_from_planner(state: AgentState) -> str:
    """
    Conditional edge router that reads the plan and returns the next route.
    """
    plan = state.get("plan", [])
    if "OUT_OF_SCOPE" in plan:
        return "OUT_OF_SCOPE"
    return "UNIFIED"
