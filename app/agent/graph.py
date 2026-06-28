from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes.planner_agent import generate_plan, route_from_planner
from app.agent.nodes.spatial_agent import analyze_spatial
from app.agent.nodes.vision_agent import analyze_vision
from app.agent.nodes.policy_agent import analyze_policy
from app.agent.nodes.risk_agent import analyze_risk
from app.agent.nodes.report_agent import generate_report

def build_graph():
    """
    Builds the stateful LangGraph workflow for the Geospatial Copilot.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner_agent", generate_plan)
    workflow.add_node("spatial_agent", analyze_spatial)
    workflow.add_node("vision_agent", analyze_vision)
    workflow.add_node("policy_agent", analyze_policy)
    workflow.add_node("risk_agent", analyze_risk)
    workflow.add_node("report_agent", generate_report)
    
    # Define edges sequentially to resolve data dependencies
    workflow.add_edge(START, "planner_agent")
    # Conditional routing to specialized agents or END
    workflow.add_conditional_edges(
        "planner_agent",
        route_from_planner,
        {
            "UNIFIED": "spatial_agent",
            "OUT_OF_SCOPE": END
        }
    )
    
    # Sequential fall-throughs for data dependencies
    workflow.add_edge("spatial_agent", "vision_agent")
    workflow.add_edge("vision_agent", "risk_agent")
    workflow.add_edge("risk_agent", "policy_agent")
    workflow.add_edge("policy_agent", "report_agent")
    workflow.add_edge("report_agent", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

# Export a compiled instance
graph_app = build_graph()
