import sys
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agent.graph import graph_app
from app.agent.state import AgentState

def run_test_query(query_name: str, query_text: str):
    print(f"\n{'='*60}")
    print(f"TEST: {query_name}")
    print(f"QUERY: {query_text}")
    print(f"{'='*60}")
    
    # Initial state
    initial_state: AgentState = {
        "user_query": query_text,
        "query_type": None,
        "chip_path": None,
        "retrieved_chunks": [],
        "vision_result": None,
        "prediction_result": None,
        "citations": [],
        "final_answer": None,
        "errors": [],
        "trace_log": ["Started test run"]
    }
    
    # Run graph
    try:
        final_state = graph_app.invoke(initial_state)
        
        print("\n--- GRAPH TRACE LOG ---")
        for log_entry in final_state.get("trace_log", []):
            print(f"- {log_entry}")
            
        print("\n--- FINAL ANSWER ---")
        print(final_state.get("final_answer", "No answer generated."))
        
        print(f"\n{'='*60}")
    except Exception as e:
        print(f"Graph execution failed: {e}")

def main():
    queries = {
        "Document Question": "What is the policy for flood guidance?",
        "Flood Image Question": "Analyze this satellite imagery for flood detection.",
        "Risk Prediction Question": "What is the geospatial risk and forecast for this region?",
        "Mixed Question": "Check this satellite imagery for flood detection and give me the flood guidance policy."
    }
    
    for name, text in queries.items():
        run_test_query(name, text)

if __name__ == "__main__":
    main()
