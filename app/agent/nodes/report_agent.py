import json
from langchain_core.messages import HumanMessage, SystemMessage
from app.agent.state import AgentState
from app.llm.provider import get_llm

def generate_report(state: AgentState) -> dict:
    """
    Report Agent node.
    Synthesizes intelligence into a Geospatial Intelligence Brief using the configured LLM.
    Required sections: Executive Summary, Assessment, Evidence, Recommendations, Sources.
    """
    query = state.get("user_query", "")
    retrieved_chunks = state.get("retrieved_chunks") or []
    vision_result = state.get("vision_result")
    prediction_result = state.get("prediction_result")
    spatial_result = state.get("spatial_result")
    citations = state.get("citations") or []
    intent_analysis = state.get("intent_analysis") or {}
    
    reasoning = ["Synthesizing intelligence into Geospatial Intelligence Brief using LLM."]
    tools_used = ["IntelligenceBriefGenerator"]
    
    retrieved_chunk_count = len(retrieved_chunks)
    retrieved_token_count = sum(len(c) for c in retrieved_chunks) // 4
    
    if retrieved_chunk_count == 0:
        return {
            "final_answer": "The current knowledge base does not contain sufficient evidence to answer this question confidently.",
            "executed_agents": ["report_agent"],
            "reasoning_path": reasoning + ["Skipped generation: No retrieved evidence."],
            "tools_used": tools_used,
            "trace_log": [
                "Executed Report Agent.", 
                f"Retrieved chunks: {retrieved_chunk_count}", 
                f"Retrieved tokens: {retrieved_token_count}", 
                "Skipped generation due to lack of evidence."
            ]
        }
        
    avg_similarity_score = 0
    if citations:
        scores = [c.get("score", 0) for c in citations if isinstance(c.get("score"), (int, float))]
        if scores:
            avg_similarity_score = sum(scores) / len(scores)

    evidence_confidence = {
        "retrieval_quality": "High" if avg_similarity_score > 0.75 else ("Medium" if avg_similarity_score > 0.5 else "Low"),
        "similarity_score": avg_similarity_score,
        "supporting_chunks": retrieved_chunk_count
    }
    
    context = []
    if vision_result:
        context.append(f"Vision Analysis: {json.dumps(vision_result)}")
    if prediction_result:
        context.append(f"Risk Prediction: {json.dumps(prediction_result)}")
    if spatial_result:
        context.append(f"Spatial Data: {json.dumps(spatial_result)}")
        if isinstance(spatial_result, list) and len(spatial_result) > 0:
            district = spatial_result[0].get("district", "Unknown")
            state_loc = spatial_result[0].get("state", "Unknown")
            context.append(f"Location: {district}, {state_loc}")
    if retrieved_chunks:
        context.append("Retrieved Policy Documents:")
        for i, chunk in enumerate(retrieved_chunks):
            context.append(f"--- Chunk {i+1} ---\n{chunk}")
            
    if citations:
        context.append("Citations:")
        for i, cit in enumerate(citations):
            context.append(f"Source {i+1}: {cit.get('source', 'Unknown')} (Page: {cit.get('page', 'Unknown')})")
            
    memory = state.get("memory", {})
    if memory:
        if memory.get("history"):
            context.append(f"Conversational History: {memory['history']}")
        if memory.get("previous_recommendations"):
            context.append(f"Previous Recommendations: {memory['previous_recommendations']}")
        if memory.get("recent_analyses"):
            context.append(f"Recent Analyses: {memory['recent_analyses']}")

    query_lower = query.lower()
    response_mode = "CHAT"
    decision_keywords = ["where should i", "which", "compare", "vs", "versus", "better option", "recommend", "which option"]
    if any(kw in query_lower for kw in decision_keywords):
        response_mode = "DECISION"
    elif any(kw in query_lower for kw in ["full report", "detailed analysis", "intelligence brief", "show details"]):
        response_mode = "REPORT"
    elif "brief" in query_lower:
        response_mode = "BRIEF"

    rag_strict_instructions = (
        "\nYou are not a general assistant.\n"
        "You are an analyst working ONLY from the supplied evidence.\n"
        "Do not introduce facts not present in the retrieved context.\n"
    )

    xai_instructions = (
        rag_strict_instructions +
        "\nExplainable AI Requirements:\n"
        "Every recommendation should include: Why?\n"
        "The explanation should reference: Vision evidence, Spatial evidence, Policy evidence, and Historical evidence.\n"
        "Format:\n"
        "Recommendation\n"
        "Why I recommend this\n"
        "Confidence\n"
        "Never expose internal chain-of-thought.\n"
        "Only expose concise evidence-based reasoning."
    )

    system_prompts = {
        "CHAT": (
            "You are a conversational AI assistant and Senior Geospatial Intelligence Analyst. Behave naturally, like ChatGPT or Claude, rather than a government report generator.\n"
            "Synthesize the provided telemetry, vision results, and policy documents to answer the user's query.\n"
            "Guidelines for CHAT mode:\n"
            "1. Answer the user's question directly in the first sentence.\n"
            "2. Keep the response between 2 to 5 sentences. Keep responses under 120 words.\n"
            "3. Tone must be concise, natural, conversational, and actionable.\n"
            "4. End your response with one relevant follow-up question whenever appropriate.\n"
            "5. Never generate large reports or use large multi-section headings unless explicitly requested.\n"
            "6. Preserve internal source citations.\n"
            "7. Never show raw JSON or raw chunks to end users."
            + xai_instructions
        ),
        "BRIEF": (
            "You are a natural, conversational AI assistant and Senior Geospatial Intelligence Analyst.\n"
            "Provide a very short, bulleted brief answering the user's query based on the context.\n"
            "Guidelines:\n"
            "1. Answer the user's question directly in the first sentence.\n"
            "2. Keep it extremely concise (1-3 bullet points) and conversational. Keep responses under 120 words.\n"
            "3. Preserve internal source citations.\n"
            "4. End with one relevant follow-up question whenever appropriate.\n"
            "5. Never show raw JSON or raw chunks to end users."
            + xai_instructions
        ),
        "REPORT": (
            "You are a conversational AI assistant and Senior Geospatial Intelligence Analyst. Behave naturally, like ChatGPT or Claude.\n"
            "Synthesize the provided telemetry, vision results, and policy documents into a cohesive Geospatial Intelligence Brief answering the user's query.\n"
            "Even though this is a report, maintain a natural and helpful tone.\n"
            "You must structure your response with exactly these markdown headings:\n"
            "# Executive Summary\n"
            "# Situation Assessment\n"
            "# Flood Analysis\n"
            "# Risk Evaluation\n"
            "# Recommended Actions\n"
            "# Supporting Evidence\n\n"
            "Guidelines:\n"
            "1. Answer the user's question directly in the first section.\n"
            "2. End the report with one relevant follow-up question whenever appropriate.\n"
            "3. Preserve internal source citations.\n"
            "4. Never show raw JSON to end users.\n"
            "5. Never show raw chunks to end users.\n"
            "6. Use retrieved chunks only as evidence.\n"
            "7. Synthesize all information and adopt a professional yet conversational intelligence briefing tone."
            + xai_instructions
        ),
        "DECISION": (
            "You are a conversational AI assistant and Senior Geospatial Intelligence Analyst.\n"
            "The user is trying to make a decision or choose between alternatives.\n"
            "Instead of only answering questions, generate a recommendation table.\n"
            "Compare the options using exactly these columns: Flood Risk, Historical Risk, Policy Guidance, Confidence, Advantages, Disadvantages, Overall Recommendation.\n"
            "Format the table in markdown.\n"
            "Guidelines:\n"
            "1. Conclude your response with exactly: \"I recommend Option [X] because...\" (replace [X] with the recommended option).\n"
            "2. Preserve internal source citations.\n"
            "3. Never show raw JSON or raw chunks to end users."
            + xai_instructions
        )
    }

    system_prompt = system_prompts[response_mode]
    
    if intent_analysis:
        system_prompt += (
            f"\n\nIntent Analysis:\n"
            f"- Primary Intent: {intent_analysis.get('primary_intent', 'N/A')}\n"
            f"- Secondary Intent: {intent_analysis.get('secondary_intent', 'N/A')}\n"
            f"- User Goal: {intent_analysis.get('user_goal', 'N/A')}\n\n"
            "CRITICAL INSTRUCTION: Ensure your recommendations and responses directly support the user's stated objective based on the Intent Analysis. "
            "Do not focus solely on flood information. Focus on helping the user achieve their broader goal."
        )
    
    llm = get_llm()
    if llm:
        # Response Compression Agent Step
        compression_system_prompt = (
            "You are a Response Compression Agent.\n"
            "Your task is to filter the provided context based on the user's query before final response generation.\n"
            "Requirements:\n"
            "1. Determine what information actually answers the user's question.\n"
            "2. Remove irrelevant sections (e.g. do NOT include chip analysis, Nagaon flood percentage, confidence scores, local satellite observations unless explicitly relevant to the query).\n"
            "3. Include only information needed to answer the query.\n"
            "4. Provide Minimal Relevant Context instead of All Available Context.\n"
            "5. Target a 70% reduction in context size while preserving answer quality.\n"
            "6. You MUST preserve the '--- Chunk X ---' markers for any chunks you keep.\n"
            "Output ONLY the compressed context. Do not attempt to answer the user's query yourself."
        )
        compression_user_prompt = f"User Query: {query}\n\nAll Available Context:\n" + "\n\n".join(context)
        
        try:
            compression_response = llm.invoke([SystemMessage(content=compression_system_prompt), HumanMessage(content=compression_user_prompt)])
            compressed_context = compression_response.content
            reasoning.append("Context successfully compressed by Response Compression Agent.")
        except Exception as e:
            reasoning.append(f"Context compression failed: {e}. Using original context.")
            compressed_context = "\n\n".join(context)

        user_prompt = f"User Query: {query}\n\nMinimal Relevant Context:\n{compressed_context}"
        
        used_chunks_count = compressed_context.count("--- Chunk")
        final_prompt_size = (len(system_prompt) + len(user_prompt)) // 4
        
        try:
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            final_answer = response.content
            reasoning.append("Mission Brief successfully generated via LLM.")
            reasoning.append(f"RAG Logging - Retrieved chunks: {retrieved_chunk_count} | Used chunks: {used_chunks_count} | Final prompt size: {final_prompt_size}")
        except Exception as e:
            reasoning.append(f"LLM generation failed: {e}. Falling back to error response.")
            final_answer = f"# Executive Summary\nLLM Generation Failed: {e}"
    else:
        used_chunks_count = 0
        final_prompt_size = 0
        reasoning.append("No LLM provider available. Falling back to error response.")
        final_answer = "# Executive Summary\nNo LLM configured in the system."
        
    trace_log = [
        "Executed Report Agent.",
        f"Retrieved chunks: {retrieved_chunk_count}",
        f"Chunks actually used: {used_chunks_count}",
        f"Final prompt size: {final_prompt_size}"
    ]
        
    return {
        "final_answer": final_answer,
        "evidence_confidence": evidence_confidence,
        "executed_agents": ["report_agent"],
        "reasoning_path": reasoning,
        "tools_used": tools_used,
        "trace_log": trace_log
    }
