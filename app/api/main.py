import time
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError
import os
import rasterio

class InvalidSentinelInputError(Exception):
    pass

from app.api.models import (
    AnalyzeRequest, AnalyzeResponse,
    RetrieveRequest, RetrieveResponse,
    VisionInferRequest, VisionInferResponse,
    HealthResponse, ReadyResponse, ChipsResponse, ChipDetailsResponse,
    VisionDebugRequest, VisionDebugResponse,
    ChangeDetectionRequest, ChangeDetectionResponse,
    ClearMemoryResponse
)

from app.agent.graph import graph_app
from app.agent.state import AgentState
from app.rag.indexer import QdrantIndexer
from app.vision.service import FloodSegmentationService
from app.llm.provider import get_llm_status, get_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Geospatial Copilot API",
    description="Hardened API for the Geospatial Copilot Agent",
    version="1.1.0"
)

# Global services (initialized lazily or mocked if not available)
qdrant_indexer: Optional[QdrantIndexer] = None
vision_service: Optional[FloodSegmentationService] = None
llm_generation_working: bool = False

# Conversational Memory Store
CONVERSATIONS: dict[str, dict] = {}

@app.on_event("startup")
async def startup_event():
    global qdrant_indexer, vision_service, llm_generation_working
    logger.info("Initializing global services...")
    
    # Perform a real test call
    try:
        llm = get_llm()
        if llm:
            logger.info("Performing generation test call...")
            res = llm.invoke("Reply with the word ACTIVE")
            logger.info(f"Generation test passed: {res.content}")
            llm_generation_working = True
        else:
            logger.warning("No LLM available to test.")
    except Exception as e:
        logger.error(f"Generation test failed: {e}")
        llm_generation_working = False
        
    llm_status = get_llm_status()
    logger.info(f"LLM Provider: {llm_status.get('provider')}")
    logger.info(f"Model Name: {llm_status.get('model')}")
    logger.info(f"GOOGLE_API_KEY detected: {llm_status.get('google_api_key_present')}")
    logger.info(f"GEMINI_API_KEY detected: {llm_status.get('gemini_api_key_present')}")
    logger.info(f"Generation test passed: {llm_generation_working}")
    
    try:
        qdrant_indexer = QdrantIndexer()
    except Exception as e:
        logger.error(f"Failed to initialize QdrantIndexer: {e}")
        
    try:
        # Load vision model if path is provided, else dummy mode
        # The model path could be parameterized, using default dummy for now
        vision_service = FloodSegmentationService(model_path="models/checkpoints/best_model.pt")
    except Exception as e:
        logger.error(f"Failed to initialize FloodSegmentationService: {e}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Attach request_id to request state
    request.state.request_id = request_id
    
    logger.info(f"Request started: {request_id} - {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request_id} - {request.method} {request.url.path} - Status: {response.status_code} - Latency: {process_time:.4f}s")
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request_id} - {request.method} {request.url.path} - Latency: {process_time:.4f}s - Error: {str(e)}")
        raise e

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "error_message": f"Validation Error: {exc.errors()}"}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error_message": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "error_message": f"Internal Server Error: {str(exc)}"}
    )

@app.exception_handler(InvalidSentinelInputError)
async def invalid_sentinel_input_handler(request: Request, exc: InvalidSentinelInputError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "code": "INVALID_SENTINEL_INPUT",
            "message": "Expected Sentinel-1 S1Hand.tif with VV and VH bands"
        }
    )

def validate_chip_path(chip_path: str):
    chip_path = chip_path.replace("\\", "/")
    logger.info(f"Validating chip path: {chip_path}")
    if not os.path.exists(chip_path):
        logger.error(f"Validation failed: Chip path does not exist: {chip_path}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Chip path does not exist: {chip_path}")
        
    chip_path_lower = chip_path.lower()
    if not chip_path_lower.endswith(".tif"):
        logger.error(f"Validation failed: Not a .tif file: {chip_path}")
        raise InvalidSentinelInputError()
        
    if "labelhand" in chip_path_lower:
        logger.error(f"Validation failed: LabelHand.tif is not allowed: {chip_path}")
        raise InvalidSentinelInputError()
        
    if "s1hand" not in chip_path_lower:
        logger.error(f"Validation failed: Expected S1Hand.tif: {chip_path}")
        raise InvalidSentinelInputError()
        
    try:
        with rasterio.open(chip_path) as src:
            if src.count != 2:
                logger.error(f"Validation failed: Expected 2 bands, got {src.count} for {chip_path}")
                raise InvalidSentinelInputError()
    except Exception as e:
        if isinstance(e, InvalidSentinelInputError):
            raise
        logger.error(f"Validation failed: Failed to open raster file: {str(e)}")
        raise InvalidSentinelInputError()
        
    logger.info(f"Validation successful for: {chip_path}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")

@app.get("/system/llm-status")
async def get_system_llm_status():
    status = get_llm_status()
    status["generation_working"] = llm_generation_working
    return status

@app.get("/ready", response_model=ReadyResponse)
async def ready_check():
    q_status = "ok" if qdrant_indexer else "error"
    v_status = "ok" if vision_service else "error"
    g_status = "ok" if graph_app else "error"
    
    overall_status = "ready" if all(s == "ok" for s in [q_status, v_status, g_status]) else "not_ready"
    
    return ReadyResponse(
        status=overall_status,
        qdrant_status=q_status,
        vision_status=v_status,
        graph_status=g_status
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Endpoint to invoke the LangGraph copilot.
    """
    start_time = time.time()
    logger.info(f"Received /analyze request. Query: '{request.query}', Chip Path: '{request.chip_path}'")
    
    if request.chip_path:
        request.chip_path = request.chip_path.replace("\\", "/")
        validate_chip_path(request.chip_path)
    
    # Fetch conversational memory
    conv_id = request.conversation_id
    if conv_id not in CONVERSATIONS:
        CONVERSATIONS[conv_id] = {
            "history": [],
            "current_objective": None,
            "industry": None,
            "current_region": None,
            "previous_recommendations": [],
            "recent_analyses": []
        }
    
    memory = CONVERSATIONS[conv_id]
    
    # Construct the initial state
    initial_state: AgentState = {
        "user_query": request.query,
        "query_type": None,
        "chip_path": request.chip_path,
        "retrieved_chunks": [],
        "vision_result": None,
        "prediction_result": None,
        "spatial_result": None,
        "citations": [],
        "final_answer": None,
        "errors": [],
        "memory": memory,
        "conversation_id": conv_id,
        "plan": [],
        "executed_agents": [],
        "reasoning_path": [],
        "tools_used": [],
        "trace_log": []
    }
    
    try:
        # Invoke the compiled graph
        result_state = graph_app.invoke(initial_state)
        
        raw_citations = result_state.get("citations") or []
        retrieved_chunks = result_state.get("retrieved_chunks") or []
        formatted_sources = []
        for cit, chunk in zip(raw_citations, retrieved_chunks):
            source_path = cit.get("source", "Unknown")
            basename = os.path.basename(source_path.replace("\\", "/"))
            name, _ = os.path.splitext(basename)
            name = name.replace("_", " ")
            formatted_sources.append({
                "name": name,
                "page": cit.get("page", "N/A"),
                "snippet": chunk.strip(),
                "score": cit.get("score", 0.0)
            })
        
        errors = result_state.get("errors", [])
        
        # Update memory if successful
        if not errors:
            intent = result_state.get("intent_analysis", {})
            if intent:
                if intent.get("current_objective"): CONVERSATIONS[conv_id]["current_objective"] = intent["current_objective"]
                if intent.get("industry"): CONVERSATIONS[conv_id]["industry"] = intent["industry"]
                if intent.get("current_region"): CONVERSATIONS[conv_id]["current_region"] = intent["current_region"]
            
            answer = result_state.get("final_answer", "")
            CONVERSATIONS[conv_id]["history"].append(f"User: {request.query}")
            CONVERSATIONS[conv_id]["history"].append(f"Agent: {answer}")
            
            # Store only last 10 interactions (20 messages)
            if len(CONVERSATIONS[conv_id]["history"]) > 20:
                CONVERSATIONS[conv_id]["history"] = CONVERSATIONS[conv_id]["history"][-20:]
                
            if result_state.get("vision_result") or result_state.get("spatial_result"):
                CONVERSATIONS[conv_id]["recent_analyses"].append("Performed spatial/vision analysis.")
            
            if "recommend" in answer.lower():
                CONVERSATIONS[conv_id]["previous_recommendations"].append(answer[:200] + "...")
                
            if len(CONVERSATIONS[conv_id]["recent_analyses"]) > 5:
                CONVERSATIONS[conv_id]["recent_analyses"] = CONVERSATIONS[conv_id]["recent_analyses"][-5:]
            if len(CONVERSATIONS[conv_id]["previous_recommendations"]) > 5:
                CONVERSATIONS[conv_id]["previous_recommendations"] = CONVERSATIONS[conv_id]["previous_recommendations"][-5:]

        if errors:
            return AnalyzeResponse(
                answer=result_state.get("final_answer"),
                summary=result_state.get("final_answer"),
                sources=formatted_sources,
                citations=result_state.get("citations"),
                vision_result=result_state.get("vision_result"),
                prediction_result=result_state.get("prediction_result"),
                executed_agents=result_state.get("executed_agents", []),
                reasoning_path=result_state.get("reasoning_path", []),
                tools_used=result_state.get("tools_used", []),
                trace_log=result_state.get("trace_log", []),
                execution_time=time.time() - start_time,
                status="error",
                error_message="; ".join(errors)
            )
            
        return AnalyzeResponse(
            answer=result_state.get("final_answer"),
            summary=result_state.get("final_answer"),
            sources=formatted_sources,
            citations=result_state.get("citations"),
            vision_result=result_state.get("vision_result"),
            prediction_result=result_state.get("prediction_result"),
            executed_agents=result_state.get("executed_agents", []),
            reasoning_path=result_state.get("reasoning_path", []),
            tools_used=result_state.get("tools_used", []),
            trace_log=result_state.get("trace_log", []),
            execution_time=time.time() - start_time,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        return AnalyzeResponse(
            trace_log=[],
            execution_time=time.time() - start_time,
            status="error",
            error_message=f"Graph execution failed: {str(e)}"
        )

@app.delete("/memory/{conversation_id}", response_model=ClearMemoryResponse)
async def clear_memory(conversation_id: str):
    if conversation_id in CONVERSATIONS:
        del CONVERSATIONS[conversation_id]
        return ClearMemoryResponse(status="success", message=f"Memory for conversation '{conversation_id}' cleared.")
    return ClearMemoryResponse(status="success", message=f"No memory found for conversation '{conversation_id}'.")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if not qdrant_indexer:
        return RetrieveResponse(
            status="error",
            error_message="Qdrant service is not available."
        )
        
    try:
        results = qdrant_indexer.search(request.query)
        chunks = [{"text": r.get("text", ""), "score": r.get("score", 0.0)} for r in results]
        citations = [r.get("metadata", {}) for r in results]
        return RetrieveResponse(chunks=chunks, citations=citations, status="success")
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return RetrieveResponse(status="error", error_message=f"Retrieval failed: {str(e)}")

@app.post("/vision/infer", response_model=VisionInferResponse)
async def vision_infer(request: VisionInferRequest):
    if not vision_service:
        return VisionInferResponse(
            status="error",
            error_message="Vision service is not available."
        )
        
    if request.chip_path:
        request.chip_path = request.chip_path.replace("\\", "/")
    validate_chip_path(request.chip_path)
        
    try:
        result = vision_service.infer(request.chip_path)
        if "error" in result:
             return VisionInferResponse(
                status="error",
                error_message=result["error"]
            )
        return VisionInferResponse(vision_result=result, status="success")
    except Exception as e:
        logger.error(f"Error during vision inference: {e}")
        return VisionInferResponse(status="error", error_message=f"Vision inference failed: {str(e)}")

@app.post("/vision/debug", response_model=VisionDebugResponse)
async def vision_debug(request: VisionDebugRequest):
    if not vision_service:
        return VisionDebugResponse(
            status="error",
            error_message="Vision service is not available."
        )
        
    if request.chip_path:
        request.chip_path = request.chip_path.replace("\\", "/")
    validate_chip_path(request.chip_path)
        
    try:
        result = vision_service.infer(request.chip_path, debug=True)
        if "error" in result:
             return VisionDebugResponse(
                status="error",
                error_message=result["error"]
            )
        # Assuming the explainability artifacts paths are stored in the result dict under a specific key if generated
        explain_outputs = result.pop("explainability_outputs", {})
        return VisionDebugResponse(vision_result=result, explainability_outputs=explain_outputs, status="success")
    except Exception as e:
        logger.error(f"Error during vision debug inference: {e}")
        return VisionDebugResponse(status="error", error_message=f"Vision debug failed: {str(e)}")

@app.post("/vision/change-detection", response_model=ChangeDetectionResponse)
async def vision_change_detection(request: ChangeDetectionRequest):
    if not vision_service:
        return ChangeDetectionResponse(
            status="error",
            error_message="Vision service is not available."
        )
        
    if request.before_image:
        request.before_image = request.before_image.replace("\\", "/")
    if request.after_image:
        request.after_image = request.after_image.replace("\\", "/")
    validate_chip_path(request.before_image)
    validate_chip_path(request.after_image)
        
    try:
        result = vision_service.infer_change(request.before_image, request.after_image)
        if "error" in result:
             return ChangeDetectionResponse(
                status="error",
                error_message=result["error"]
            )
        return ChangeDetectionResponse(
            changed_area=result.get("changed_area"),
            flood_expansion=result.get("flood_expansion"),
            flood_reduction=result.get("flood_reduction"),
            change_percentage=result.get("change_percentage"),
            change_mask_path=result.get("change_mask_path"),
            status="success"
        )
    except Exception as e:
        logger.error(f"Error during vision change detection: {e}")
        return ChangeDetectionResponse(status="error", error_message=f"Vision change detection failed: {str(e)}")

import pandas as pd

@app.get("/chips", response_model=ChipsResponse)
async def list_chips():
    """Returns available satellite chips."""
    try:
        parquet_path = "data/chip_catalog.parquet"
        if not os.path.exists(parquet_path):
            return ChipsResponse(status="error", error_message="Chip catalog not found.")
            
        df = pd.read_parquet(parquet_path)
        valid_chips = []
        for record in df.to_dict(orient="records"):
            chip_path = record.get("file_path", "")
            if isinstance(chip_path, str):
                chip_path = chip_path.replace("\\", "/")
                record["file_path"] = chip_path
            try:
                validate_chip_path(chip_path)
                valid_chips.append(record)
            except Exception:
                pass
                
        return ChipsResponse(chips=valid_chips, status="success")
    except Exception as e:
        logger.error(f"Error listing chips: {e}")
        return ChipsResponse(status="error", error_message=str(e))

@app.get("/chips/{chip_id}", response_model=ChipDetailsResponse)
async def get_chip_details(chip_id: str):
    """Returns details for a specific chip."""
    try:
        parquet_path = "data/chip_catalog.parquet"
        if not os.path.exists(parquet_path):
            raise HTTPException(status_code=404, detail="Chip catalog not found.")
            
        df = pd.read_parquet(parquet_path)
        chip_row = df[df["chip_id"] == chip_id]
        
        if chip_row.empty:
            raise HTTPException(status_code=404, detail="Chip not found.")
            
        metadata = chip_row.iloc[0].to_dict()
        
        centroid = None
        if "centroid_lon" in metadata and "centroid_lat" in metadata:
            centroid = [metadata["centroid_lon"], metadata["centroid_lat"]]
            
        area_km2 = metadata.get("area_km2")
        if area_km2 is None and "area_sqm" in metadata:
            area_km2 = metadata["area_sqm"] / 1e6
            
        bounds_raw = metadata.get("bounds_wgs84")
        if isinstance(bounds_raw, str):
            import ast
            try:
                bounds_raw = ast.literal_eval(bounds_raw)
            except Exception:
                bounds_raw = None
                
        return ChipDetailsResponse(
            chip_id=chip_id, 
            metadata=metadata, 
            status="success",
            district=metadata.get("district"),
            state=metadata.get("state"),
            country=metadata.get("country"),
            area_km2=area_km2,
            centroid=centroid,
            bounds=bounds_raw
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chip details: {e}")
        return ChipDetailsResponse(chip_id=chip_id, metadata={}, status="error", error_message=str(e))

from app.core.timeline import get_timeline

@app.get("/risk-timeline")
async def get_risk_timeline():
    """Returns the historical risk timeline data."""
    try:
        timeline_data = get_timeline()
        return {"status": "success", "data": timeline_data}
    except Exception as e:
        logger.error(f"Error fetching risk timeline: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_message": str(e)}
        )

