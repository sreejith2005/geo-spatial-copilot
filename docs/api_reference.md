# API Reference

The FastAPI backend exposes several endpoints for integrating the Geospatial Copilot into external systems. All requests and responses are strongly typed using Pydantic.

## `GET /health`
Basic liveness probe.

**Response:**
```json
{
  "status": "ok"
}
```

## `GET /ready`
Readiness probe that checks the availability of downstream services (Qdrant, Vision Service, LangGraph).

**Response:**
```json
{
  "status": "ready",
  "qdrant_status": "ok",
  "vision_status": "ok",
  "graph_status": "ok"
}
```

## `POST /analyze`
The primary orchestration endpoint. Invokes the LangGraph state machine.

**Request:**
```json
{
  "query": "Assess the flood risk for the provided satellite image chip and check local policy.",
  "chip_path": "data/raw/sen1floods11/test_image.tif"
}
```

**Response:**
```json
{
  "summary": "Based on the image analysis, there is a high probability of flooding. According to local policy...",
  "citations": [{"source": "flood_policy.pdf", "page": 4}],
  "vision_result": {"flood_probability": 0.85, "water_pixels": 10540},
  "prediction_result": null,
  "trace_log": ["Routed to Vision", "Routed to RAG", "Synthesized Response"],
  "status": "success",
  "error_message": null
}
```

## `POST /retrieve`
Directly queries the Qdrant vector database (RAG layer) without invoking the full LangGraph orchestration.

**Request:**
```json
{
  "query": "Emergency response procedures for flooding",
  "top_k": 3
}
```

**Response:**
```json
{
  "chunks": [
    {
      "text": "...evacuation routes should be established...",
      "score": 0.89
    }
  ],
  "citations": [
    {"source": "emergency_plan.pdf"}
  ],
  "status": "success",
  "error_message": null
}
```

## `POST /vision/infer`
Directly runs the vision segmentation model on a provided image chip.

**Request:**
```json
{
  "chip_path": "data/raw/sen1floods11/test_image.tif"
}
```

**Response:**
```json
{
  "vision_result": {
    "flood_probability": 0.85,
    "water_pixels": 10540
  },
  "status": "success",
  "error_message": null
}
```
