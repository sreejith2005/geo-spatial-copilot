# Geospatial Risk Copilot

An AI-powered copilot for geospatial risk assessment, integrating intelligent agents, vision models, RAG (Retrieval-Augmented Generation), and robust geospatial analyses.

## Overview

The Geospatial Risk Copilot is a modular intelligent system designed to parse, analyze, and synthesize critical insights from diverse geospatial data sources. It leverages large language models (LLMs) managed by LangGraph, robust vector retrieval (Qdrant), and deep learning vision pipelines to interpret Earth observation data (e.g., Sentinel-1 SAR imagery) alongside policy and geographical documents.

## Problem Statement

Geospatial risk assessment often requires analysts to manually synthesize information from unstructured text (e.g., disaster reports) and complex imagery (e.g., satellite radar). This process is slow, prone to human error, and fragmented. The Geospatial Risk Copilot unifies these workflows by combining computer vision for flood segmentation with an intelligent RAG-driven conversational agent, providing a single seamless interface for querying multimodal risk data.

## Architecture

The project consists of three main pillars:
1. **RAG Layer**: Indexes and retrieves PDF documents using Sentence-Transformers and Qdrant.
2. **Vision Pipeline**: A specialized UNet-based segmentation model trained on Sen1Floods11 to detect surface water from synthetic aperture radar (SAR) imagery.
3. **LangGraph Orchestration**: A stateful workflow engine that intelligently routes user queries to the RAG layer, the Vision Pipeline, or a synthesis engine based on the query intent.

*For more details, see [docs/architecture.md](docs/architecture.md).*

## Data Sources

- **Sen1Floods11**: Used for training the flood segmentation vision model. It consists of Sentinel-1 S1Hand images and LabelHand water masks.
- **Elevation Data**: SRTM raster datasets for regional topography (planned).
- **Documents Corpus**: RAG relies on a curated set of PDF documents located in `data/documents`.

## Setup Steps

Please refer to the [Local Setup Runbook](docs/runbook.md) for detailed instructions on virtual environments, dependencies, and Qdrant setup.

## How to Run

### 1. Ingestion (RAG)
To parse PDFs from `data/documents` and populate the local Qdrant database:
```powershell
python scripts\run_ingestion_pipeline.py
```

### 2. Vision Model Training
To train the UNet flood segmentation model using the Sen1Floods11 dataset:
```powershell
python scripts\train_flood_model.py
```

### 3. Running the Backend API
Start the FastAPI server:
```powershell
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Running Tests
Execute the test suite using pytest:
```powershell
pytest tests\
```
To run the end-to-end smoke test against a running backend:
```powershell
python tests\smoke_test.py
```

## Example API Requests

*Check [docs/api_reference.md](docs/api_reference.md) for complete details.*

**Retrieve Documents:**
```json
POST /retrieve
{
  "query": "What are the flood risks in region X?",
  "top_k": 3
}
```

**End-to-End Analysis:**
```json
POST /analyze
{
  "query": "Summarize the risk based on the attached image and reports.",
  "chip_path": "data/raw/sen1floods11/some_image.tif"
}
```

## Known Limitations

- **Vision Modality**: Currently optimized primarily for SAR (Sentinel-1) flood detection; other modalities (optical, multi-spectral) require additional model training.
- **RAG Dependency**: Hallucinations may occur if the vector database lacks relevant context. The current chunking strategy is basic.
- **Windows Exclusivity**: Pathing and certain scripts are tailored to Windows environments (Powershell).

## Future Improvements

- Support for raster DEMs (Elevation data) to improve flood prediction based on topology.
- Dockerizing the FastAPI backend and model inference for Kubernetes deployment.
- Expanding the LangGraph agent tools to include real-time weather API integration.
