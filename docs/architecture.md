# Architecture

The Geospatial Risk Copilot leverages a unified, modular architecture combining intelligent orchestration, vector retrieval, and computer vision.

## RAG Layer

The **Retrieval-Augmented Generation (RAG)** layer provides the agent with domain-specific knowledge by indexing a corpus of PDF documents.
- **Components**: Document parsers, `sentence-transformers` for embeddings, and a local **Qdrant** instance for vector storage.
- **Workflow**: Documents in `data/documents` are loaded, chunked with semantic overlap, embedded, and indexed. When a query is made, the top chunks are retrieved and injected into the LLM prompt.

## LangGraph Orchestration

**LangGraph** acts as the central brain, orchestrating stateful multi-agent workflows.
- **State**: A shared Python dictionary containing the user query, chip path, retrieved chunks, vision results, citations, and final answer.
- **Routing**: Nodes in the graph interpret the user query to decide whether to invoke the RAG retriever, run the vision pipeline on an image chip, or do both.
- **Synthesis**: The final node takes all gathered data (text chunks, vision probabilities) and generates a coherent, human-readable summary.

## Vision Pipeline

The **Vision Pipeline** handles the ingestion and inference of remote sensing imagery.
- **Model Architecture**: A PyTorch-based UNet segmentation model.
- **Dataset**: Sen1Floods11 (Sentinel-1 SAR imagery and water masks).
- **Function**: Takes a path to a geospatial image chip (e.g., `.tif`) and outputs a probability mask indicating the presence of surface water/flooding.

## Backend

The **Backend** is built on **FastAPI** to provide robust, typed, and easily consumable endpoints.
- **Pydantic**: Enforces strict request/response validation.
- **Middleware**: Injects unique Request IDs and tracks endpoint latency.
- **Services**: Wraps the LangGraph orchestration, Qdrant indexer, and Vision models as long-lived singletons.

## End-to-End Data Flow

1. **Client Request**: User sends a query (and optionally an image path) to `POST /analyze`.
2. **Initialization**: API initializes the LangGraph state.
3. **Execution**: Graph routes the query:
   - If textual context is needed, Qdrant retrieves relevant document chunks.
   - If an image path is provided, the Vision Service runs inference.
4. **Synthesis**: The LLM synthesizes the retrieved text and vision results into a final answer.
5. **Response**: API returns the structured response containing the summary, citations, and vision metadata.
