# Local Setup Runbook

Follow these steps to configure, train, and run the Geospatial Copilot locally on Windows.

## 1. Activate Virtual Environment
Ensure you have Python 3.10+ installed.

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 2. Start Qdrant
Qdrant is required for the RAG layer to store and retrieve vector embeddings.

### Option A: Use Docker Compose (Recommended)
Since a `docker-compose.yml` is provided, you can start Qdrant and let Docker manage container reuse and lifecycle:
```powershell
# Start Qdrant in the background (detached mode)
docker compose up -d qdrant
```
To stop the container, run:
```powershell
docker compose down
```

### Option B: Use Docker Run (with cleanup)
If you prefer using `docker run` directly, add the `--rm` flag to automatically delete the container when it stops (since Qdrant's data is persisted in the volume, you won't lose your vector database), and name the container with `--name`:
```powershell
docker run --rm --name geospatial-qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}\qdrant_data:/qdrant/storage" qdrant/qdrant
```

## 3. Run Ingestion (Optional)
If you have new PDF documents in `data/documents`, run the ingestion pipeline to embed and index them into Qdrant.

```powershell
python scripts\run_ingestion_pipeline.py
```

## 4. Run Training
Train the UNet flood segmentation model using the Sen1Floods11 dataset. Ensure `data/raw/sen1floods11` is populated.

```powershell
python scripts\train_flood_model.py
```

## 5. Run Evaluation
Evaluate the trained vision model to generate performance metrics.

```powershell
python scripts\evaluate_flood_model.py
```

## 6. Run the Backend
Start the FastAPI server. The server will automatically connect to Qdrant and load the compiled LangGraph workflow.

```powershell
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at http://localhost:8000. You can view the Swagger UI at http://localhost:8000/docs.*

## 7. Run Integration Tests
With the backend running (Step 6), execute the test suite to verify functionality.

```powershell
# Run API unit/integration tests
pytest tests\

# Run the end-to-end smoke test
python tests\smoke_test.py
```
