# Geospatial Copilot Startup Guide

This guide covers how to start the Geospatial Copilot application. 

## 1. Quick Start (Recommended)

The easiest and recommended way to start the entire application (Frontend, Backend, and Vector Database) is using Docker Compose.

```powershell
docker compose up
```
*(Add the `-d` flag to run in detached mode in the background: `docker compose up -d`)*

### Accessing the Application
Once the containers are built and running, you can access the application at:
- **Frontend UI:** [http://localhost](http://localhost) (or [http://localhost:80](http://localhost:80))
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

To stop the application, press `Ctrl+C` in the terminal, or run:
```powershell
docker compose down
```

---

## 2. Manual Startup (Legacy)

If you need to run the services manually without the main docker-compose frontend/backend network (e.g., for local development), follow these steps:

### Step 1: Start Qdrant (Vector DB)
You still need Qdrant running for the document retrieval system. You can start it using Docker:
```powershell
docker compose up -d qdrant
```
*(Or use `docker run --rm --name geospatial-qdrant -p 6333:6333 -p 6334:6334 -v "${PWD}\qdrant_data:/qdrant/storage" qdrant/qdrant`)*

### Step 2: Activate the Python Environment
Open a new terminal and activate your virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Start the Backend Server
Run the FastAPI backend server directly via Uvicorn:
```powershell
python -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 4: Start the Frontend (If applicable locally)
Open another terminal, navigate to the frontend folder, and start the development server:
```powershell
cd frontend
npm run dev
```
*(The frontend will typically be accessible at `http://localhost:5173` or whichever port Vite allocates).*
