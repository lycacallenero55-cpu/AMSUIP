from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from config import settings
from api.training import router as training_router
from api.verification import router as verification_router
from api.progress import router as progress_router
from api.versioning import router as versioning_router
from api.utils import router as utils_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(settings.LOCAL_MODELS_DIR, exist_ok=True)
    print(f"🚀 AI Training Backend started on {settings.HOST}:{settings.PORT}")
    print(f"📁 Local models directory: {settings.LOCAL_MODELS_DIR}")
    yield
    # Shutdown
    print("🛑 AI Training Backend stopped")

app = FastAPI(
    title="Signature AI Training API",
    description="AI-powered signature verification training backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training_router, prefix="/api/training", tags=["ai-training"])
app.include_router(verification_router, prefix="/api/verification", tags=["ai-verification"])
app.include_router(progress_router, prefix="/api/progress", tags=["progress"])
app.include_router(versioning_router, prefix="/api/versioning", tags=["versioning"])
app.include_router(utils_router, prefix="/api/utils", tags=["utils"])

@app.get("/")
async def root():
    return {
        "message": "Signature AI Training API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )