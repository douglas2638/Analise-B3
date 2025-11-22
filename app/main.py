from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.endpoints import router as api_router
from app.core.config import settings
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.middleware("http")
async def catch_json_errors(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except ValueError as e:
        if "Out of range float values" in str(e):
            logger.error(f"JSON serialization error in {request.url}: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Erro interno de serialização de dados. Valores numéricos inválidos detectados."}
            )
        # Re-lançar outros erros ValueError
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in {request.url}: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Erro interno do servidor"}
        )


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "B3 Data Analyzer API", 
        "status": "online",
        "version": settings.VERSION
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": __import__('datetime').datetime.now()}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )