#!/usr/bin/env python3
"""
Script principal para executar a API Analise B3
"""
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    print("🚀 Iniciando B3 Analyser com Coletor Robusto...")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )