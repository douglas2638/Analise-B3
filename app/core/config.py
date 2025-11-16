from decouple import config
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "B3 Data Analyzer"
    VERSION: str = "1.0.0"
    DEBUG: bool = config('DEBUG', default=False, cast=bool)
    DATABASE_URL: str = config('DATABASE_URL', default='sqlite:///./b3_data.db')
    API_PORT: int = config('API_PORT', default=8000, cast=int)
    API_HOST: str = config('API_HOST', default='0.0.0.0')
    
    class Config:
        env_file = ".env"

settings = Settings()