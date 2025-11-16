from decouple import config
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "B3 Data Analyzer"
    DEBUG: bool = config('DEBUG', default=False, cast=bool)
    DATABASE_URL: str = config('DATABASE_URL', default='sqlite:///./b3_data.db')
    
    class Config:
        env_file = ".env"

settings = Settings()