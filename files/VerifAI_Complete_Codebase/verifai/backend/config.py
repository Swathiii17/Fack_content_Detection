from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    tavily_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./verifai.db"
    model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"

settings = Settings()
