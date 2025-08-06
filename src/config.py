# src/config.py
from pydantic import BaseSettings, Field


class AppConfig(BaseSettings):
    """
    Defines the application's configuration using Pydantic for validation.
    Reads from environment variables and provides default values.
    """

    # WebSocket Server Configuration
    WEBSOCKET_HOST: str = Field("0.0.0.0", env="WEBSOCKET_HOST")
    WEBSOCKET_PORT: int = Field(8765, env="WEBSOCKET_PORT")

    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(levelname)s - %(message)s", env="LOG_FORMAT"
    )

    # Vision Engine Configuration
    VISION_BUFFER_SIZE: int = Field(1000, env="VISION_BUFFER_SIZE")
    SIMULATED_AGENT_COUNT: int = Field(4, env="SIMULATED_AGENT_COUNT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instantiate the config
config = AppConfig()
