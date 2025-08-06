# src/config.py
import os


class Config:
    """
    Configuration settings for the SandboxVision application.
    Settings are read from environment variables with sensible defaults.
    """

    # WebSocket Server Configuration
    WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "0.0.0.0")
    WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8765))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


# Instantiate the config
config = Config()
