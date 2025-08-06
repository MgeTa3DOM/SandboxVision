# src/main.py
import asyncio
import logging
from src.config import config  # Import the config object
from src.vision_stream_engine import VisionStreamEngine
from src.websocket_broadcaster import WebSocketBroadcaster
from src.close_the_loop_validator import CloseTheLoopValidator
from src.memory_integration import VisionMemoryIntegration

# Use the configuration from the config object
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


async def start_system():
    """Orchestrates the launch of the SandboxVision system."""
    logging.info("--- [SandboxVision Operation] System starting ---")

    # 1. Initialize components with centralized configuration
    logging.info(
        f"Initializing WebSocket Broadcaster on {config.WEBSOCKET_HOST}:{config.WEBSOCKET_PORT}"
    )
    broadcaster = WebSocketBroadcaster(
        host=config.WEBSOCKET_HOST, port=config.WEBSOCKET_PORT
    )
    vision_engine = VisionStreamEngine()

    # 2. Connect components
    logging.info("Connecting modules...")
    validator = CloseTheLoopValidator(vision_engine)
    memory = VisionMemoryIntegration(vision_engine)

    vision_engine.subscribe(lambda frame: broadcaster.broadcast(frame.to_dict()))

    # 3. Start services
    logging.info("Starting services...")
    vision_engine.start_streaming()

    try:
        await broadcaster.start()
    except OSError as e:
        logging.error(f"Could not start WebSocket server. Error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("System shutdown.")


def main():
    """Synchronous entry point to run the asyncio event loop."""
    try:
        asyncio.run(start_system())
    except KeyboardInterrupt:
        logging.info("\n--- [SandboxVision Operation] Manual shutdown requested ---")


if __name__ == "__main__":
    main()
