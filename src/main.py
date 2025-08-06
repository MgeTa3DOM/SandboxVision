# src/main.py
import asyncio
import logging
from src.config import config  # Import the config object
from src.vision_stream_engine import VisionStreamEngine, SimulatedVisionDataSource
from src.websocket_broadcaster import WebSocketBroadcaster
from src.close_the_loop_validator import CloseTheLoopValidator
from src.memory_integration import VisionMemoryIntegration

# Use the configuration from the config object
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


async def start_system():
    """Orchestrates the launch of the SandboxVision system."""
    logging.info("--- [SandboxVision Operation] System starting ---")

    # 1. Initialize components
    broadcaster = WebSocketBroadcaster(
        host=config.WEBSOCKET_HOST, port=config.WEBSOCKET_PORT
    )
    data_source = SimulatedVisionDataSource()
    vision_engine = VisionStreamEngine(
        data_source, buffer_size=config.VISION_BUFFER_SIZE
    )
    validator = CloseTheLoopValidator(vision_engine)
    memory = VisionMemoryIntegration(vision_engine)

    # 2. Connect components
    vision_engine.subscribe(broadcaster.broadcast)

    # 3. Start services
    logging.info("Starting services...")
    vision_engine.start_streaming()

    # 4. Run services concurrently
    tasks = [
        asyncio.create_task(broadcaster.start()),
        # Add other long-running tasks here
    ]

    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
    finally:
        logging.info("--- [SandboxVision Operation] System shutting down ---")
        vision_engine.stop_streaming()
        broadcaster.stop()
        logging.info("All services have been shut down.")


def main():
    """Synchronous entry point to run the asyncio event loop."""
    try:
        asyncio.run(start_system())
    except KeyboardInterrupt:
        logging.info("\n--- [SandboxVision Operation] Manual shutdown requested ---")


if __name__ == "__main__":
    main()
