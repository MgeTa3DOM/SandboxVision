# tests/test_integration.py
import asyncio
import pytest
from src.vision_stream_engine import VisionStreamEngine
from src.websocket_broadcaster import WebSocketBroadcaster


@pytest.mark.asyncio
async def test_engine_to_broadcaster_integration():
    """
    Tests if the VisionStreamEngine correctly sends frames to the WebSocketBroadcaster.
    """
    # Arrange
    engine = VisionStreamEngine()
    broadcaster = WebSocketBroadcaster()
    received_messages = []

    # Mock the broadcaster's broadcast method
    async def mock_broadcast(message):
        received_messages.append(message)
        await asyncio.sleep(0)

    broadcaster.broadcast = mock_broadcast

    # Subscribe the broadcaster to the engine
    engine.subscribe(
        lambda frame: asyncio.create_task(broadcaster.broadcast(frame.to_dict()))
    )

    # Act
    engine.start_streaming()
    await asyncio.sleep(1)  # Allow some time for frames to be generated

    # Assert
    assert len(received_messages) > 0
    assert "agent_id" in received_messages[0]
    assert "vision_type" in received_messages[0]
