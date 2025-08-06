# tests/test_vision_engine.py
import asyncio
import pytest
from src.vision_stream_engine import VisionStreamEngine, VisionFrame


@pytest.mark.asyncio
async def test_engine_subscription_and_capture():
    """
    Tests if a subscriber receives a frame captured by the engine.
    """
    # Arrange
    engine = VisionStreamEngine()
    received_frame = None

    # Define an async subscriber
    async def subscriber(frame: VisionFrame):
        nonlocal received_frame
        received_frame = frame
        await asyncio.sleep(0)  # Yield control

    engine.subscribe(subscriber)

    test_frame = VisionFrame(
        timestamp=12345.678,
        agent_id="test_agent",
        vision_type="test",
        data={"key": "value"},
    )

    # Act
    await engine.capture_vision(test_frame)

    # Assert
    assert received_frame is not None
    assert received_frame.agent_id == "test_agent"
    assert received_frame.data["key"] == "value"


def test_engine_initialization():
    """
    Tests the initial state of the VisionStreamEngine.
    """
    # Arrange & Act
    engine = VisionStreamEngine()

    # Assert
    assert not engine.streaming
    assert engine.stream_thread is None
    assert len(engine.subscribers) == 0
