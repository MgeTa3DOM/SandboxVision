# src/vision_stream_engine.py
import asyncio
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import logging
from src.config import config


@dataclass
class VisionFrame:
    timestamp: float
    agent_id: str
    vision_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "vision_type": self.vision_type,
            "data": self.data,
            "metadata": self.metadata,
            "validation_status": self.validation_status,
            "confidence": self.confidence,
        }


class VisionDataSource:
    """Abstracts the source of vision data, allowing for interchangeable sources."""

    def __init__(self):
        pass

    def get_frame(self) -> VisionFrame:
        """Returns a new vision frame. To be implemented by subclasses."""
        raise NotImplementedError


class SimulatedVisionDataSource(VisionDataSource):
    """Generates simulated vision data for testing and development."""

    def get_frame(self) -> VisionFrame:
        vision_types = ["activation", "thought", "decision", "action"]
        return VisionFrame(
            timestamp=datetime.now().timestamp(),
            agent_id=f"agent_{np.random.randint(1, config.SIMULATED_AGENT_COUNT + 1)}",
            vision_type=np.random.choice(vision_types),
            data={"value": np.random.rand()},
            confidence=np.random.uniform(0.7, 1.0),
        )


class VisionStreamEngine:
    def __init__(self, data_source: VisionDataSource, buffer_size: int = 1000):
        self.data_source = data_source
        self.subscribers: List[Callable] = []
        self.streaming = False
        self._stream_task: Optional[asyncio.Task] = None

    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)

    async def _notify_subscribers(self, frame: VisionFrame):
        """Asynchronously notifies all subscribers, with error handling."""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(frame)
                else:
                    subscriber(frame)
            except Exception as e:
                logging.error(f"Error in subscriber {subscriber.__name__}: {e}")

    async def _stream_loop(self):
        """The main loop that generates and processes vision frames."""
        while self.streaming:
            frame = self.data_source.get_frame()
            await self._notify_subscribers(frame)
            await asyncio.sleep(0.5)

    def start_streaming(self):
        """Starts the vision stream in a non-blocking manner."""
        if not self.streaming:
            self.streaming = True
            self._stream_task = asyncio.create_task(self._stream_loop())
            logging.info("Vision stream started.")

    def stop_streaming(self):
        """Stops the vision stream gracefully."""
        if self.streaming:
            self.streaming = False
            if self._stream_task:
                self._stream_task.cancel()
                logging.info("Vision stream stopped.")
