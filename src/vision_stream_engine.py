# src/vision_stream_engine.py
import asyncio
import threading
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
import logging


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


class VisionStreamEngine:
    def __init__(self, buffer_size: int = 1000):
        self.subscribers: List[Callable] = []
        self.streaming = False
        self.stream_thread = None

    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)

    async def capture_vision(self, frame: VisionFrame):
        tasks = []
        for subscriber in self.subscribers:
            if asyncio.iscoroutinefunction(subscriber):
                tasks.append(asyncio.create_task(subscriber(frame)))
            else:
                subscriber(frame)
        if tasks:
            await asyncio.gather(*tasks)

    def _generate_simulated_frame(self) -> VisionFrame:
        vision_types = ["activation", "thought", "decision", "action"]
        return VisionFrame(
            timestamp=datetime.now().timestamp(),
            agent_id=f"agent_{np.random.randint(1, 5)}",
            vision_type=np.random.choice(vision_types),
            data={"value": np.random.rand()},
            confidence=np.random.uniform(0.7, 1.0),
        )

    async def _stream_loop(self):
        while self.streaming:
            frame = self._generate_simulated_frame()
            await self.capture_vision(frame)
            await asyncio.sleep(0.5)

    def _run_stream_loop_in_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._stream_loop())
        loop.close()

    def start_streaming(self):
        if not self.streaming:
            self.streaming = True
            self.stream_thread = threading.Thread(
                target=self._run_stream_loop_in_thread, daemon=True
            )
            self.stream_thread.start()
            logging.info("🎬 Streaming de vision démarré.")
