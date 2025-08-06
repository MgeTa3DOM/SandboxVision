# src/memory_integration.py
import logging
from collections import deque
from .vision_stream_engine import VisionFrame


class VisionMemoryIntegration:
    def __init__(self, vision_engine):
        self.vision_engine = vision_engine
        self.memory_buffer = deque(maxlen=200)
        self.vision_engine.subscribe(self.collect_for_memory)

    async def collect_for_memory(self, frame: VisionFrame):
        if frame.validation_status == "rejected" or frame.confidence < 0.5:
            self.memory_buffer.append(frame)
            logging.info(f"Frame {frame.agent_id} added to memory buffer.")
