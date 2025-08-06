# src/close_the_loop_validator.py
import logging
from typing import Tuple
from .vision_stream_engine import VisionFrame


class CloseTheLoopValidator:
    def __init__(self, vision_engine):
        self.vision_engine = vision_engine
        self.vision_engine.subscribe(self.validate_frame)

    async def validate_frame(self, frame: VisionFrame):
        is_valid, confidence, reason = self._validate_action(frame)
        frame.validation_status = "validated" if is_valid else "rejected"
        frame.confidence = confidence
        logging.info(f"Frame {frame.agent_id} validation: {frame.validation_status}")

    def _validate_action(self, frame: VisionFrame) -> Tuple[bool, float, str]:
        # Simple validation logic for demo
        if frame.data.get("value", 0) > 0.1:
            return True, frame.confidence, "OK"
        else:
            return False, frame.confidence * 0.5, "Low value"
