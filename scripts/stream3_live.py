#!/usr/bin/env python3
"""
STREAM3 Live — Local real-time window for GMA Vision SoundBox

- Synthetic stream by default; enable webcam with --webcam
- Displays an 8x8 agent grid (red=errors, blue=corrections, cyan=normal)
- Shows live metrics overlay; writes JSONL metrics to output/stream3_live_metrics.jsonl
- Press ESC to quit

Note: This is a local tool, not for Kaggle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "OpenCV (cv2) is required for live window. pip install opencv-python"
    ) from e


def draw_agents_grid(agents: np.ndarray, scale: int = 40) -> np.ndarray:
    # agents: shape (64,), values in {0 normal, 1 error, 2 correction}
    colors = {
        0: (255, 255, 0),  # cyan (BGR: (255,255,0))
        1: (0, 0, 255),  # red
        2: (255, 0, 0),  # blue
    }
    grid = np.zeros((8 * scale, 8 * scale, 3), dtype=np.uint8)
    for i in range(64):
        y, x = divmod(i, 8)
        c = colors.get(int(agents[i]), (255, 255, 255))
        grid[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale] = c
    return grid


class LiveVision:
    def __init__(self, use_webcam: bool = False, precision: float = 0.92) -> None:
        self.use_webcam = use_webcam
        self.precision = precision
        self.errors_total = 0
        self.corrections_total = 0
        self.agent_states = np.zeros(
            512, dtype=np.int8
        )  # 0 normal, 1 error, 2 correction
        self.metrics_path = Path("output/stream3_live_metrics.jsonl")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = None
        if self.use_webcam:
            self.cap = cv2.VideoCapture(0)

    def _step_agents(self) -> None:
        # randomly trigger error cascade and apply 92% corrections
        if np.random.rand() > 0.7:
            n_err = int(np.random.randint(10, 30))
            idx = np.random.choice(self.agent_states.size, n_err, replace=False)
            self.agent_states[idx] = 1
            self.errors_total += n_err
        # correct some errors
        error_idx = np.where(self.agent_states == 1)[0]
        if error_idx.size > 0:
            n_fix = int(error_idx.size * 0.92)
            if n_fix > 0:
                to_fix = error_idx[:n_fix]
                self.agent_states[to_fix] = 2
                self.corrections_total += n_fix
        # return corrected to normal
        self.agent_states[self.agent_states == 2] = 0

    def _compose_frame(self) -> np.ndarray:
        grid = draw_agents_grid(self.agent_states[:64], scale=50)
        h, w = grid.shape[:2]

        # optional webcam thumbnail
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                thumb = cv2.resize(frame, (w // 3, h // 3))
                grid[0 : thumb.shape[0], 0 : thumb.shape[1]] = thumb

        # overlay metrics
        lines = [
            f"Precision: {int(self.precision * 100)}%",
            f"Errors: {self.errors_total}",
            f"Corrections: {self.corrections_total}",
            f"Agents N/E/C: {np.sum(self.agent_states == 0)}/"
            f"{np.sum(self.agent_states == 1)}/{np.sum(self.agent_states == 2)}",
        ]
        y = 30
        for line in lines:
            cv2.putText(
                grid,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 28
        return grid

    def _write_metrics(self) -> None:
        record: Dict[str, object] = {
            "ts": time.time(),
            "precision": self.precision,
            "errors_total": int(self.errors_total),
            "corrections_total": int(self.corrections_total),
            "agents": {
                "normal": int(np.sum(self.agent_states == 0)),
                "error": int(np.sum(self.agent_states == 1)),
                "correction": int(np.sum(self.agent_states == 2)),
            },
        }
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def run(self) -> None:
        last_write = time.time()
        try:
            while True:
                self._step_agents()
                frame = self._compose_frame()
                cv2.imshow("GMA Vision SoundBox — Live", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == 27:  # ESC
                    break
                if time.time() - last_write > 1.0:
                    self._write_metrics()
                    last_write = time.time()
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STREAM3 Live Window")
    p.add_argument(
        "--webcam",
        action="store_true",
        help="Enable webcam thumbnail (default synthetic only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = LiveVision(use_webcam=args.webcam)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
