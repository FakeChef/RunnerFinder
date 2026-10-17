"""
Read-only frame access for video files (OpenCV VideoCapture, already part of the engine runtime).

* The source file is only opened for reading; nothing is written next to it, nothing is
  re-encoded, no temporary files are created. Decoded frames stay in memory.
* OpenCV applies the display rotation (CAP_PROP_ORIENTATION_AUTO, default on), so frames are
  returned as displayed (1080x1920 for the portrait Canon clips of B1).
* Seeking: CAP_PROP_POS_FRAMES (keyframe + decode forward inside OpenCV). B1 verified 65/65
  frames pixel-identical to a full sequential decode, forward and backward.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v")


class UnreadableVideoError(Exception):
    """The file cannot be opened or decoded as a video."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int          # as displayed (rotation applied)
    height: int
    duration_s: float   # frame_count / fps (constant frame rate)


def frame_index(t: float, fps: float, n_frames: int) -> int:
    """The frame shown at presentation time t (largest pts <= t) - the B1 rule."""
    import math
    return int(min(max(math.floor(t * fps + 1e-6), 0), n_frames - 1))


class VideoReader:
    """One persistent VideoCapture per video. Use as a context manager."""

    def __init__(self, path: str):
        import cv2
        self._cv2 = cv2
        self.path = path
        t = time.perf_counter()
        self.cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        self.open_ms = (time.perf_counter() - t) * 1000
        if not self.cap.isOpened():
            self.cap.release()
            raise UnreadableVideoError("cannot open the file as a video")
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or n <= 0:
            self.cap.release()
            raise UnreadableVideoError(f"no frames (fps {fps}, frames {n})")
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.info = VideoInfo(fps, n, w, h, n / fps)

    def frame_rgb(self, index: int):
        """(PIL RGB image as displayed, pts seconds, seek_ms, decode_ms). Raises UnreadableVideoError."""
        from PIL import Image
        cv2 = self._cv2
        t = time.perf_counter()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        seek_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        ok, frame = self.cap.read()
        decode_ms = (time.perf_counter() - t) * 1000
        if not ok or frame is None:
            raise UnreadableVideoError(f"frame {index} cannot be decoded")
        pts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        del frame
        return image, pts, seek_ms, decode_ms

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def probe(path: str) -> VideoInfo:
    """Metadata only (opens and closes the file; decodes the first frame to get the displayed size)."""
    with VideoReader(path) as r:
        img, *_ = r.frame_rgb(0)
        w, h = img.size
        return VideoInfo(r.info.fps, r.info.frame_count, w, h, r.info.duration_s)
