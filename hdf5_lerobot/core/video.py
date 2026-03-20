"""FFmpeg-based video writer for encoding frames to MP4."""

import logging
import os
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


class FFmpegVideoWriter:
    """Write frames to MP4 via ffmpeg subprocess."""

    def __init__(self, output_path: str, fps: int, height: int, width: int,
                 codec: str = "libx264", pix_fmt: str = "yuv420p", crf: int = 23):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
            "-pix_fmt", pix_fmt,
            "-crf", str(crf),
            "-an",
            output_path,
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self.frame_count = 0

    def write_frame(self, frame: np.ndarray):
        """Write an HWC uint8 RGB frame."""
        assert frame.dtype == np.uint8
        self.proc.stdin.write(frame.tobytes())
        self.frame_count += 1

    def close(self) -> int:
        self.proc.stdin.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            stderr = self.proc.stderr.read().decode()
            logger.error(f"ffmpeg error for {self.output_path}: {stderr[-500:]}")
        return self.frame_count
