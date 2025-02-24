from pathlib import Path
import numpy as np
import imageio
from typing import List, Optional
#ADD full craftax recording later
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN


class EpisodeRecorder:
    """Records frames from a Craftax episode."""

    def __init__(self, save_dir: str = "recordings", enabled: bool = True, ):
        self.save_dir = Path(save_dir)
        self.enabled = enabled
        if enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.frames: List[np.ndarray] = []

    def record_frame(self, state: EnvState) -> None:
        """Record a single frame from the environment state."""
        if not self.enabled:
            return

        pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_HUMAN)
        frame = np.array(pixels).astype(np.uint8)
        self.frames.append(frame)

    def save_video(self, filename: str, fps: int = 10) -> Optional[Path]:
        """Save recorded frames as a video file."""
        if not self.enabled or not self.frames:
            return None

        save_path = self.save_dir / filename
        imageio.mimsave(str(save_path), self.frames, fps=fps, codec="mpeg4")
        #print(f"Saved {len(self.frames)} frames to {save_path}")
        self.frames = []  # Clear frames after saving
        return save_path

    def clear(self) -> None:
        """Clear recorded frames without saving."""
        if self.enabled:
            self.frames = []
