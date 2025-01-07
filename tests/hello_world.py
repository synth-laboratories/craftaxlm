import jax
import numpy as np
import tqdm

from craftaxlm.classic.aci import CraftaxClassicACI
from craftaxlm.recording import EpisodeRecorder


def run_random_episode(
    num_steps: int = 100,
    output_path: str = "random_episode.mp4",
    fps: int = 10,
    seed: int = 42,
):
    """Run and record a random episode."""
    # Initialize
    key = jax.random.PRNGKey(seed)
    env = CraftaxClassicACI(seed=seed)
    recorder = EpisodeRecorder()

    # Reset environment
    obs, state = env.env.reset(key, env.env_params)
    recorder.record_frame(state)

    # Run episode
    for _ in tqdm.tqdm(range(num_steps)):
        # Random action
        action = np.random.randint(0, 8)

        # Step environment
        _, state, reward, done, info = env.env.step(key, state, action, env.env_params)
        recorder.record_frame(state)

        if done:
            break

    # Save video
    recorder.save_video(output_path, fps=fps)


if __name__ == "__main__":
    run_random_episode(num_steps=100, output_path="random_episode.mp4", fps=10)
