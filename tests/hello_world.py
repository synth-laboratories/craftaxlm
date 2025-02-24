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

    # Add debug print statements
    print("Environment initialized")
    print(f"Action space: {env.env.action_space}")

    # Reset environment
    obs, state = env.env.reset(key, env.env_params)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial state: {state}")
    recorder.record_frame(state)

    # Run episode
    for step in tqdm.tqdm(range(num_steps)):
        # Random action
        action = np.random.randint(0, 8)
        print(f"Step {step}, Action: {action}")

        # Step environment
        obs, state, reward, done, info = env.env.step(
            key, state, action, env.env_params
        )
        print(f"Reward: {reward}, Done: {done}")
        if info:
            print(f"Info: {info}")
        recorder.record_frame(state)

        if done:
            print("Episode finished early")
            break

    # Save video
    print(f"Saving video to {output_path}")
    recorder.save_video(output_path, fps=fps)


if __name__ == "__main__":
    # Add error handling
    try:
        run_random_episode(num_steps=100, output_path="random_episode.mp4", fps=10)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
