from craftax.craftax.constants import *
from craftax.craftax_env import make_craftax_env_from_name

from craftaxlm.full.metadata import (
    CRAFTAX_FULL_ACHIEVEMENTS,
    CRAFTAX_FULL_ACTION_MAPPING,
)
from craftaxlm.full.state import render_craftax_text_custom
from craftaxlm.shared import (
    CraftaxBaseACI,
)


class CraftaxACI(CraftaxBaseACI):
    def make_env(self):
        return make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)

    def create_starting_obs(self):
        return {
            "state": render_craftax_text_custom(self.state).render_to_text_simple(
                verbose=self.verbose
            )
        }

    def map_action_string_to_int(self, action_string: str) -> int:
        return CRAFTAX_FULL_ACTION_MAPPING.get(action_string.lower(), 0)

    def get_achievements(self, state):
        return {
            "achievements": {
                k: state.achievements[i] for i, k in CRAFTAX_FULL_ACHIEVEMENTS.items()
            }
        }

    def create_step_info(self, state, reward, done):
        return {
            "state": render_craftax_text_custom(state).render_to_text_simple(
                verbose=self.verbose
            ),
            "reward": float(reward),
            "done": bool(done),
        }


if __name__ == "__main__":
    craftax_aci = CraftaxACI()
    action = 0
    import time

    ts = []
    for i in range(20):
        t0 = time.time()
        step_info = craftax_aci._step(action)
        ts.append(time.time() - t0)
        print("Took", ts[-1])
        if step_info["done"]:
            break
    print("Mean time", sum(ts) / len(ts))
    print("Q3 time", sorted(ts)[int(len(ts) * 0.75)])
    print("P90 time", sorted(ts)[int(len(ts) * 0.90)])
    print("P99 time", sorted(ts)[int(len(ts) * 0.99)])
