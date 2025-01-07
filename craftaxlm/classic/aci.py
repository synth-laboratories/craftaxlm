from craftax.craftax.constants import *
from craftax.craftax_env import make_craftax_env_from_name

from craftaxlm.classic.metadata import (
    CRAFTAX_CLASSIC_ACHIEVEMENTS,
    CRAFTAX_CLASSIC_ACTION_MAPPING,
)
from craftaxlm.classic.state import (
    render_craftax_classic_text_custom,
)
from craftaxlm.shared import CraftaxBaseACI


class CraftaxClassicACI(CraftaxBaseACI):
    def make_env(self):
        return make_craftax_env_from_name(
            "Craftax-Classic-Symbolic-v1", auto_reset=False
        )

    def create_starting_obs(self):
        return {
            "state": render_craftax_classic_text_custom(
                self.state
            ).render_to_text_simple(verbose=self.verbose),
            "reward": 0.0,
            "done": False,
        }

    def map_action_string_to_int(self, action_string: str) -> int:
        return CRAFTAX_CLASSIC_ACTION_MAPPING.get(action_string.lower(), 0)

    def get_achievements(self, state):
        return {
            "achievements": {
                k: state.achievements[i]
                for i, k in CRAFTAX_CLASSIC_ACHIEVEMENTS.items()
            }
        }

    def create_step_info(self, state, reward, done):
        return {
            "state": render_craftax_classic_text_custom(state).render_to_text_simple(
                verbose=self.verbose
            ),
            "reward": float(reward),
            "done": bool(done),
        }


if __name__ == "__main__":
    craftax_aci = CraftaxClassicACI()
    action = 0
    step_info = craftax_aci._step(action)
