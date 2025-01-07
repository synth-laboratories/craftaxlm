from dataclasses import dataclass
from typing import Dict, List, Literal
from abc import abstractclassmethod
from typing import List, Tuple, Dict
import jax


def mob_id_to_name(id):
    if id == 0:
        return "Zombie"
    elif id == 1:
        return "Gnome Warrior"
    elif id == 2:
        return "Orc Soldier"
    elif id == 3:
        return "Lizard"
    elif id == 4:
        return "Knight"
    elif id == 5:
        return "Troll"
    elif id == 6:
        return "Pigman"
    elif id == 7:
        return "Frost Troll"
    elif id == 8:
        return "Cow"
    elif id == 9:
        return "Bat"
    elif id == 10:
        return "Snail"
    elif id == 16:
        return "Skeleton"
    elif id == 17:
        return "Gnome Archer"
    elif id == 18:
        return "Orc Mage"
    elif id == 19:
        return "Kobold"
    elif id == 20:
        return "Archer"
    elif id == 21:
        return "Deep Thing"
    elif id == 22:
        return "Fire Elemental"
    elif id == 23:
        return "Ice Elemental"
    elif id == 24:
        return "Arrow"
    elif id == 25:
        return "Dagger"
    elif id == 26:
        return "Fireball"
    elif id == 27:
        return "Iceball"
    elif id == 28:
        return "Arrow"
    elif id == 29:
        return "Slimeball"
    elif id == 30:
        return "Fireball"
    elif id == 31:
        return "Iceball"
    elif id == 32:
        return "Arrow (Player)"
    elif id == 33:
        return "Dagger (Player)"
    elif id == 34:
        return "Fireball (Player)"
    elif id == 35:
        return "Iceball (Player)"
    elif id == 36:
        return "Arrow (Player)"
    elif id == 37:
        return "Slimeball (Player)"
    elif id == 38:
        return "Fireball (Player)"
    elif id == 39:
        return "Iceball (Player)"


def level_to_material(level):
    if level == 1:
        return "Wood"
    elif level == 2:
        return "Stone"
    elif level == 3:
        return "Iron"
    elif level == 4:
        return "Diamond"


def level_to_enchantment(level):
    if level == 0:
        return "No"
    if level == 1:
        return "Fire"
    elif level == 2:
        return "Ice"


def get_armour_level(level):
    if level == 1:
        return "Iron"
    elif level == 2:
        return "Diamond"


@dataclass
class CraftaxState:
    map: List[Dict]
    inventory: Dict
    player: Dict
    environment: Dict

    @abstractclassmethod
    def render_map_to_text(self, ignore_distant_low_salience=True):
        pass

    @abstractclassmethod
    def render_inventory_to_text(self, include_absent_inventory=True):
        pass

    @abstractclassmethod
    def render_environment_to_text(self, include_absent_environment_attributes=True):
        pass

    def render_json_to_text_via_md(self, json_data: Dict) -> str:
        keys_to_not_format = [
            "resources",
            "tools",
            "potions",
            "armor",
            "player",
            "environment",
        ]

        def format_as_list(data, level):
            list_output = ""
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, dict):
                        list_output += (
                            f"{' ' * level}- {k}:\n{format_as_list(v, level + 2)}"
                        )
                    else:
                        list_output += f"{' ' * level}- {k}: {v}\n"
            elif isinstance(data, list):
                for item in data:
                    list_output += f"{' ' * level}- {item}\n"
            else:
                list_output += f"{' ' * level}- {data}\n"
            return list_output

        def dict_to_md(data, level=1, parent_key=""):
            md_output = ""
            for key, value in data.items():
                full_key = f"{parent_key}.{key}" if parent_key else key
                md_output += f"{'#' * level} {key.capitalize()}\n"
                if key in keys_to_not_format:
                    md_output += format_as_list(value, level)
                elif isinstance(value, dict):
                    md_output += dict_to_md(value, level + 1, full_key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            md_output += dict_to_md(item, level + 1, full_key)
                        else:
                            md_output += f"- {item}\n"
                else:
                    md_output += f"{value}\n"
            return md_output

        return dict_to_md(json_data).strip()

    def render_json_to_text_via_xml(self, json_data: Dict) -> str:
        def dict_to_xml(data):
            xml_output = ""
            for key, value in data.items():
                xml_output += f"<{key}>"
                if isinstance(value, dict):
                    xml_output += dict_to_xml(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml_output += dict_to_xml(item)
                        else:
                            xml_output += str(item)
                else:
                    xml_output += str(value)
                xml_output += f"</{key}>"
            return xml_output

        return dict_to_xml(json_data).strip()

    def render_to_text_simple(
        self, verbose=True, formatting: Literal["md", "xml"] = "md"
    ) -> str:
        rendered_map = self.render_map_to_text(ignore_distant_low_salience=not verbose)
        rendered_inventory = self.render_inventory_to_text(
            include_absent_inventory=verbose
        )
        rendered_environment = self.render_environment_to_text(
            include_absent_environment_attributes=verbose
        )
        if formatting == "md":
            return self.render_json_to_text_via_md(
                {
                    "map": rendered_map,
                    "inventory": rendered_inventory,
                    "player": self.player,
                    "environment": rendered_environment,
                }
            )
        elif formatting == "xml":
            return self.render_json_to_text_via_xml(
                {
                    "map": rendered_map,
                    "inventory": rendered_inventory,
                    "player": self.player,
                    "environment": rendered_environment,
                }
            )
        else:
            raise ValueError(f"Unknown formatting: {formatting}")


class CraftaxBaseACI:
    def __init__(self, seed=0, actions_to_start_with: List[int] = [], verbose=True):
        self.verbose = verbose
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        self.rngs = jax.random.split(_rng, 3)
        self.reset()
        if actions_to_start_with:
            self.go_forward(actions_to_start_with)

    def reset(self):
        self.env = self.make_env()
        self.env_params = self.env.default_params
        obs, self.state = self.env.reset(self.rngs[0], self.env_params)
        self.starting_obs = self.create_starting_obs()
        self.action_history = []
        self.achievements = {}
        self.achievement_deltas = []

    def make_env(self):
        pass

    def create_starting_obs(self):
        pass

    def go_back(self, n_steps):
        actions = self.action_history[-n_steps:]
        self.reset()
        return self.go_forward(actions)

    def go_forward(self, actions):
        step_info = {}
        for action in actions:
            step_info = self._step(action)
        return step_info

    def render_achivements(self, info):
        achievements = {}
        for key, value in info.items():
            if key.startswith("Achievements/"):
                achievement_name = key.split("/")[-1]
                achievements[achievement_name] = float(value)
        return {
            "achievements": achievements,
            "discount": float(info.get("discount", 1.0)),
        }

    def get_achievement_delta(self, achievements):
        delta = []
        for key, value in achievements["achievements"].items():
            if key in self.achievements:
                if value > self.achievements[key]:
                    delta.append(key)
        self.achievements = achievements["achievements"]
        return delta

    def map_action_string_to_int(self, action_string: str) -> int:
        pass

    def _step(self, action):
        _, state, reward, done, info = self.env.step(
            self.rngs[2], self.state, action, self.env_params
        )
        achievements = self.get_achievements(state)
        achievement_delta = self.get_achievement_delta(achievements)
        if achievement_delta and self.verbose:
            print(achievement_delta)
        self.achievement_deltas.append(achievement_delta)
        self.state = state
        step_info = self.create_step_info(state, reward, done)
        return step_info

    @abstractclassmethod
    def get_achievements(self, state):
        pass

    @abstractclassmethod
    def create_step_info(self, state, reward, done):
        pass

    def multistep(self, actions: List[str]) -> Tuple[List[Dict], List[float], bool]:
        done = False
        step_infos = []
        rewards = []
        for action in actions:
            step_info = self._step(self.map_action_string_to_int(action))
            step_infos.append(step_info)
            rewards.append(step_info["reward"])
            done = step_info["done"]
            if done:
                break
        return step_infos, rewards, done

    def terminate(self):
        return self.achievements