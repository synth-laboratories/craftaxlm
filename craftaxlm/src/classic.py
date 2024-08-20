from dataclasses import dataclass
from typing import Dict, List, Tuple, Dict, Literal
import jax
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import *
from craftax.craftax_env import make_craftax_env_from_name
from craftaxlm.src.shared import mob_id_to_name, CraftaxState, CraftaxBaseACI


classic_achievements = {
    0: "Collect Wood",
    1: "Place Table",
    2: "Eat Cow",
    3: "Collect Sapling",
    4: "Collect Drink",
    5: "Make Wood Pickaxe",
    6: "Make Wood Sword",
    7: "Place Plant",
    8: "Defeat Zombie",
    9: "Collect Stone",
    10: "Place Stone",
    11: "Eat Plant",
    12: "Defeat Skeleton",
    13: "Make Stone Pickaxe",
    14: "Make Stone Sword",
    15: "Wake Up",
    16: "Place Furnace",
    17: "Collect Coal",
    18: "Collect Iron",
    19: "Collect Diamond",
    20: "Make Iron Pickaxe",
    21: "Make Iron Sword",
}
classic_action_mapping = {
    "noop": 0,
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "do": 5,
    "sleep": 6,
    "place_stone": 7,
    "place_table": 8,
    "place_furnace": 9,
    "place_plant": 10,
    "make_wood_pickaxe": 11,
    "make_stone_pickaxe": 12,
    "make_iron_pickaxe": 13,
    "make_wood_sword": 14,
    "make_stone_sword": 15,
    "make_iron_sword": 16,
}


@dataclass
class CraftaxClassicState(CraftaxState):
    map: List[Dict]
    inventory: Dict
    player: Dict
    environment: Dict

    def render_map_to_text(self, ignore_distant_low_salience=True):
        backdrop_block_types = ["grass", "sand"]
        low_salience_objects = ["water", "stone", "tree", "wood", "path", "plant"]
        low_salience_mobs = []
        high_salience_objects = [
            "coal",
            "iron",
            "diamond",
            "crafting_table",
            "furnace",
            "lava",
            "ripe_plant",
        ]
        high_salience_mobs = ["Skeleton", "Zombie", "Cow", "Arrow"]

        unique_blocks = list(
            set([tile["block"] for tile in self.map if "block" in tile])
        )
        if not set(unique_blocks).issubset(
            set(backdrop_block_types + low_salience_objects + high_salience_objects)
        ):
            raise ValueError(
                f"Unknown block types: {set(unique_blocks) - set(backdrop_block_types+low_salience_objects+high_salience_objects)}"
            )
        unique_mobs = list(set([tile["mob"] for tile in self.map if "mob" in tile]))
        if not set(unique_mobs).issubset(set(low_salience_mobs + high_salience_mobs)):
            raise ValueError(
                f"Unknown mob types: {set(unique_mobs) - set(low_salience_mobs+high_salience_mobs)}"
            )

        def count_nearby_blocks(center_x, center_y, radius):
            block_counts = {}
            for tile in self.map:
                dx = abs(tile["position"]["x"] - center_x)
                dy = abs(tile["position"]["y"] - center_y)
                if dx <= radius and dy <= radius:
                    if not "block" in tile:
                        continue
                    block_type = tile["block"]
                    if block_type in backdrop_block_types:
                        block_counts[block_type] = block_counts.get(block_type, 0) + 1
            return block_counts

        center_x, center_y = 0, 0
        nearby_blocks = count_nearby_blocks(center_x, center_y, 2)
        backdrop = (
            max(nearby_blocks, key=nearby_blocks.get) if nearby_blocks else "unknown"
        )

        low_salience_objects.extend(
            [object for object in backdrop_block_types if object != backdrop]
        )

        def describe_xy(x, y):
            # THIS IS SO CONFUSING - X is UP and DOWN, Y is LEFT and RIGHT
            description = ""
            if x > 0:
                description += f"{x} steps down"
            elif x < 0:
                description += f"{-x} steps up"
            if (x != 0) and (y != 0):
                description += " and "
            if y < 0:
                description += f"{-y} steps left"
            elif y > 0:
                description += f"{y} steps right"
            return description

        periphery = []
        for direction in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            found = False
            for distance in range(1, 6):
                if found:
                    break
                x = center_x + direction[0] * distance
                y = center_y + direction[1] * distance
                for tile in self.map:
                    if tile["position"]["x"] == x and tile["position"]["y"] == y:
                        if tile["block"] in low_salience_objects:
                            periphery.append(
                                tile["block"].capitalize() + " is " + describe_xy(x, y)
                            )
                            if ignore_distant_low_salience:
                                found = True
                        if "mob" in tile and tile["mob"] in low_salience_mobs:
                            periphery.append(
                                "A " + tile["mob"] + " is " + describe_xy(x, y)
                            )
                            if ignore_distant_low_salience:
                                found = True

        high_salience = []
        for tile in self.map:
            if tile["visible"] and tile["block"] in high_salience_objects:
                high_salience.append(
                    tile["block"].capitalize()
                    + " is "
                    + describe_xy(tile["position"]["x"], tile["position"]["y"]),
                )
            elif (
                tile["visible"] and "mob" in tile and tile["mob"] in high_salience_mobs
            ):
                high_salience.append(
                    "A "
                    + tile["mob"]
                    + " is "
                    + describe_xy(tile["position"]["x"], tile["position"]["y"])
                )
        facing_and_position = {
            "up": "is one steps up",
            "down": "is one steps down",
            "left": "is one steps left",
            "right": "is one steps right",
        }
        return {
            "terrain_underneath_you": backdrop,
            "surroundings": periphery + high_salience,
            "object_you_are_facing": (
                [
                    surrounding_object.split(" is")[0]
                    for surrounding_object in periphery + high_salience
                    if facing_and_position[self.player["direction_facing"]]
                    in surrounding_object
                ]
                + ["No object directly in front of you"]
            )[0],
        }

    def render_inventory_to_text(self, include_absent_inventory=True):
        def process_inventory(inv_dict):
            processed = {}
            for key, value in inv_dict.items():
                if isinstance(value, dict):
                    processed_sub = process_inventory(value)
                    if processed_sub:
                        processed[key] = processed_sub
                elif include_absent_inventory or value > 0:
                    processed[key] = value
            return processed

        return process_inventory(self.inventory)

    def render_environment_to_text(self, include_absent_environment_attributes=True):
        attributes_to_hide = []
        environment = {}
        for key, value in self.environment.items():
            if key not in attributes_to_hide and (
                include_absent_environment_attributes or value
            ):
                environment[key] = value
        return environment


def render_craftax_classic_text_custom(state: EnvState) -> CraftaxClassicState:
    map_data = []
    for x in range(state.map.shape[0]):
        for y in range(state.map.shape[1]):
            if (
                not max(
                    abs(x - state.player_position[0]), abs(y - state.player_position[1])
                )
                <= 4
            ):
                continue
            tile = {
                "position": {
                    "x": x - state.player_position[0],
                    "y": y - state.player_position[1],
                },
                "visible": max(
                    abs(x - state.player_position[0]), abs(y - state.player_position[1])
                )
                <= 4,
                "block": BlockType(state.map[x, y]).name.lower(),
            }
            if state.mob_map[x, y].max() > 0.5:
                tile["mob"] = mob_id_to_name(state.mob_map[x, y].argmax())
            map_data.append(tile)

    inventory_data = {
        "resources": {
            "wood": int(state.inventory.wood),
            "stone": int(state.inventory.stone),
            "coal": int(state.inventory.coal),
            "iron": int(state.inventory.iron),
            "diamond": int(state.inventory.diamond),
            "sapling": int(state.inventory.sapling),
        },
        "tools": {},
    }

    inventory_data["tools"]["pickaxe"] = {
        "wood": int(state.inventory.wood_pickaxe),
        "stone": int(state.inventory.stone_pickaxe),
        "iron": int(state.inventory.iron_pickaxe),
    }

    inventory_data["tools"]["sword"] = {
        "wood": int(state.inventory.wood_sword),
        "stone": int(state.inventory.stone_sword),
        "iron": int(state.inventory.iron_sword),
    }

    player_data = {
        "health": int(state.player_health),
        "food": int(state.player_food),
        "drink": int(state.player_drink),
        "energy": int(state.player_energy),
        "direction_facing": Action(state.player_direction).name.lower(),
    }

    environment_data = {
        "light_level": float(state.light_level),
        "is_sleeping": bool(state.is_sleeping),
        "floor": 0,  # Assuming single floor for now
    }

    def to_json_friendly(data):
        if isinstance(data, dict):
            return {key: to_json_friendly(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [to_json_friendly(item) for item in data]
        elif isinstance(data, (jnp.ndarray, np.ndarray)):
            if data.ndim == 0:
                return data.item()
            return [to_json_friendly(item) for item in data]
        elif isinstance(data, (jnp.bool_, np.bool_)):
            return bool(data)
        elif isinstance(data, (jnp.integer, np.integer)):
            return int(data)
        elif isinstance(data, (jnp.floating, np.floating)):
            return float(data)
        else:
            return data

    return CraftaxClassicState(
        map=to_json_friendly(map_data),
        inventory=to_json_friendly(inventory_data),
        player=to_json_friendly(player_data),
        environment=to_json_friendly(environment_data),
    )


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
        return classic_action_mapping.get(action_string.lower(), 0)

    def get_achievements(self, state):
        return {
            "achievements": {
                k: state.achievements[i] for i, k in classic_achievements.items()
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
