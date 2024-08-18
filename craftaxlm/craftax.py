import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Dict, Literal
import jax
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import *
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable
import json
from craftaxlm.shared import (
    mob_id_to_name,
    level_to_material,
    level_to_enchantment,
    get_armour_level,
)


@dataclass
class CraftaxState:
    map: List[Dict]
    inventory: Dict
    player: Dict
    environment: Dict

    def render_map_to_text(self, ignore_distant_low_salience=True):
        backdrop_block_types = [
            "grass",
            "sand",
            "path",
            "fire_grass",
            "ice_grass",
            "gravel",
        ]
        low_salience_objects = [
            "stone",
            "tree",
            "wood",
            "plant",
            "wall",
            "darkness",
            "wall_moss",
            "stalagmite",
            "fire_tree",
            "ice_shrub",
            "grave",
            "grave2",
            "grave3",
        ]
        low_salience_mobs = []
        low_salience_items = []
        high_salience_objects = [
            "water",
            "lava",
            "coal",
            "iron",
            "diamond",
            "sapphire",
            "ruby",
            "crafting_table",
            "furnace",
            "ripe_plant",
            "chest",
            "fountain",
            "enchantment_table_fire",
            "enchantment_table_ice",
            "necromancer",
            "necromancer_vulnerable",
            "skeleton",
            "zombie",
        ]
        high_salience_items = []

        unique_blocks = list(set([tile["block"] for tile in self.map]))
        if not set(unique_blocks).issubset(
            set(backdrop_block_types + low_salience_objects + high_salience_objects)
        ):
            raise ValueError(
                f"Unknown block types: {set(unique_blocks) - set(backdrop_block_types+low_salience_objects+high_salience_objects)}"
            )
        unique_mobs = list(set([tile["mob"] for tile in self.map if "mob" in tile]))
        if not set(unique_mobs).issubset(
            set(low_salience_mobs + high_salience_objects)
        ):
            raise ValueError(
                f"Unknown mob types: {set(unique_mobs) - set(low_salience_mobs+high_salience_objects)}"
            )
        unique_items = list(set([tile["item"] for tile in self.map if "item" in tile]))
        if not set(unique_items).issubset(
            set(low_salience_items + high_salience_items)
        ):
            raise ValueError(
                f"Unknown item types: {set(unique_items) - set(low_salience_items+high_salience_items)}"
            )

        def count_nearby_blocks(center_x, center_y, radius):
            block_counts = {}
            for tile in self.map:
                dx = abs(tile["position"]["x"] - center_x)
                dy = abs(tile["position"]["y"] - center_y)
                if dx <= radius and dy <= radius:
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
            description = ""
            if x < 0:
                description += f"{-x} steps east"
            elif x > 0:
                description += f"{x} steps west"
            if (x != 0) and (y != 0):
                description += " and "
            if y < 0:
                description += f"{-y} steps south"
            elif y > 0:
                description += f"{y} steps north"
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
            for distance in range(1, 6):
                x = center_x + direction[0] * distance
                y = center_y + direction[1] * distance
                for tile in self.map:
                    if not tile["visible"]:
                        continue
                    if tile["position"]["x"] == x and tile["position"]["y"] == y:
                        if "mob" in tile and tile["mob"] in low_salience_mobs:
                            periphery.append((tile["mob"], describe_xy(x, y)))
                            if ignore_distant_low_salience:
                                break
                        if tile["item"] in low_salience_items:
                            periphery.append((tile["item"], describe_xy(x, y)))
                            if ignore_distant_low_salience:
                                break
                        if tile["block"] in low_salience_objects:
                            periphery.append((tile["block"], describe_xy(x, y)))
                            if ignore_distant_low_salience:
                                break

        high_salience = []
        for tile in self.map:
            if not tile["visible"]:
                continue
            if "mob" in tile and tile["mob"] in high_salience_mobs:
                high_salience.append(
                    (
                        tile["mob"],
                        describe_xy(tile["position"]["x"], tile["position"]["y"]),
                    )
                )
            if tile["item"] in high_salience_items:
                high_salience.append(
                    (
                        tile["item"],
                        describe_xy(tile["position"]["x"], tile["position"]["y"]),
                    )
                )
            if tile["block"] in high_salience_objects:
                high_salience.append(
                    (
                        tile["block"],
                        describe_xy(tile["position"]["x"], tile["position"]["y"]),
                    )
                )

        return {
            "backdrop": backdrop,
            "periphery": periphery,
            "high_salience": high_salience,
        }

    def render_inventory_to_text(self, include_absent_inventory=True):
        inventory = {}
        for key, value in self.inventory.items():
            if include_absent_inventory or value > 0:
                inventory[key] = value
        return inventory

    def render_environment_to_text(self, include_absent_environment_attributes=True):
        attributes_to_hide = [
            "learned_fireball",
            "learned_iceball",
            "floor",
            "ladder_open",
            "is_boss_vulnerable",
        ]
        environment = {}
        for key, value in self.environment.items():
            if key not in attributes_to_hide and (
                include_absent_environment_attributes or value
            ):
                environment[key] = value
        return environment

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


def render_craftax_text_custom(state: EnvState) -> CraftaxState:
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)
    map = state.map[state.player_level]
    padded_grid = jnp.pad(
        map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )
    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    padded_items_map = jnp.pad(
        state.item_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=ItemType.NONE.value,
    )
    item_map_view = jax.lax.dynamic_slice(padded_items_map, tl_corner, OBS_DIM)

    mob_types_per_class = 8
    mob_map = jnp.zeros((*OBS_DIM, 5 * mob_types_per_class), dtype=jnp.int32)

    def _add_mob_to_map(carry, mob_index):
        mob_map, mobs, mob_class_index = carry
        local_position = (
            mobs.position[mob_index]
            - state.player_position
            + jnp.array([OBS_DIM[0], OBS_DIM[1]]) // 2
        )
        on_screen = jnp.logical_and(
            local_position >= 0, local_position < jnp.array([OBS_DIM[0], OBS_DIM[1]])
        ).all()
        on_screen *= mobs.mask[mob_index]
        mob_identifier = mob_class_index * mob_types_per_class + mobs.type_id[mob_index]
        mob_map = mob_map.at[local_position[0], local_position[1], mob_identifier].set(
            on_screen.astype(jnp.int32)
        )
        return (mob_map, mobs, mob_class_index), None

    for mob_type, mob_class_index in [
        (state.melee_mobs, 0),
        (state.passive_mobs, 1),
        (state.ranged_mobs, 2),
        (state.mob_projectiles, 3),
        (state.player_projectiles, 4),
    ]:
        (mob_map, _, _), _ = jax.lax.scan(
            _add_mob_to_map,
            (
                mob_map,
                jax.tree_util.tree_map(lambda x: x[state.player_level], mob_type),
                mob_class_index,
            ),
            jnp.arange(mob_type.mask.shape[1]),
        )

    padded_light_map = jnp.pad(
        state.light_map[state.player_level],
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=0.0,
    )
    light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, OBS_DIM) > 0.05

    map_data = []
    for x in range(OBS_DIM[0]):
        for y in range(OBS_DIM[1]):
            tile = {
                "position": {"x": y - OBS_DIM[1] // 2, "y": x - OBS_DIM[0] // 2},
                "visible": bool(light_map_view[x, y]),
            }
            if light_map_view[x, y]:
                if mob_map[x, y].max() > 0.5:
                    tile["mob"] = mob_id_to_name(mob_map[x, y].argmax())
                if item_map_view[x, y] != ItemType.NONE.value:
                    tile["item"] = ItemType(item_map_view[x, y]).name.lower()
                tile["block"] = BlockType(map_view[x, y]).name.lower()
            map_data.append(tile)

    inventory_data = {
        "resources": {
            "wood": state.inventory.wood,
            "stone": state.inventory.stone,
            "coal": state.inventory.coal,
            "iron": state.inventory.iron,
            "diamond": state.inventory.diamond,
            "sapphire": state.inventory.sapphire,
            "ruby": state.inventory.ruby,
            "sapling": state.inventory.sapling,
            "torch": state.inventory.torches,
            "arrow": state.inventory.arrows,
            "book": state.inventory.books,
        },
        "tools": {},
        "potions": {
            "red": state.inventory.potions[0],
            "green": state.inventory.potions[1],
            "blue": state.inventory.potions[2],
            "pink": state.inventory.potions[3],
            "cyan": state.inventory.potions[4],
            "yellow": state.inventory.potions[5],
        },
        "armor": {},
    }

    if state.inventory.pickaxe > 0:
        inventory_data["tools"]["pickaxe"] = {
            "material": level_to_material(state.inventory.pickaxe)
        }
    if state.inventory.sword > 0:
        inventory_data["tools"]["sword"] = {
            "material": level_to_material(state.inventory.sword),
            "enchantment": level_to_enchantment(state.sword_enchantment),
        }
    if state.inventory.bow > 0:
        inventory_data["tools"]["bow"] = {
            "enchantment": level_to_enchantment(state.bow_enchantment)
        }

    for i, piece in enumerate(["helmet", "chestplate", "leggings", "boots"]):
        if state.inventory.armour[i] > 0:
            inventory_data["armor"][piece] = {
                "material": get_armour_level(state.inventory.armour[i]),
                "enchantment": level_to_enchantment(state.armour_enchantments[i]),
            }

    player_data = {
        "health": state.player_health,
        "food": state.player_food,
        "drink": state.player_drink,
        "energy": state.player_energy,
        "mana": state.player_mana,
        "xp": state.player_xp,
        "dexterity": state.player_dexterity,
        "strength": state.player_strength,
        "intelligence": state.player_intelligence,
        "direction": Action(state.player_direction).name.lower(),
    }

    environment_data = {
        "light_level": state.light_level,
        "is_sleeping": state.is_sleeping,
        "is_resting": state.is_resting,
        "learned_fireball": state.learned_spells[0],
        "learned_iceball": state.learned_spells[1],
        "floor": state.player_level,
        "ladder_open": state.monsters_killed[state.player_level]
        >= MONSTERS_KILLED_TO_CLEAR_LEVEL,
        "is_boss_vulnerable": is_boss_vulnerable(state),
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

    return CraftaxState(
        map=to_json_friendly(map_data),
        inventory=to_json_friendly(inventory_data),
        player=to_json_friendly(player_data),
        environment=to_json_friendly(environment_data),
    )


class CraftaxACI:
    def __init__(self, seed=0, actions_to_start_with: List[int] = []):
        rng = jax.random.PRNGKey(0)
        rng, _rng = jax.random.split(rng)
        self.rngs = jax.random.split(_rng, 3)
        self.reset()
        if actions_to_start_with:
            self.go_forward(actions_to_start_with)

    def reset(self):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.env_params = self.env.default_params
        obs, self.state = self.env.reset(self.rngs[0], self.env_params)
        self.starting_obs = {"raw_obs": obs}
        self.action_history = []
        self.achievements = {}
        self.achievement_deltas = []

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
        delta = {}
        for key, value in achievements["achievements"].items():
            if key in self.achievements:
                if value > self.achievements[key]:
                    delta[key] = value - self.achievements[key]
            else:
                if value > 0:
                    delta[key] = value
        self.achievements = achievements["achievements"]
        return delta

    def _step(self, action):
        _, state, reward, done, info = self.env.step(
            self.rngs[2], self.state, action, self.env_params
        )
        achievements = self.render_achivements(info)
        achievement_delta = self.get_achievement_delta(achievements)
        self.achievement_deltas.append(achievement_delta)
        self.state = state

        step_info = {
            "state": render_craftax_text_custom(state),
            "reward": float(reward),
            "done": bool(done),
        }
        return step_info


if __name__ == "__main__":
    craftax_aci = CraftaxACI()
    action = 0
    step_info = craftax_aci._step(action)
    print(step_info["state"].render_to_text_simple())
