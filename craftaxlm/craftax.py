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

craftax_action_mapping = {
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
    "rest": 17,
    "descend": 18,
    "ascend": 19,
    "make_diamond_pickaxe": 20,
    "make_diamond_sword": 21,
    "make_iron_armour": 22,
    "make_diamond_armour": 23,
    "shoot_arrow": 24,
    "make_arrow": 25,
    "cast_fireball": 26,
    "cast_iceball": 27,
    "place_torch": 28,
    "drink_potion_red": 29,
    "drink_potion_green": 30,
    "drink_potion_blue": 31,
    "drink_potion_pink": 32,
    "drink_potion_cyan": 33,
    "drink_potion_yellow": 34,
    "read_book": 35,
    "enchant_sword": 36,
    "enchant_armour": 37,
    "make_torch": 38,
    "level_up_dexterity": 39,
    "level_up_strength": 40,
    "level_up_intelligence": 41,
    "enchant_bow": 42,
}

craftax_achievements = {
    0: "collect_wood",
    1: "place_table",
    2: "eat_cow",
    3: "collect_sapling",
    4: "collect_drink",
    5: "make_wood_pickaxe",
    6: "make_wood_sword",
    7: "place_plant",
    8: "defeat_zombie",
    9: "collect_stone",
    10: "place_stone",
    11: "eat_plant",
    12: "defeat_skeleton",
    13: "make_stone_pickaxe",
    14: "make_stone_sword",
    15: "wake_up",
    16: "place_furnace",
    17: "collect_coal",
    18: "collect_iron",
    19: "collect_diamond",
    20: "make_iron_pickaxe",
    21: "make_iron_sword",
    22: "make_arrow",
    23: "make_torch",
    24: "place_torch",
    25: "make_diamond_sword",
    26: "make_iron_armour",
    27: "make_diamond_armour",
    28: "enter_gnomish_mines",
    29: "enter_dungeon",
    30: "enter_sewers",
    31: "enter_vault",
    32: "enter_troll_mines",
    33: "enter_fire_realm",
    34: "enter_ice_realm",
    35: "enter_graveyard",
    36: "defeat_gnome_warrior",
    37: "defeat_gnome_archer",
    38: "defeat_orc_solider",
    39: "defeat_orc_mage",
    40: "defeat_lizard",
    41: "defeat_kobold",
    42: "defeat_troll",
    43: "defeat_deep_thing",
    44: "defeat_pigman",
    45: "defeat_fire_elemental",
    46: "defeat_frost_troll",
    47: "defeat_ice_elemental",
    48: "damage_necromancer",
    49: "defeat_necromancer",
    50: "eat_bat",
    51: "eat_snail",
    52: "find_bow",
    53: "fire_bow",
    54: "collect_sapphire",
    55: "learn_fireball",
    56: "cast_fireball",
    57: "learn_iceball",
    58: "cast_iceball",
    59: "collect_ruby",
    60: "make_diamond_pickaxe",
    61: "open_chest",
    62: "drink_potion",
    63: "enchant_sword",
    64: "enchant_armour",
    65: "defeat_knight",
    66: "defeat_archer",
}


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
        ]
        high_salience_items = []
        high_salience_mobs = [
            "Zombie",
            "Gnome Warrior",
            "Orc Soldier",
            "Lizard",
            "Knight",
            "Troll",
            "Pigman",
            "Frost Troll",
            "Cow",
            "Bat",
            "Snail",
            "Skeleton",
            "Gnome Archer",
            "Orc Mage",
            "Kobold",
            "Archer",
            "Deep Thing",
            "Fire Elemental",
            "Ice Elemental",
            "Arrow",
            "Dagger",
            "Fireball",
            "Iceball",
            "Slimeball",
            "Arrow (Player)",
            "Dagger (Player)",
            "Fireball (Player)",
            "Iceball (Player)",
            "Slimeball (Player)",
        ]

        unique_blocks = list(set([tile["block"] for tile in self.map if "block" in tile]))
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
            for distance in range(1, 6):
                x = center_x + direction[0] * distance
                y = center_y + direction[1] * distance
                for tile in self.map:
                    if not tile["visible"]:
                        continue
                    if tile["position"]["x"] == x and tile["position"]["y"] == y:
                        if tile["block"] in low_salience_objects:
                            periphery.append(
                                f"{tile['block'].capitalize()} is {describe_xy(x, y)}"
                            )
                            if ignore_distant_low_salience:
                                break
                        if "mob" in tile and tile["mob"] in low_salience_mobs:
                            periphery.append(f"A {tile['mob']} is {describe_xy(x, y)}")
                            if ignore_distant_low_salience:
                                break
                        if "item" in tile and tile["item"] in low_salience_items:
                            periphery.append(f"A {tile['item']} is {describe_xy(x, y)}")
                            if ignore_distant_low_salience:
                                break

        high_salience = []
        for tile in self.map:
            if not tile["visible"]:
                continue
            if "mob" in tile and tile["mob"] in high_salience_mobs:
                high_salience.append(
                    f"A {tile['mob']} is {describe_xy(tile['position']['x'], tile['position']['y'])}"
                )
            if "item" in tile and tile["item"] in high_salience_items:
                high_salience.append(
                    f"A {tile['item']} is {describe_xy(tile['position']['x'], tile['position']['y'])}"
                )
            if tile["block"] in high_salience_objects:
                high_salience.append(
                    f"{tile['block'].capitalize()} is {describe_xy(tile['position']['x'], tile['position']['y'])}"
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
                    if facing_and_position[self.player["direction"]]
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
                    processed[key] = process_inventory(value)
                elif include_absent_inventory or value > 0:
                    processed[key] = value
            return processed

        return process_inventory(self.inventory)

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

    # REALLY NEEDS TO BE SPED UP:
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
    def __init__(self, seed=0, actions_to_start_with: List[int] = [], verbose=True):
        self.verbose = verbose
        rng = jax.random.PRNGKey(seed)
        rng, _rng = jax.random.split(rng)
        self.rngs = jax.random.split(_rng, 3)
        self.reset()
        if actions_to_start_with:
            self.go_forward(actions_to_start_with)

    def reset(self):
        self.env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
        self.env_params = self.env.default_params
        obs, self.state = self.env.reset(self.rngs[0], self.env_params)
        self.starting_obs = {
            "state": render_craftax_text_custom(self.state).render_to_text_simple(
                verbose=self.verbose
            )
        }
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
        delta = []
        for key, value in achievements["achievements"].items():
            if key in self.achievements:
                if value > self.achievements[key]:
                    delta.append(key)
        self.achievements = achievements["achievements"]
        return delta

    def map_action_string_to_int(self, action_string: str) -> int:
        return craftax_action_mapping.get(action_string.lower(), 0)

    def _step(self, action):
        _, state, reward, done, info = self.env.step(
            self.rngs[2], self.state, action, self.env_params
        )
        achievements = {
            "achievements": {
                k: state.achievements[i] for i, k in craftax_achievements.items()
            }
        }
        achievement_delta = self.get_achievement_delta(achievements)
        self.achievement_deltas.append(achievement_delta)
        self.state = state

        step_info = {
            "state": render_craftax_text_custom(state).render_to_text_simple(
                verbose=self.verbose
            ),
            "reward": float(reward),
            "done": bool(done),
        }
        return step_info

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
