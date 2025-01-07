from dataclasses import dataclass
from typing import Dict, List

import jax
from craftax.craftax.constants import *
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.util.game_logic_utils import is_boss_vulnerable

from craftaxlm.full.metadata import (
    CRAFTAX_FULL_BACKDROP_BLOCK_TYPES,
    CRAFTAX_FULL_HIGH_SALIENCE_ITEMS,
    CRAFTAX_FULL_HIGH_SALIENCE_MOBS,
    CRAFTAX_FULL_HIGH_SALIENCE_OBJECTS,
    CRAFTAX_FULL_LOW_SALIENCE_ITEMS,
    CRAFTAX_FULL_LOW_SALIENCE_MOBS,
    CRAFTAX_FULL_LOW_SALIENCE_OBJECTS,
)
from craftaxlm.shared import (
    CraftaxState,
    get_armour_level,
    level_to_enchantment,
    level_to_material,
    mob_id_to_name,
)


@dataclass
class CraftaxFullState(CraftaxState):
    map: List[Dict]
    inventory: Dict
    player: Dict
    environment: Dict

    def render_map_to_text(self, ignore_distant_low_salience=True):
        unique_blocks = list(
            set([tile["block"] for tile in self.map if "block" in tile])
        )
        if not set(unique_blocks).issubset(
            set(
                CRAFTAX_FULL_BACKDROP_BLOCK_TYPES
                + CRAFTAX_FULL_LOW_SALIENCE_OBJECTS
                + CRAFTAX_FULL_HIGH_SALIENCE_OBJECTS
            )
        ):
            raise ValueError(
                f"Unknown block types: {set(unique_blocks) - set(CRAFTAX_FULL_BACKDROP_BLOCK_TYPES + CRAFTAX_FULL_LOW_SALIENCE_OBJECTS + CRAFTAX_FULL_HIGH_SALIENCE_OBJECTS)}"
            )
        unique_mobs = list(set([tile["mob"] for tile in self.map if "mob" in tile]))
        if not set(unique_mobs).issubset(
            set(CRAFTAX_FULL_LOW_SALIENCE_MOBS + CRAFTAX_FULL_HIGH_SALIENCE_MOBS)
        ):
            raise ValueError(
                f"Unknown mob types: {set(unique_mobs) - set(CRAFTAX_FULL_LOW_SALIENCE_MOBS + CRAFTAX_FULL_HIGH_SALIENCE_MOBS)}"
            )
        unique_items = list(set([tile["item"] for tile in self.map if "item" in tile]))
        if not set(unique_items).issubset(
            set(CRAFTAX_FULL_LOW_SALIENCE_ITEMS + CRAFTAX_FULL_HIGH_SALIENCE_ITEMS)
        ):
            raise ValueError(
                f"Unknown item types: {set(unique_items) - set(CRAFTAX_FULL_LOW_SALIENCE_ITEMS + CRAFTAX_FULL_HIGH_SALIENCE_ITEMS)}"
            )

        def count_nearby_blocks(center_x, center_y, radius):
            block_counts = {}
            for tile in self.map:
                dx = abs(tile["position"]["x"] - center_x)
                dy = abs(tile["position"]["y"] - center_y)
                if dx <= radius and dy <= radius:
                    if "block" not in tile:
                        continue
                    block_type = tile["block"]
                    if block_type in CRAFTAX_FULL_BACKDROP_BLOCK_TYPES:
                        block_counts[block_type] = block_counts.get(block_type, 0) + 1
            return block_counts

        center_x, center_y = 0, 0
        nearby_blocks = count_nearby_blocks(center_x, center_y, 2)
        backdrop = (
            max(nearby_blocks, key=nearby_blocks.get) if nearby_blocks else "unknown"
        )

        low_salience_objects = CRAFTAX_FULL_LOW_SALIENCE_OBJECTS.copy()
        low_salience_objects.extend(
            [obj for obj in CRAFTAX_FULL_BACKDROP_BLOCK_TYPES if obj != backdrop]
        )

        def describe_xy(x, y):
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
                        if (
                            "mob" in tile
                            and tile["mob"] in CRAFTAX_FULL_LOW_SALIENCE_MOBS
                        ):
                            periphery.append(f"A {tile['mob']} is {describe_xy(x, y)}")
                            if ignore_distant_low_salience:
                                break
                        if (
                            "item" in tile
                            and tile["item"] in CRAFTAX_FULL_LOW_SALIENCE_ITEMS
                        ):
                            periphery.append(f"A {tile['item']} is {describe_xy(x, y)}")
                            if ignore_distant_low_salience:
                                break

        high_salience = []
        for tile in self.map:
            if not tile["visible"]:
                continue
            if "mob" in tile and tile["mob"] in CRAFTAX_FULL_HIGH_SALIENCE_MOBS:
                high_salience.append(
                    f"A {tile['mob']} is {describe_xy(tile['position']['x'], tile['position']['y'])}"
                )
            if "item" in tile and tile["item"] in CRAFTAX_FULL_HIGH_SALIENCE_ITEMS:
                high_salience.append(
                    f"A {tile['item']} is {describe_xy(tile['position']['x'], tile['position']['y'])}"
                )
            if tile["block"] in CRAFTAX_FULL_HIGH_SALIENCE_OBJECTS:
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
                elif isinstance(value, bool) and (include_absent_inventory or value):
                    processed[key] = value
                elif include_absent_inventory or value:
                    processed[key] = value is not None
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


def render_craftax_text_custom(state: EnvState) -> CraftaxFullState:
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

    return CraftaxFullState(
        map=to_json_friendly(map_data),
        inventory=to_json_friendly(inventory_data),
        player=to_json_friendly(player_data),
        environment=to_json_friendly(environment_data),
    )
