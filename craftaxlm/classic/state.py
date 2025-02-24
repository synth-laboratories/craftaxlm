from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import *
from craftax.craftax.constants import BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax.craftax_state import EnvState
from craftax.craftax_classic.renderer import render_craftax_pixels

from craftaxlm.classic.metadata import (
    CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES,
    CRAFTAX_CLASSIC_HIGH_SALIENCE_MOBS,
    CRAFTAX_CLASSIC_HIGH_SALIENCE_OBJECTS,
    CRAFTAX_CLASSIC_LOW_SALIENCE_MOBS,
    CRAFTAX_CLASSIC_LOW_SALIENCE_OBJECTS,
)
from craftaxlm.shared import CraftaxState, mob_id_to_name

player_chars = {1: "←", 2: "→", 3: "↑", 4: "↓"}


@dataclass
class CraftaxClassicState(CraftaxState):
    map_full: List[Dict]
    map_compact: str
    inventory: Dict
    player: Dict
    environment: Dict
    image: np.ndarray
    recorder: Optional["CraftaxRecorder"] = None

    @classmethod
    def create(cls, *args, enable_recording=False, **kwargs):
        recorder = CraftaxRecorder() if enable_recording else None
        return cls(*args, **kwargs, recorder=recorder)

    def record_if_enabled(self, state: EnvState):
        if self.recorder is not None:
            self.recorder.record_frame(state)

    def save_recording(self, filename="episode.mp4", fps=30):
        if self.recorder is not None:
            self.recorder.save_video(filename, fps)

    def render_map_to_text(self, ignore_distant_low_salience=True):
        unique_blocks = list(
            set([tile["block"] for tile in self.map_full if "block" in tile])
        )
        if not set(unique_blocks).issubset(
            set(
                CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES
                + CRAFTAX_CLASSIC_LOW_SALIENCE_OBJECTS
                + CRAFTAX_CLASSIC_HIGH_SALIENCE_OBJECTS
            )
        ):
            raise ValueError(
                f"Unknown block types: {set(unique_blocks) - set(CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES + CRAFTAX_CLASSIC_LOW_SALIENCE_OBJECTS + CRAFTAX_CLASSIC_HIGH_SALIENCE_OBJECTS)}"
            )
        unique_mobs = list(
            set([tile["mob"] for tile in self.map_full if "mob" in tile])
        )
        if not set(unique_mobs).issubset(
            set(CRAFTAX_CLASSIC_LOW_SALIENCE_MOBS + CRAFTAX_CLASSIC_HIGH_SALIENCE_MOBS)
        ):
            raise ValueError(
                f"Unknown mob types: {set(unique_mobs) - set(CRAFTAX_CLASSIC_LOW_SALIENCE_MOBS + CRAFTAX_CLASSIC_HIGH_SALIENCE_MOBS)}"
            )

        def count_nearby_blocks(center_x, center_y, radius):
            block_counts = {}
            for tile in self.map_full:
                dx = abs(tile["position"]["x"] - center_x)
                dy = abs(tile["position"]["y"] - center_y)
                if dx <= radius and dy <= radius:
                    if not "block" in tile:
                        continue
                    block_type = tile["block"]
                    if block_type in CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES:
                        block_counts[block_type] = block_counts.get(block_type, 0) + 1
            return block_counts

        center_x, center_y = 0, 0
        nearby_blocks = count_nearby_blocks(center_x, center_y, 2)
        backdrop = (
            max(nearby_blocks, key=nearby_blocks.get) if nearby_blocks else "unknown"
        )

        low_salience_objects = CRAFTAX_CLASSIC_LOW_SALIENCE_OBJECTS + [
            object
            for object in CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES
            if object != backdrop
        ]

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
                for tile in self.map_full:
                    if tile["position"]["x"] == x and tile["position"]["y"] == y:
                        if tile["block"] in low_salience_objects:
                            periphery.append(
                                tile["block"].capitalize() + " is " + describe_xy(x, y)
                            )
                            if ignore_distant_low_salience:
                                found = True
                        if (
                            "mob" in tile
                            and tile["mob"] in CRAFTAX_CLASSIC_LOW_SALIENCE_MOBS
                        ):
                            periphery.append(
                                "A " + tile["mob"] + " is " + describe_xy(x, y)
                            )
                            if ignore_distant_low_salience:
                                found = True

        high_salience = []
        for tile in self.map_full:
            if (
                tile["visible"]
                and tile["block"] in CRAFTAX_CLASSIC_HIGH_SALIENCE_OBJECTS
            ):
                high_salience.append(
                    tile["block"].capitalize()
                    + " is "
                    + describe_xy(tile["position"]["x"], tile["position"]["y"]),
                )
            elif (
                tile["visible"]
                and "mob" in tile
                and tile["mob"] in CRAFTAX_CLASSIC_HIGH_SALIENCE_MOBS
            ):
                high_salience.append(
                    "A "
                    + tile["mob"]
                    + " is "
                    + describe_xy(tile["position"]["x"], tile["position"]["y"])
                )

        # print("Surrounding objects: ", periphery + high_salience)

        direction = self.player["direction_facing"]
        direction_to_offset = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        # Get the offset for the direction we're facing
        dx, dy = direction_to_offset.get(direction, (0, 0))

        # Look for objects exactly one step in the direction we're facing
        facing_objects = []
        for tile in self.map_full:
            if tile["position"]["x"] == dx and tile["position"]["y"] == dy:
                if (
                    "block" in tile
                    and tile["block"] not in CRAFTAX_CLASSIC_BACKDROP_BLOCK_TYPES
                ):
                    facing_objects.append(tile["block"].capitalize())
                if "mob" in tile:
                    facing_objects.append(f"A {tile['mob']}")

        # If no match found, we treat it as "No object"
        if not facing_objects:
            facing_objects = ["No object directly in front of you"]

        #print("Facing objects: ", facing_objects)
        return {
            "terrain_underneath_you": backdrop,
            "surroundings": periphery + high_salience,
            "object_you_are_facing": facing_objects[0],
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


# def render_craftax_classic_text_custom(
#     state: EnvState, enable_recording=False
# ) -> CraftaxClassicState:
#     map_data = []
#     for x in range(state.map.shape[0]):
#         for y in range(state.map.shape[1]):
#             if (
#                 not max(
#                     abs(x - state.player_position[0]), abs(y - state.player_position[1])
#                 )
#                 <= 4
#             ):
#                 continue
#             tile = {
#                 "position": {
#                     "x": x - state.player_position[0],
#                     "y": y - state.player_position[1],
#                 },
#                 "visible": max(
#                     abs(x - state.player_position[0]), abs(y - state.player_position[1])
#                 )
#                 <= 4,
#                 "block": BlockType(state.map[x, y]).name.lower(),
#             }
#             if state.mob_map[x, y].max() > 0.5:
#                 tile["mob"] = mob_id_to_name(state.mob_map[x, y].argmax())
#             map_data.append(tile)

#     inventory_data = {
#         "resources": {
#             "wood": int(state.inventory.wood),
#             "stone": int(state.inventory.stone),
#             "coal": int(state.inventory.coal),
#             "iron": int(state.inventory.iron),
#             "diamond": int(state.inventory.diamond),
#             "sapling": int(state.inventory.sapling),
#         },
#         "tools": {},
#     }

#     inventory_data["tools"]["pickaxe"] = {
#         "wood": int(state.inventory.wood_pickaxe),
#         "stone": int(state.inventory.stone_pickaxe),
#         "iron": int(state.inventory.iron_pickaxe),
#     }

#     inventory_data["tools"]["sword"] = {
#         "wood": int(state.inventory.wood_sword),
#         "stone": int(state.inventory.stone_sword),
#         "iron": int(state.inventory.iron_sword),
#     }

#     player_data = {
#         "health": int(state.player_health),
#         "food": int(state.player_food),
#         "drink": int(state.player_drink),
#         "energy": int(state.player_energy),
#         "direction_facing": Action(state.player_direction).name.lower(),
#     }

#     environment_data = {
#         "light_level": float(state.light_level),
#         "is_sleeping": bool(state.is_sleeping),
#         "floor": 0,  # Hardcode to 0 since player_level isn't in EnvState
#     }

#     def to_json_friendly(data):
#         if isinstance(data, dict):
#             return {key: to_json_friendly(value) for key, value in data.items()}
#         elif isinstance(data, (list, tuple)):
#             return [to_json_friendly(item) for item in data]
#         elif isinstance(data, (jnp.ndarray, np.ndarray)):
#             if data.ndim == 0:
#                 return data.item()
#             return [to_json_friendly(item) for item in data]
#         elif isinstance(data, (jnp.bool_, np.bool_)):
#             return bool(data)
#         elif isinstance(data, (jnp.integer, np.integer)):
#             return int(data)
#         elif isinstance(data, (jnp.floating, np.floating)):
#             return float(data)
#         else:
#             return data

#     craftax_state = CraftaxClassicState.create(
#         map=to_json_friendly(map_data),
#         inventory=to_json_friendly(inventory_data),
#         player=to_json_friendly(player_data),
#         environment=to_json_friendly(environment_data),
#         enable_recording=enable_recording,
#     )

#     # Record frame if recording is enabled
#     craftax_state.record_if_enabled(state)

#     return craftax_state


def render_craftax_text(state):
    """Renders the CraftAX Classic state as text representation."""
    obs_dim_array = jnp.array([OBS_DIM[0], OBS_DIM[1]], dtype=jnp.int32)

    # Map
    padded_grid = jnp.pad(
        state.map,
        (MAX_OBS_DIM + 2, MAX_OBS_DIM + 2),
        constant_values=BlockType.OUT_OF_BOUNDS.value,
    )

    tl_corner = state.player_position - obs_dim_array // 2 + MAX_OBS_DIM + 2
    map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, OBS_DIM)

    # Convert blocks to text representation
    block_chars = {
        BlockType.INVALID.value: ".",
        BlockType.OUT_OF_BOUNDS.value: "#",
        BlockType.GRASS.value: "G",
        BlockType.WATER.value: "~",
        BlockType.STONE.value: "S",
        BlockType.TREE.value: "T",
        BlockType.WOOD.value: "W",
        BlockType.PATH.value: "P",
        BlockType.COAL.value: "C",
        BlockType.IRON.value: "I",
        BlockType.DIAMOND.value: "D",
        BlockType.CRAFTING_TABLE.value: "B",  # B for crafting Bench
        BlockType.FURNACE.value: "F",
        BlockType.SAND.value: "_",
        BlockType.LAVA.value: "L",
        BlockType.PLANT.value: "p",
        BlockType.RIPE_PLANT.value: "r",
        BlockType.WALL.value: "X",
        BlockType.DARKNESS.value: " ",
        BlockType.WALL_MOSS.value: "M",
        BlockType.STALAGMITE.value: "^",
        BlockType.SAPPHIRE.value: "$",
        BlockType.RUBY.value: "R",
        BlockType.CHEST.value: "H",
        BlockType.FOUNTAIN.value: "O",
        BlockType.FIRE_GRASS.value: "f",
        BlockType.ICE_GRASS.value: "i",
        BlockType.GRAVEL.value: "g",
        BlockType.FIRE_TREE.value: "t",
        BlockType.ICE_SHRUB.value: "s",
        BlockType.ENCHANTMENT_TABLE_FIRE.value: "E",
        BlockType.ENCHANTMENT_TABLE_ICE.value: "e",
        BlockType.NECROMANCER.value: "N",
        BlockType.GRAVE.value: "v",
        BlockType.GRAVE2.value: "V",
        BlockType.GRAVE3.value: "u",
        BlockType.NECROMANCER_VULNERABLE.value: "n",
    }

    # Create text grid - swap x and y to match game coordinates
    height, width = map_view.shape
    text_grid = []
    for y in range(height):
        row = []
        for x in range(width):
            block_value = int(map_view[y, x])
            row.append(block_chars.get(block_value, "?"))
        text_grid.append(row)

    # Add player at center
    center_y, center_x = obs_dim_array // 2

    # print(state.player_direction)
    # print("Player chars:")
    # print(player_chars)
    player_direction = int(state.player_direction)
    text_grid[center_y][center_x] = player_chars.get(player_direction, "P")
    # print("Final player char:")
    # print(text_grid[center_y][center_x])

    # Add mobs - ensure coordinates are properly aligned
    def add_mob_to_grid(mob_positions, mob_masks, symbol):
        for pos, mask in zip(mob_positions, mob_masks):
            if not mask:
                continue
            # Calculate local position relative to player
            local_pos = pos - state.player_position + obs_dim_array // 2
            # Check if mob is within visible area
            if (local_pos >= 0).all() and (local_pos < obs_dim_array).all():
                y, x = int(local_pos[0]), int(local_pos[1])
                text_grid[y][x] = symbol

    add_mob_to_grid(state.zombies.position, state.zombies.mask, "Z")
    add_mob_to_grid(state.cows.position, state.cows.mask, "c")
    add_mob_to_grid(state.skeletons.position, state.skeletons.mask, "K")
    add_mob_to_grid(state.arrows.position, state.arrows.mask, "a")

    # Convert to string - join rows in correct order
    text_map = "\n".join("".join(row) for row in text_grid)

    return text_map


def render_craftax_classic_text_custom(
    state: EnvState, enable_recording=False
) -> CraftaxClassicState:
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
    text_map = render_craftax_text(state)

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
        "floor": 0,  # Hardcode to 0 since player_level isn't in EnvState
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

    pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_HUMAN)
    frame = np.array(pixels).astype(np.uint8)
    craftax_state = CraftaxClassicState(
        map_full=to_json_friendly(map_data),
        map_compact=text_map,
        inventory=to_json_friendly(inventory_data),
        player=to_json_friendly(player_data),
        environment=to_json_friendly(environment_data),
        image=frame,
        recorder=None if not enable_recording else CraftaxRecorder(),
    )

    # Record frame if recording is enabled
    if enable_recording:
        craftax_state.recorder.record_frame(state)

    return craftax_state


class CraftaxRecorder:
    def __init__(self, save_dir="recordings"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []

    def record_frame(self, state):
        pixels = render_craftax_pixels(state, BLOCK_PIXEL_SIZE_HUMAN)
        # Convert from JAX array to numpy and ensure uint8 format
        frame = np.array(pixels).astype(np.uint8)
        self.frames.append(frame)
        # print("Added frame - total frames:", len(self.frames))

    def save_video(self, sfilename, fps=1):
        save_path = self.save_dir / filename
        imageio.mimsave(str(save_path), self.frames, fps=fps, codec="mpeg4")
        # print(f"Saved video to {save_path} - {len(self.frames)} frames")
        # Clear frames after saving
        self.frames = []
