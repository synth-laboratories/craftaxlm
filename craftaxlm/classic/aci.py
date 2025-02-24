import math
from typing import Dict, List, Literal

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import *
from craftax.craftax.craftax_state import EnvState
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

from craftaxlm.classic.metadata import (
    CRAFTAX_CLASSIC_ACHIEVEMENTS,
    CRAFTAX_CLASSIC_ACTION_MAPPING,
)
from craftaxlm.classic.state import (
    CraftaxClassicState,
    CraftaxRecorder,
    mob_id_to_name,
    player_chars,
    render_craftax_classic_text_custom,
)
from craftaxlm.shared import CraftaxBaseACI


class CraftaxClassicACI(CraftaxBaseACI):
    formatting: Literal["md", "xml"] = "md"
    map_format: Literal["full", "compact"] = "full"

    def make_env(self):
        return make_craftax_env_from_name(
            "Craftax-Classic-Symbolic-v1", auto_reset=False
        )

    def create_starting_obs(self):
        step_info = self.create_step_info(self.state, 0.0, False)
        return {
            "state_image": step_info["state_image"],
            "state_text": step_info["state_text"],
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
        craftaxlm_state = render_craftax_classic_text_custom(state)
        return {
            "state_image": craftaxlm_state.image,
            "state_text": craftaxlm_state.render_to_text_simple(
                verbose=self.verbose,
                formatting=self.formatting,
                map_format=self.map_format,
            ),
            "reward": float(reward),
            "done": bool(done),
        }


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
    player_direction = int(state.player_direction)
    # print("Player direction: ", player_direction)
    text_grid[center_y][center_x] = player_chars.get(player_direction, "P")

    # print("Player char: ", text_grid[center_y][center_x])
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


compact_map_translation_full = {
    # Basic blocks
    "G": "Grass",
    "~": "Water",
    "S": "Stone",
    "T": "Tree",
    "W": "Wood",
    "P": "Path",
    "C": "Coal",
    "I": "Iron",
    "D": "Diamond",
    "B": "Crafting Bench",
    "F": "Furnace",
    "E": "Enchantment Table Fire",
    "e": "Enchantment Table Ice",
    # Additional blocks
    "_": "Sand",
    "L": "Lava",
    "p": "Plant",
    "r": "Ripe Plant",
    "X": "Wall",
    " ": "Darkness",
    "M": "Wall Moss",
    "^": "Stalagmite",
    "$": "Sapphire",
    "R": "Ruby",
    "H": "Chest",
    "O": "Fountain",
    # Special terrain
    "f": "Fire Grass",
    "i": "Ice Grass",
    "g": "Gravel",
    "t": "Fire Tree",
    "s": "Ice Shrub",
    # Necromancer and graves
    "N": "Necromancer",
    "n": "Necromancer Vulnerable",
    "v": "Grave",
    "V": "Grave2",
    "u": "Grave3",
    # Mobs
    "Z": "Zombie",
    "c": "Cow",
    "K": "Skeleton",
    "a": "Arrow",
    # Boundaries
    ".": "Invalid",
    "#": "Out of Bounds",
}

compact_map_translation_classic = {
    # Basic blocks
    "G": "Grass",
    "~": "Water",
    "S": "Stone",
    "T": "Tree",
    "W": "Wood",
    "P": "Path",
    "C": "Coal",
    "I": "Iron",
    "D": "Diamond",
    "B": "Crafting Bench",
    "F": "Furnace",
    "E": "Enchantment Table Fire",
    "e": "Enchantment Table Ice",
    # Additional blocks
    "_": "Sand",
    "L": "Lava",
    "p": "Plant",
    "r": "Ripe Plant",
    "X": "Wall",
    # Mobs
    "Z": "Zombie",
    "c": "Cow",
    "K": "Skeleton",
    "a": "Arrow",
    # Boundaries
    ".": "Invalid",
    "#": "Out of Bounds",
}


def render_craftax_classic_text_exp(
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
    # Convert from JAX array to numpy and ensure uint8 format
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


def canonical_crafter_score_classic(achievements_by_run: List[Dict]):
    assert all(
        isinstance(achievements, dict)
        and all(
            k in achievements
            for k in [
                "Collect Wood",
                "Place Table",
                "Eat Cow",
                "Collect Sapling",
                "Collect Drink",
                "Make Wood Pickaxe",
                "Make Wood Sword",
                "Place Plant",
                "Defeat Zombie",
                "Collect Stone",
                "Place Stone",
                "Eat Plant",
                "Defeat Skeleton",
                "Make Stone Pickaxe",
                "Make Stone Sword",
                "Wake Up",
                "Place Furnace",
                "Collect Coal",
                "Collect Iron",
                "Collect Diamond",
                "Make Iron Pickaxe",
                "Make Iron Sword",
            ]
        )
        and all(isinstance(v, int) for v in achievements.values())
        for achievements in achievements_by_run
    )
    successes_by_achievement = {
        achievement: sum(1 for run in achievements_by_run if run[achievement] > 0)
        for achievement in achievements_by_run[0].keys()
    }
    success_rates = [
        count / len(achievements_by_run) for count in successes_by_achievement.values()
    ]

    # Compute the Crafter score: S = exp((1/N)*sum(ln(1 + s_i))) - 1, where s_i is the success rate for achievement i.
    N = len(success_rates)
    return math.exp(sum(math.log(1 + s) for s in success_rates) / N) - 1


if __name__ == "__main__":
    craftax_aci = CraftaxClassicACI(formatting="xml", map_format="compact")
    craftax_aci.reset()
    normal_info = craftax_aci.create_step_info(craftax_aci.state, 0.0, False)
    print(normal_info)
    achievements_info = craftax_aci.get_achievements(craftax_aci.state)
    print(achievements_info)
