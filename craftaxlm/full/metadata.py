CRAFTAX_FULL_ACTION_MAPPING = {
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
CRAFTAX_FULL_ACHIEVEMENTS = {
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
    22: "Make Arrow",
    23: "Make Torch",
    24: "Place Torch",
    25: "Make Diamond Sword",
    26: "Make Iron Armour",
    27: "Make Diamond Armour",
    28: "Enter Gnomish Mines",
    29: "Enter Dungeon",
    30: "Enter Sewers",
    31: "Enter Vault",
    32: "Enter Troll Mines",
    33: "Enter Fire Realm",
    34: "Enter Ice Realm",
    35: "Enter Graveyard",
    36: "Defeat Gnome Warrior",
    37: "Defeat Gnome Archer",
    38: "Defeat Orc Solider",
    39: "Defeat Orc Mage",
    40: "Defeat Lizard",
    41: "Defeat Kobold",
    42: "Defeat Troll",
    43: "Defeat Deep Thing",
    44: "Defeat Pigman",
    45: "Defeat Fire Elemental",
    46: "Defeat Frost Troll",
    47: "Defeat Ice Elemental",
    48: "Damage Necromancer",
    49: "Defeat Necromancer",
    50: "Eat Bat",
    51: "Eat Snail",
    52: "Find Bow",
    53: "Fire Bow",
    54: "Collect Sapphire",
    55: "Learn Fireball",
    56: "Cast Fireball",
    57: "Learn Iceball",
    58: "Cast Iceball",
    59: "Collect Ruby",
    60: "Make Diamond Pickaxe",
    61: "Open Chest",
    62: "Drink Potion",
    63: "Enchant Sword",
    64: "Enchant Armour",
    65: "Defeat Knight",
    66: "Defeat Archer",
}

CRAFTAX_FULL_BACKDROP_BLOCK_TYPES = [
    "grass",
    "sand",
    "path",
    "fire_grass",
    "ice_grass",
    "gravel",
]
CRAFTAX_FULL_LOW_SALIENCE_OBJECTS = [
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
CRAFTAX_FULL_LOW_SALIENCE_MOBS = []
CRAFTAX_FULL_LOW_SALIENCE_ITEMS = []
CRAFTAX_FULL_HIGH_SALIENCE_OBJECTS = [
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
CRAFTAX_FULL_HIGH_SALIENCE_ITEMS = ["ladder_down", "ladder_up"]
CRAFTAX_FULL_HIGH_SALIENCE_MOBS = [
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

FULL_TUTORIAL = """
# Craftax Wiki: Tutorial

This page explains how to beat the game!

## Basic Mechanics
Craftax is a game about exploring dungeons, mining, crafting and fighting enemies.
The player can move in the four cardinal directions using WASD and can interact using SPACE.
Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest.

The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy).
Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively.
Mana is used for casting spells or enchanting items and will naturally recover.
Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0.
If the players health falls beneath 0 they will die and the game will restart.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level.
Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level.
The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld).
There are 9 levels in total.

## Floor 0: Overworld

The overworld consists of grasslands, lakes and mountains.
The player should begin by mining trees, before making a crafting table and using it to craft wooden tools by pressing the appropriate crafting keys while adjacent to the placed table.
You can then use the wooden pickaxe to mine stone and craft stone tools.
With stone tools, the player can mine coal and iron.
By placing a furnace next to the crafting table and standing adjacent to both (diagonal counts) the player can craft iron tools using wood, iron and coal.
Keep up your food by eating cows, make sure to drink water from the lakes and sleep when tired.
Bear in mind that the player is very vulnerable when sleeping so you should block yourself in by placing stone around you to sleep.

Now the player should find the first ladder down to the dungeon.  Make sure to collect lots of wood from the overworld as it is rarer in the dungeons.
The player might also want to collect seeds by pressing space on grass and plant these.  If the player returns to the overworld they will then have some easy food.

## Floor 1: Dungeon

Stand on the downward ladder and press the DESCEND key to climb down to the dungeon.
Note that you will appear on an upwards ladder, which can be climbed with the ASCEND key to return to the overworld.

The dungeon consists of a set of rooms joined by paths.  Rooms can contain fountains to drink from and chests with random items.
The first chest opened will contain a bow.  Arrows can be crafted at a crafting table using wood and stone.
The dungeon is inhabited by orc warriors and mages.  Once 8 of these have been killed the ladder to the next floor will open up.
The floor also contains snails which can be eaten.

If the player finds themselves on low health, they should block themselves in and either sleep or (if already at maximum energy) rest.
Resting causes the player to execute no-op actions until an intrinsic decays to 0, the player is attacked or the player recovers to full health.

### Attributes
Upon descending to each floor for the first time the player will be awarded an experience point.
These can be assigned to your attributes by pressing the appropriate key.
Each attribute starts at level 1 and can be upgraded to a maximum of level 5, with assignment being permanent.

**Dexterity**: Dexterity increases your maximum food, water and energy reserves, as well as making them decay slower.  It also increase damage done with a bow.

**Strength**: Strength increases your melee damage and maximum health.

**Intelligence**:  Intelligence increases your maximum mana, reduces mana decay, increases damage from spells and increases effectiveness of enchantments.

### Potions
The player will find potions of six different colours in the chests.
Potions will provide either +8 or -3 of either health, mana or energy.
However, the effects of each colour of potions is permuted every time the game is run, so the player will need to perform trial and error to figure out the ordering each game.

## Floor 2: Gnomish Mines

The gnomish mines are the first dark level.  To see on this level the player will need to place torches, which can be crafted with wood and coal.
There are pools of water to drink from and bats to eat.
The edges of the caverns are rich in ores to mine and should be taken advantage of.
The gnomes are stronger than the orcs and care must be taken to avoid being surrounded in the open spaces.
As well as coal and iron the player might find diamonds, sapphires and rubies.
Diamonds can be used to craft a sword (requires 2) or a pickaxe (requires 3) which can be used to mine sapphires and rubies.
The player can also craft iron armour (3 iron and 3 coal per piece) or diamond armour (3 diamonds) to reduce damage.

## Floor 3: Sewers

The sewers are similar to the dungeons in layout.  There are patches of water which need to be filled by placing stone and then mining it.
The lizards that inhabit this level are very dangerous and can swim through the water, while the Kobolds through high-damage daggers.

### Spellcasting
The first opened chest on this level will include a book, which can be read to learn a random spell (either fireball or iceball).
These are ranged attacks that consume 2 mana to cast.
Until now the only possible damage type has been 'physical', however the spells do fire and ice damage respectively.
Creatures on later levels have high resistance to physical damage and require fire or ice damage to kill.

### Enchanting
The sewers will contain the ice enchantment table.  The player can enchant their sword, armour or bow by standing next to the enchantment table and consuming 9 mana as well as the appropriate gemstone.
Sapphires are used for ice enchantments and rubies for fire enchantments.
An enchantment on the sword or bow will cause the player to deal +50% damage of the respective type.
An enchantment on armour will reduce damage of that type by 20% for each armour piece.

## Floor 4: Vaults
The vaults are another dungeon level.  Another book will be found along with the fire enchantment table.
The knights and archers on this floor are armoured and physical damage is halved, so using enchantments or spells is recommended.

## Floor 5: Troll Mines
The troll mines are a dark cavern-like level similar to the Gnomish Mines.
This level is the richest with ores so the player should make use of these and ideally craft full diamond armour.
The trolls are strong enemies and do a lot of damage.
The deep things in the water are very weak if you can hit them but do a lot of damage with their ranged attacks.

## Floor 6: Fire Realm
The fire realm consists of a set of islands separated by lava.
There is no way to obtain water on this level so the player may have to periodically return to the troll mines to drink.
The pig men and fire elementals are entirely resistant to fire damage and resistant to almost all physical damage, so ice damage (either through spells or enchantments) will probably be required to kill them.
Note that the player can build bridges across the lava by placing and mining stone.
There are also lots of coal and rubies to be found on this level.

## Floor 7: Ice Realm
The ice realm is a dark level inhabited by frost trolls and ice elementals.
These are the strongest creatures in the game and require fire damage to kill them.
There is no food on this level so the player may have to return to the fire realm to eat.
There are also lots of sapphires and rubies in the mountains.

## Floor 8: Graveyard
The graveyard is home to the necromancer who is the final enemy in the game.
There is no ladder out of the boss level.
There is no food or water on the level but the players hunger, thirst and energy will not decay on this level.
The necromancer will summon waves of enemies from the graves.
Once the player has defeated a wave of enemies the necromancer will enter his 'vulnerable' state, at which point the player can attack him.
Doing so will trigger the next wave of enemies.
Each wave of enemies corresponds to the creatures from a particular floor of the game.
Once the final wave (the ice realm wave) has been defeated the player can attack the necromancer and win the game!"""
