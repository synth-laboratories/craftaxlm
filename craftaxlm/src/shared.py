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
