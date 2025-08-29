######################
### Custom Domains ###
DOMAIN_A = {
    'texplane': 'light-gray-floor-tile.png', 
    'tex-ceramic': 'kitchen_tabletop.png', 
    'tex-light-gray-plaster': 'blue_wallpaper.png', 
    'tex-light-wood' : 'kitchen_tabletop.png', 
    'tex-dark-wood' : 'sink_top.png', 

}

DOMAIN_B = {
    'texplane': 'light-gray-floor-tile.png', 
    'tex-ceramic': 'ceramic.png', 
    'tex-light-gray-plaster': 'light-gray-plaster.png', 
}

DOMAIN_C = {
    'texplane': 'wood-varnished-panels.png', 
    'tex-ceramic': 'desk_tabletop.png', 
    'tex-light-gray-plaster': 'yellow-plaster.png', 
}

DOMAIN_D = {
    'texplane': 'pink-plaster.png', 
    'tex-ceramic': 'steel-scratched.png', 
    'tex-light-gray-plaster': 'pink-texture.png', 
}


DOMAIN_MAP = {
    'A' : DOMAIN_A, 
    'B' : DOMAIN_B,
    'C' : DOMAIN_C, 
    'D' : DOMAIN_D
}


TASK_OVERRIDES = {
    'HammerCleanup_D0' : [103, 106, 115, 117, 121, 123],  # left gripper, right gripper, hammer_handle, hammer_head, hammer_face, hammer_claw
    'PickPlaceCan' : [17, 18, 19, 20, 35, 36, 37, 38, 99, 102, 121],  # left table legs, right table legs, gripper, table objects
    'Coffee_D0' : [73, 76, 85],  #  left gripper, right gripper, kcup
    'Stack_D0' : [73, 76, 85, 87],  # left gripper, right gripper, small block, big block
    'NutAssemblySquare' : [14, 16, 77, 80, 93, 94, 95, 96, 97],  # round peg, square peg, left gripper, right gripper, square nut
    'Threading_D0' : [73, 76, 85, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137],  # left gripper, right gripper, thread end, thread base, 
    'StackThree_D1' : [85, 87, 89], 
    'MugCleanup_D1' : [111, 112, 113], 
    'PickPlace_D0' : [17, 18, 19, 20, 35, 36, 37, 38]
}


STACK3_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 112,  # franka
    23 : 210,  # red block
    24 : 211,  # green block
    25 : 212,  # blue block
    45 : 114,  # robot base
}

SQUARE_IDS = {
    0 : 110, 
    1 : 111, 
    4 : 112,
    2 : 310,  # square peg
    25 : 311,  # square key
    45 : 114,  # robot base
}

TPA_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 112, 
    23 : 410, 
    24 : 411, 
    25 : 412, 
    45 : 114, 
}

MUG_IDS = {
    0 : 110,
    1 : 111,
    2 : 112,
    23 : 510,  # tool box ID 
    26 : 511,  # mug
    45 : 114, 
}

NUT_ASSEMBLY_IDS = {
    0 : 110,
    1 : 111, 
    2 : 310,  # square peg
    3 : 312,  # round peg
    4 : 113,  # sawyer robot
    35 : 311,  # square key
    36 : 313,  # round key
    45 : 115,  # sawyer base
}

KITCHEN_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 112, 
    23 : 610, 
    24 : 611, 
    25 : 612, 
    28 : 613, 
    31 : 614, 
    45 : 114, 
}

COFFEE_PREP_IDS = {
    0 : 110,
    1 : 111, 
    2 : 112, 
    24 : 710, 
    32 : 510,  # toolbox
    35 : 511,  # mug 
    45 : 114, 
}

PICK_PLACE_IDS = {
    0 : 110, 
    1 : 810, 
    2 : 811, 
    3 : 113, 
    4 : 811, 
    34 : 812, 
    35 : 813, 
    36 : 814, 
    37 : 815,
    38 : 816, 
    39 : 817, 
    40 : 818, 
    41 : 819, 
    45 : 115, 
}

MASTER_LIST_IDS = {
    'StackThree_D1' : STACK3_IDS, 
    'Square_D2' : SQUARE_IDS, 
    'ThreePieceAssembly_D2' : TPA_IDS, 
    'MugCleanup_D1' : MUG_IDS,
    'NutAssembly_D0' : NUT_ASSEMBLY_IDS, 
    'Kitchen_D1' : KITCHEN_IDS, 
    'CoffeePreparation_D1' : COFFEE_PREP_IDS, 
    'PickPlace_D0' : PICK_PLACE_IDS, 
}


HAMMER_SEG_KEYS = {
    3 : 2, 
    4 : 2, 
}


STACK3_SEG_KEYS = {
    82 : 45, 
}

SQUARE_SEG_KEYS = {
    86 : 45, 
}

TPA_SEG_KEYS = {
    82 : 45, 
}

MUG_SEG_KEYS = {
    82 : 45, 
}

NUT_ASSEMBLY_SEG_KEYS = {
    99 : 45, 
}

KITCHEN_SEG_KEYS = {
    82 : 45, 
}

COFFEE_PREP_SEG_KEYS = {
    82 : 45, 
}

PICK_PLACE_SEG_KEYS = {
    121 : 45, 
}


GEOM_MAPS = {
    'HammerCleanup_D0' : HAMMER_SEG_KEYS, 
    'StackThree_D1' : STACK3_SEG_KEYS, 
    'Square_D2' : SQUARE_SEG_KEYS, 
    'ThreePieceAssembly_D2' : TPA_SEG_KEYS, 
    'MugCleanup_D1' : MUG_SEG_KEYS, 
    'NutAssembly_D0' : NUT_ASSEMBLY_SEG_KEYS, 
    'Kitchen_D1' : KITCHEN_SEG_KEYS, 
    'CoffeePreparation_D1' : COFFEE_PREP_SEG_KEYS, 
    'PickPlace_D0' : PICK_PLACE_SEG_KEYS, 
}