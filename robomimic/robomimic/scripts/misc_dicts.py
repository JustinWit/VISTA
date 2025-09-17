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

# DOMAIN_C = {
#     'texplane': 'black_texture.png', 
#     'tex-ceramic': 'kitchen_tabletop.png', 
#     'tex-light-gray-plaster': 'black_texture.png', 
#     'tex-steel-brushed' : 'black_texture.png', 
#     'tex-light-wood' : 'kitchen_tabletop.png', 
#     'tex-dark-wood' : 'sink_top.png', 
# }

DOMAIN_C = {
    'texplane': 'black_texture.png', 
    'tex-ceramic': 'gray-felt.png', 
    'tex-light-gray-plaster': 'black_texture.png', 
    'tex-steel-brushed' : 'black_texture.png', 
    'tex-light-wood' : 'gray-felt.png', 
    'tex-dark-wood' : 'dirt.png', 
}

DOMAIN_D = {
    'texplane': 'pink-plaster.png', 
    'tex-ceramic': 'steel-brushed.png', 
    'tex-light-gray-plaster': 'blue_wallpaper.png', 
    'tex-light-wood' : 'steel-brushed.png', 
    'tex-dark-wood' : 'wood-varnished-panels.png', 
}

# DOMAIN_D = {
#     'texplane': 'pink-plaster.png', 
#     'tex-ceramic': 'woodentabletop.png', 
#     'tex-light-gray-plaster': 'blue_wallpaper.png', 
#     'tex-light-wood' : 'woodentabletop.png', 
#     'tex-dark-wood' : 'sink_top.png', 
# }


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
    47 : 116,  # table legs
}

SQUARE_IDS = {
    0 : 110, 
    1 : 111, 
    4 : 112,
    2 : 310,  # square peg
    25 : 311,  # square key
    45 : 114,  # robot base
    47 : 116,  # table legs
}

TPA_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 112, 
    23 : 410, 
    24 : 411, 
    25 : 412, 
    45 : 114, 
    47 : 116,  # table legs
}

MUG_IDS = {
    0 : 110,
    1 : 111,
    2 : 112,
    23 : 510,  # tool box ID 
    26 : 511,  # mug
    45 : 114, 
    47 : 116,  # table legs
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
    47 : 116,  # table legs
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
    47 : 116,  # table legs
}

COFFEE_PREP_IDS = {
    0 : 110,
    1 : 111, 
    2 : 112, 
    23 : 711, 
    24 : 710, 
    32 : 510,  # toolbox
    35 : 511,  # mug 
    45 : 114, 
    47 : 116,  # table legs
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
    47 : 116,  # table legs
}

PICK_PLACE_CAN_IDS = {
    0 : 110, 
    1 : 810, 
    2 : 811, 
    3 : 112, 
    4 : 811, 
    24 : 812, 
    25 : 813, 
    26 : 814, 
    27 : 815,
    31 : 819, 
    45 : 114, 
    47 : 116,  # table legs
}

COFFEE_D0_IDS = {
    0 : 110,
    1 : 111, 
    2 : 112, 
    23 : 711, 
    24 : 710, 
    45 : 114, 
    47 : 116, 
}

HAMMER_IDS = {
    0 : 110,
    1 : 111, 
    5 : 112, 
    26 : 910, 
    49 : 510, 
    45 : 114, 
    47 : 116, 

}

STACK_SIMPLE_IDS = {
    0 : 110,
    1 : 111, 
    2 : 112,
    23 : 210, 
    24 : 211, 
    45 : 114, 
    47 : 116, 
}

NUT_SQUARE_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 310,
    3 : 312, 
    4 : 112, 
    25 : 311, 
    45 : 114, 
    47 : 116, 
}

THREADING_IDS = {
    0 : 110, 
    1 : 111, 
    2 : 112, 
    23 : 911, 
    24 : 912, 
    45 : 114, 
    47 : 116, 
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
    'PickPlaceCan' : PICK_PLACE_CAN_IDS, 
    'Coffee_D0' : COFFEE_D0_IDS, 
    'HammerCleanup_D0' : HAMMER_IDS, 
    'Stack_D0' : STACK_SIMPLE_IDS, 
    'NutAssemblySquare' : NUT_SQUARE_IDS, 
    'Threading_D0' : THREADING_IDS, 
}


STACK3_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

SQUARE_SEG_KEYS = {
    86 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

TPA_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

MUG_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

NUT_ASSEMBLY_SEG_KEYS = {
    99 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

KITCHEN_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

COFFEE_PREP_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

PICK_PLACE_SEG_KEYS = {
    121 : 45,  # robot base
    17 : 47, 
    18 : 47, 
    19 : 47, 
    20 : 47, 
    35 : 47, 
    36 : 47, 
    37 : 47, 
    38 : 47, 
}

PICK_PLACE_CAN_SEG_KEYS = {
    108 : 45,  # robot base
    17 : 47, 
    18 : 47, 
    19 : 47, 
    20 : 47, 
    35 : 47, 
    36 : 47, 
    37 : 47, 
    38 : 47, 
}

COFFEE_SEG_KEYS = {
    82 : 45,  # robot base
    9 : 47, 
    10 : 47, 
    11 : 47, 
    12 : 47, 

}

HAMMER_SEG_KEYS = {
    9 : 47, 
    10 : 47, 
    11 : 47, 
    12 : 47, 
    20 : 49, 
    21 : 49, 
    22 : 49, 
    23 : 49, 
    24 : 49, 
    25 : 49, 
    26 : 49, 
    35 : 49, 
    36 : 49, 
    37 : 49, 
    38 : 49, 
    39 : 49, 
    40 : 49, 
    41 : 49, 
    42 : 49, 
    112 : 45, 
}


STACK_SIMPLE_SEG_KEYS = {
    82 : 45, 
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

NUS_SQUARE_SEG_KEYS = {
    86 : 45,
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
}

THREADING_SEG_KEYS = {
    82 : 45,
    9 : 47,  # table legs
    10 : 47, 
    11 : 47, 
    12 : 47, 
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
    'PickPlaceCan' : PICK_PLACE_CAN_SEG_KEYS, 
    'Coffee_D0' : COFFEE_SEG_KEYS, 
    'Stack_D0' : STACK_SIMPLE_SEG_KEYS, 
    'NutAssemblySquare' : NUS_SQUARE_SEG_KEYS, 
    'Threading_D0' : THREADING_SEG_KEYS, 
}