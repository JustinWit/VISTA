import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.stats import mode
import cv2

import torch
import torch.nn.functional as F

######################
### Custom Domains ###

DOMAIN_A = {
    'texplane': 'light-gray-floor-tile.png', 
    'tex-ceramic': 'ceramic.png', 
    # 'tex-steel-brushed': 'steel-brushed.png', 
    'tex-light-gray-plaster': 'light-gray-plaster.png', 
}

DOMAIN_B = {
    'texplane': 'cereal.png', 
    'tex-ceramic': 'clay.png', 
    # 'tex-steel-brushed': 'bread.png', 
    'tex-light-gray-plaster': 'lemon.png', 
}

DOMAIN_C = {
    'texplane': 'wood-tiles.png', 
    'tex-ceramic': 'dark-wood.png', 
    # 'tex-steel-brushed': 'bread.png', 
    'tex-light-gray-plaster': 'yellow-plaster.png', 
}

DOMAIN_D = {
    'texplane': 'brass-ambra.png', 
    'tex-ceramic': 'pink-plaster.png', 
    # 'tex-steel-brushed': 'bread.png', 
    'tex-light-gray-plaster': 'metal.png', 
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
}


def change_to_domain(xml_str, domain):
    overrides = DOMAIN_MAP[domain]
    # change texture path as defined by domain
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    textures = asset.findall("texture")
    
    for i in textures:
        if i.get('name') in overrides.keys():
            f = Path(i.get('file'))
            i.set('file', str(f.with_name(overrides[i.get('name')])))

    return ET.tostring(root, encoding="utf8").decode("utf8")


def mode_filter(img: torch.Tensor, k: int = 3, padding: str = "reflect",
                    num_classes: int = 256) -> torch.Tensor:
    """
    Vectorized mode filter for 1-channel images with small integer ranges.

    Args:
        img: (N, H, W, 1) or (1, H, W, 1). Values should be integer-coded in [0, num_classes-1].
             uint8 or long is fine (will be cast to long internally for one_hot).
        k:   kernel size (odd).
        padding: pad mode for F.pad ('reflect', 'replicate', 'constant', etc.).
        num_classes: size of the discrete value range (256 for uint8).

    Returns:
        Tensor of shape (N, 1, H, W) with per-window mode values (same dtype as input).
    """
    
    img = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous()
    assert img.dim() == 4 and img.size(1) == 1, "img must be (N,1,H,W)"
    assert k % 2 == 1, "kernel size k must be odd"

    N, _, H, W = img.shape
    pad = k // 2

    # pad and unfold to sliding windows
    x = F.pad(img.float(), (pad, pad, pad, pad), mode=padding)
    # (N, C*k*k, L) where C=1 and L = H*W after unfold
    patches = F.unfold(x, kernel_size=k).long()  # (N, k*k, H*W)

    # scatter add to get modes
    one_hot_counts = torch.zeros(N, H*W, num_classes, device=img.device, dtype=torch.int32)
    indices = patches.transpose(1, 2)  # (N, H*W, k*k)
    one_hot_counts.scatter_add_(2, indices.to(torch.int64), torch.ones_like(indices, dtype=torch.int32))
    modes = one_hot_counts.argmax(dim=-1)  # (N, H*W)

    # reshape back to image grid in original shape
    modes = modes.view(N, 1, H, W).permute(0, 2, 3, 1)

    # cast back to original type
    modes = np.array(modes, dtype=np.uint8)

    return modes


def assign_groups(env, obs, camera_names):
    geom_seg = dict([(i, obs[f'{i}_segmentation_element']) for i in camera_names])
    mode_seg = {}
    final_seg = dict([(i, -1 * np.ones_like(geom_seg[i])) for i in camera_names])

    # loop over geom
    for geom_id in range(env.env.sim.model.ngeom):
        body_id = env.env.sim.model.geom_bodyid[geom_id]
        root_id = env.env.sim.model.body_rootid[body_id]
        # special case for hammer task, need to map body_id 3 & 4 to 2 instead of 1
        if (body_id in [3, 4]) and env.name == 'HammerCleanup_D0':
            root_id = 2
        rootbody_name = env.env.sim.model.body_id2name(body_id)

        for cam in camera_names:
            final_seg[cam][geom_seg[cam] == geom_id] = root_id

    for x in final_seg:
        # use mode filter to drop stray pixels
        final_seg[x] = mode_filter(final_seg[x], num_classes=env.env.sim.model.ngeom)

        # manual map small objects back to correct group
        for i in TASK_OVERRIDES[env.name]:
            body_id = env.env.sim.model.geom_bodyid[i]
            root_id = env.env.sim.model.body_rootid[body_id]

            final_seg[x][geom_seg[x] == i] = root_id

    assert not np.any(final_seg[camera_names[0]] == -1), "Missed something"
    return final_seg