import xml.etree.ElementTree as ET
from pathlib import Path

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


def assign_domain(domain):
    if domain == 'A':
        return DOMAIN_A
    elif domain == 'B':
        return DOMAIN_B
    elif domain == 'C':
        return DOMAIN_C
    elif domain == 'D':
        return DOMAIN_D
    else:
        assert False, "No domain defined"


def change_to_domain(xml_str, domain):
    
    overrides = assign_domain(domain)
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