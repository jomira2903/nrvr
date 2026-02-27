import json
from dataclasses import dataclass, asdict
from typing import List, Optional

# Types d'éléments supportés
ELEMENT_TYPES = {
    "SKY_CLEAR", "SKY_STORM",
    "GROUND_GRASS", "GROUND_SNOW",
    "TREE_OAK", "TREE_PINE",
    "CLOUD_CUMULUS", "TERRAIN_MOUNT",
    "STRUCT_HOUSE", "STRUCT_CASTLE",
    "LIGHT_POINT", "CHAR_HERO", "CHAR_ENEMY"
}

@dataclass
class SceneEntity:
    entity_type: str   # ex: TREE_OAK
    x: float           # position 0.0 à 1.0
    y: float           # position 0.0 à 1.0
    scale: float = 1.0
    wind: float = 0.0
    state: str = "DEFAULT"

@dataclass
class SceneGSS:
    width: int
    height: int
    entities: List[SceneEntity]
    time_of_day: float = 0.5   # 0.0=nuit, 1.0=midi
    weather: str = "CLEAR"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def size_bytes(self) -> int:
        return len(self.to_json().encode('utf-8'))

    @staticmethod
    def from_json(data: str) -> 'SceneGSS':
        d = json.loads(data)
        d['entities'] = [SceneEntity(**e) for e in d['entities']]
        return SceneGSS(**d)


# Test rapide
if __name__ == "__main__":
    scene = SceneGSS(
        width=512,
        height=512,
        entities=[
            SceneEntity("SKY_CLEAR", 0.5, 0.2),
            SceneEntity("GROUND_GRASS", 0.5, 0.8),
            SceneEntity("TREE_OAK", 0.3, 0.6, scale=1.2),
            SceneEntity("TREE_PINE", 0.7, 0.65, scale=0.9),
            SceneEntity("CLOUD_CUMULUS", 0.5, 0.15, scale=1.1),
        ],
        time_of_day=0.7,
        weather="CLEAR"
    )

    print("=== SCÈNE GSS ===")
    print(scene.to_json())
    print(f"\nTaille GSS : {scene.size_bytes()} octets")
    print(f"Nombre d'entités : {len(scene.entities)}")