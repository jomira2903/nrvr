import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ENTITY_TYPES = [
    "SKY_CLEAR", "SKY_STORM",
    "GROUND_GRASS", "GROUND_SNOW",
    "TREE_OAK", "TREE_PINE",
    "CLOUD_CUMULUS", "TERRAIN_MOUNT",
    "STRUCT_HOUSE", "STRUCT_CASTLE",
    "LIGHT_POINT", "CHAR_HERO", "CHAR_ENEMY"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
MAX_ENTITIES = 20

def gss_to_tensor(scene) -> torch.Tensor:
    data = []
    for entity in scene.entities[:MAX_ENTITIES]:
        type_idx = TYPE_TO_IDX.get(entity.entity_type, 0)
        type_norm = type_idx / len(ENTITY_TYPES)
        data.extend([type_norm, entity.x, entity.y, entity.scale, entity.wind])

    while len(data) < MAX_ENTITIES * 5:
        data.append(0.0)

    data.append(scene.time_of_day)
    data.append(1.0 if scene.weather == "CLEAR" else 0.0)

    return torch.tensor(data, dtype=torch.float32)


class NeuralRenderer(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size
        input_size = MAX_ENTITIES * 5 + 2

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, image_size * image_size * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        image = self.decoder(features)
        return image.view(-1, 3, self.image_size, self.image_size)


if __name__ == "__main__":
    from gss.scene import SceneGSS, SceneEntity

    scene = SceneGSS(
        width=64, height=64,
        entities=[
            SceneEntity("SKY_CLEAR", 0.5, 0.2),
            SceneEntity("GROUND_GRASS", 0.5, 0.8),
            SceneEntity("TREE_OAK", 0.3, 0.6, scale=1.2),
        ],
        time_of_day=0.7,
        weather="CLEAR"
    )

    model = NeuralRenderer(image_size=64)
    tensor = gss_to_tensor(scene).unsqueeze(0)
    output = model(tensor)

    print(f"Entrée  : {tensor.shape}")
    print(f"Sortie  : {output.shape}")
    print(f"Min/Max : {output.min():.3f} / {output.max():.3f}")
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    print("MODELE OK !")

    class NeuralRendererCNN(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size

        self.encoder = nn.Sequential(
            nn.Linear(102, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 4*4*256),
            nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(-1, 256, 4, 4)
        image = self.decoder(features)
        return image