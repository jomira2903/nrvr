from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import torch
import base64
import io
import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gss.scene import SceneGSS, SceneEntity
from model.network import NeuralRenderer, gss_to_tensor

app = FastAPI(title="NRVR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = NeuralRenderer(image_size=64)
model_path = os.path.join(BASE_DIR, "model_trained.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("Modèle entraîné chargé !")
else:
    print("Modèle aléatoire (pas de fichier .pth)")
model.eval()
print("Modèle chargé !")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EntityInput(BaseModel):
    entity_type: str
    x: float
    y: float
    scale: float = 1.0
    wind: float = 0.0
    state: str = "DEFAULT"

class SceneInput(BaseModel):
    width: int = 512
    height: int = 512
    entities: List[EntityInput]
    time_of_day: float = 0.5
    weather: str = "CLEAR"

@app.get("/")
def root():
    return {"status": "NRVR API active", "version": "0.1.0"}

@app.get("/test")
def test_page():
    return FileResponse(os.path.join(BASE_DIR, "test.html"))

@app.post("/render")
def render(scene_input: SceneInput):
    scene = SceneGSS(
        width=scene_input.width,
        height=scene_input.height,
        entities=[SceneEntity(**e.dict()) for e in scene_input.entities],
        time_of_day=scene_input.time_of_day,
        weather=scene_input.weather
    )

    with torch.no_grad():
        tensor = gss_to_tensor(scene).unsqueeze(0)
        output = model(tensor)

    img_array = output.squeeze(0).permute(1, 2, 0).numpy()
    img_array = (img_array * 255).astype('uint8')
    img = Image.fromarray(img_array, 'RGB')
    img = img.resize((scene_input.width, scene_input.height), Image.NEAREST)

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "image": img_b64,
        "gss_bytes": scene.size_bytes(),
        "entities": len(scene.entities),
        "inference_ms": 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)