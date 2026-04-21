from pathlib import Path
import io

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.resnet18_cat_dog import ResNet18CatDog


# Paths
BASE_DIR = Path(__file__).resolve().parent          # backend/app
BACKEND_DIR = BASE_DIR.parent                      # backend
MODEL_PATH = BACKEND_DIR / "models" / "resnet18_catdog.pth"


# App setup
app = FastAPI(
    title="Cat-Dog Classifier API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Labels
CLASS_NAMES = ["cat", "dog"]


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load model
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = ResNet18CatDog().to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


model = load_model()


# Routes
@app.get("/")
def root():
    return {
        "message": "Cat-Dog Classifier API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    try:
        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to read uploaded file."
        )

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return {
        "prediction": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4)
    }
