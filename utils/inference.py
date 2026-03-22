import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("models/best_model.pth")
META_PATH = Path("models/metadata.json")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CONDITION_MAP = {
    "good": {
        "label": "Good",
        "description": "Tyre appears structurally sound with no visible defects detected.",
        "action": "No immediate action required. Schedule routine inspection per fleet protocol.",
        "color": "#1D9E75",
    },
    "defective": {
        "label": "Defective",
        "description": "Potential defect detected — possible cracking, wear, or structural damage.",
        "action": "Recommend immediate physical inspection. Do not deploy vehicle until cleared.",
        "color": "#D85A30",
    },
}


def _build_model(num_classes: int = 2) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def load_model() -> tuple[nn.Module, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}\n"
            "Run: python train.py  to train the model first."
        )

    with open(META_PATH) as f:
        meta = json.load(f)

    classes = meta.get("classes", ["defective", "good"])
    model = _build_model(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, meta


def predict_image(model: nn.Module, meta: dict, image: Image.Image) -> dict:
    classes = meta["classes"]
    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    info = CONDITION_MAP.get(pred_class, {
        "label": pred_class.title(),
        "description": "",
        "action": "",
        "color": "#888780",
    })

    return {
        "class": pred_class,
        "label": info["label"],
        "confidence": round(confidence * 100, 1),
        "description": info["description"],
        "action": info["action"],
        "color": info["color"],
        "all_probs": {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)},
    }


def predict_batch(model: nn.Module, meta: dict, image_paths: list) -> list[dict]:
    results = []
    for path in image_paths:
        try:
            img = Image.open(path)
            result = predict_image(model, meta, img)
            result["filename"] = Path(path).name
            result["status"] = "ok"
        except Exception as e:
            result = {
                "filename": Path(path).name,
                "status": "error",
                "error": str(e),
                "class": "unknown",
                "confidence": 0,
            }
        results.append(result)
    return results
