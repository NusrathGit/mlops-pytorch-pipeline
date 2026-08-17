"""
serve.py
========
Flask app serving the trained PyTorch image classifier.

Endpoints:
    GET  /health   -> 200 if the model checkpoint is loaded, else 503
    POST /predict  -> accepts an image file, returns class probabilities

Configuration via environment variables:
    MODEL_PATH  - path to the .pt checkpoint (default: /app/checkpoints/classifier_v1.pt)
    PORT        - port to listen on (default: 8080, per assignment spec)
"""

import io
import logging
import os

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

from model import get_model

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")
PORT = int(os.environ.get("PORT", 8080))

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


def load_model():
    global model
    logger.info("Loading checkpoint from %s", MODEL_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    net = get_model(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()

    model = net
    logger.info(
        "Model loaded (architecture=%s, val_accuracy=%.4f)",
        checkpoint["architecture"], checkpoint.get("val_accuracy", float("nan")),
    )


@app.route("/health", methods=["GET"])
def health():
    is_ready = model is not None
    return jsonify({"status": "healthy" if is_ready else "unhealthy", "model_loaded": is_ready}), \
        (200 if is_ready else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded yet. Try again shortly."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Missing required file field 'image'."}), 400

    try:
        image_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    try:
        input_tensor = PREPROCESS(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        predicted_idx = int(torch.argmax(probs).item())
        return jsonify({
            "predicted_class": CLASS_NAMES[predicted_idx],
            "probabilities": {
                name: round(float(probs[i]), 4) for i, name in enumerate(CLASS_NAMES)
            },
        })
    except Exception:
        logger.exception("Inference failed.")
        return jsonify({"error": "Internal error during inference."}), 500


try:
    load_model()
except FileNotFoundError:
    logger.warning("No checkpoint found at %s yet - /health will report unhealthy until it exists.", MODEL_PATH)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)