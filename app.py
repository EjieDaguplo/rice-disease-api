from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

app = Flask(__name__)

# --- Load your trained model (CPU only) ---
model = YOLO("best.pt")
model.fuse()  # Fuse model layers for better performance

# --- Home endpoint ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Rice Disease Detection API is running!"})

# --- Predict endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Save uploaded file to /tmp (Render writable folder)
        temp_dir = "/tmp"
        saved_path = os.path.join(temp_dir, file.filename)
        file.save(saved_path)
        print(f"Received file: {file.filename}, saved at {saved_path}")

        # --- Resize image to reduce memory usage ---
        img = Image.open(saved_path)
        img = img.resize((640, 640))  # YOLO default input size
        img.save(saved_path)

        # --- Run YOLO prediction (CPU, single thread to save memory) ---
        results = model(saved_path, device='cpu',imgsz=640)

        # Clean up temp file
        os.remove(saved_path)

        detections = results[0].boxes
        if len(detections) == 0:
            return jsonify({"result": "No disease detected"})

        top_detection = detections[0]
        label = results[0].names[int(top_detection.cls[0])]
        confidence = float(top_detection.conf[0])

        return jsonify({
            "disease_name": label,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500
