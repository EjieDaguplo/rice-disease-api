from flask import Flask, request, jsonify
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

app = Flask(__name__)

# --- Load your trained model ---
# Make sure 'best.pt' is in the same folder as this file
model = YOLO("best.pt")
model.fuse()    # Fuse model layers for better performance

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Rice Disease Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']

        # Save uploaded file properly
        temp_dir = tempfile.gettempdir()
        filename = file.filename
        saved_path = os.path.join(temp_dir, filename)
        file.save(saved_path)
        print("Received file:", filename)
        print("Saved at:", saved_path)

        # Run YOLO prediction
        results = model(saved_path, device='cpu')  # Use 'cuda' if GPU is available

        # Clean up the saved file
        os.remove(saved_path)

        detections = results[0].boxes
        if len(detections) == 0:
            return jsonify({"result": "No disease detected"})

        top_detection = detections[0]
        label = results[0].names[int(top_detection.cls[0])]
        confidence = float(top_detection.conf[0])

        #os.remove(saved_path)
        return jsonify({
            "disease_name": label,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


#if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 10000))
    #app.run(host="0.0.0.0", port=port)
