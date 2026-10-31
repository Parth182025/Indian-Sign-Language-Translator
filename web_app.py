import base64
import io
import os
import time
import threading
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
from recognizer import ISLRecognizer

app = Flask(__name__)
recognizer = None
recognizer_lock = threading.Lock()

@app.before_request
def load_recognizer():
    global recognizer
    if recognizer is None:
        with recognizer_lock:
            if recognizer is None:
                recognizer = ISLRecognizer()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data)
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        with recognizer_lock:
            result = recognizer.process(frame, draw=False)
        return jsonify({
            "prediction": result["prediction"],
            "confidence": round(result["confidence"] * 100, 1),
            "status": result["status"],
            "frames": result["frames_collected"],
            "hands_detected": result["hands_detected"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local testing only. For deployment, use a production WSGI server.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
