from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Force CPU only (Render has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Absolute paths (important for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_PATH = os.path.join(BASE_DIR, UPLOAD_FOLDER)
MODEL_PATH = os.path.join(BASE_DIR, "mask_model.h5")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)

def predict_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid Image ❌"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face Detected ❌"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = np.reshape(face, (1, 128, 128, 3))

    pred = model.predict(face, verbose=0)[0][0]

    if pred < 0.45:
        return "No Mask ❌"
    elif pred > 0.55:
        return "Mask 😷"
    else:
        return "Uncertain 🤔"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_url = ""

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result="No file uploaded ❌")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result="No file selected ❌")

        # 🔥 CRITICAL FIX: Always recreate folder
        os.makedirs(UPLOAD_PATH, exist_ok=True)

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_PATH, filename)

        file.save(image_path)

        result = predict_mask(image_path)
        image_url = f"{UPLOAD_FOLDER}/{filename}"

    return render_template("index.html", result=result, image=image_url)

# Local testing only
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
