from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Force CPU (Render has no GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ------------------ Prediction Function ------------------

def predict_mask(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return "Invalid image ❌"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "No face detected ❌"

    # Use first detected face
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (128, 128))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face, verbose=0)[0][0]

    if pred < 0.45:
        return "No Mask ❌"
    elif pred > 0.55:
        return "Mask 😷"
    else:
        return "Uncertain 🤔"

# ------------------ Routes ------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result="No file uploaded ❌")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result="No file selected ❌")

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        result = predict_mask(image_path)

    return render_template("index.html", result=result, image=image_path)

# ------------------ Entry Point ------------------

if __name__ == "__main__":
    app.run()
