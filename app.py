from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

# Force CPU (important for Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder safely (Render FIX)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_model.h5")

model = tf.keras.models.load_model(MODEL_PATH)

# Face detector (load once)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image ❌"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No Face Detected ❌"

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = np.reshape(face, (1, 128, 128, 3))

    pred = model.predict(face)[0][0]

    if pred < 0.45:
        return "No Mask ❌"
    elif pred > 0.55:
        return "Mask 😷"
    else:
        return "Uncertain 🤔"

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = ""

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)
            result = predict_mask(image_path)

    return render_template("index.html", result=result, image=image_path)

# DO NOT hardcode port in production
if __name__ == "__main__":
    app.run()
