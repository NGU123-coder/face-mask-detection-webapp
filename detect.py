import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("mask_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = face / 255.0
        face = np.reshape(face, (1, 128, 128, 3))
        pred = model.predict(face)[0][0]

        if pred < 0.45:
            label = "No Mask"
            color = (0, 0, 255)
        elif pred > 0.55:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "Uncertain"
            color = (0, 255, 255)


        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
