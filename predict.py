import cv2
import numpy as np
from keras.models import load_model

# Load pre-trained model
model = load_model('model/age_gender_model.h5', compile=False)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img_resized = cv2.resize(face_img, (100, 100))  # Match model input
        face_input = np.expand_dims(face_img_resized, axis=0)
        face_input = face_input / 255.0  # Normalize

        # Predict
        predictions = model.predict(face_input)
        print("Predictions:", predictions)

        try:
            age = int(predictions[0][0][0])  # Age from shape (1,1)
            gender_probs = predictions[1][0]  # Gender from shape (1,2)
            gender = 'Male' if gender_probs[0] > gender_probs[1] else 'Female'
            label = f"{gender}, {age} yrs"
        except Exception as e:
            print("Prediction Error:", e)
            label = "Invalid model output"

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Age & Gender Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
