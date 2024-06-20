import cv2
import numpy as np
import tensorflow as tf


def recognize_gesture(model_path):
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = img.reshape(1, 64, 64, 3)

        prediction = model.predict(img)
        gesture = chr(np.argmax(prediction) + ord('A'))

        cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = '../models/asl_model_transfer_learning.h5'
    recognize_gesture(model_path)
