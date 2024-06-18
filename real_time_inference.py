import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


# Function to extract landmarks using MediaPipe
def extract_landmarks(image):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)

    # Convert image to RGB (MediaPipe requires RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image to extract hand landmarks
    results = mp_hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        hand_landmarks_list = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])
    else:
        hand_landmarks_list = np.zeros((21, 3))  # If no landmarks detected, fill with zeros

    return hand_landmarks_list


# Function for real-time inference
def real_time_inference(model_path):
    # Load trained model
    model = load_model(model_path)

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for hand landmarks
        landmarks = extract_landmarks(frame)

        # Perform inference
        prediction = model.predict(np.expand_dims(landmarks, axis=0))
        predicted_label = np.argmax(prediction)

        # Display prediction on frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Real-Time Inference', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'asl_model.h5'  # Path to your trained model
    real_time_inference(model_path)
