import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# Function to load data from directory
def load_data(data_dir):
    images = []
    labels = []
    label_map = {}
    label_index = 0

    # Iterate through each subdirectory (each sign language gesture)
    for label in os.listdir(data_dir):
        label_map[label_index] = label
        for image_file in os.listdir(os.path.join(data_dir, label)):
            image_path = os.path.join(data_dir, label, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            images.append(image)
            labels.append(label_index)
        label_index += 1

    images = np.array(images)
    labels = np.array(labels)

    return images, labels, label_map


# Function to extract landmarks using MediaPipe
def extract_landmarks(images):
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    landmarks = []

    for image in images:
        results = mp_hands.process(image)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            hand_landmarks_list = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks])
            landmarks.append(hand_landmarks_list)
        else:
            landmarks.append(np.zeros((21, 3)))  # If no landmarks detected, fill with zeros

    landmarks = np.array(landmarks)
    return landmarks


# Custom callback to display training progress
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}/{self.params['epochs']}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, "
              f"Validation Loss: {logs['val_loss']:.4f}, Validation Accuracy: {logs['val_accuracy']:.4f}")


# Main training script
def main():
    data_dir = 'asl_alphabet_train'
    images, labels, label_map = load_data(data_dir)

    landmarks = extract_landmarks(images)

    x_train, x_test, y_train, y_test = train_test_split(landmarks, labels, test_size=0.2, random_state=42)

    model = Sequential([
        Flatten(input_shape=(21, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(label_map), activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('asl_model.h5', save_best_only=True),
        TrainingProgressCallback()
    ]

    # Train the model with callbacks
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, callbacks=callbacks)

    # Save the final model
    model.save('asl_model_final.h5')


if __name__ == "__main__":
    main()
