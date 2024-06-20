import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_data(data_dir, img_size=(64, 64), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator


if __name__ == "__main__":
    data_dir = "../data/asl_alphabet_train"
    train_data, val_data = preprocess_data(data_dir)
    print(f"Train data: {len(train_data)} batches")
    print(f"Validation data: {len(val_data)} batches")
