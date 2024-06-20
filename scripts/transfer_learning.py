from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from scripts.preprocess import preprocess_data


def create_transfer_learning_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(26, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


def train_model(model, train_data, val_data, epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=epochs)
    model.save('../models/asl_model_transfer_learning.keras')
    return history


if __name__ == "__main__":
    data_dir = "../data/asl_alphabet_train"
    img_size = (64, 64, 3)

    train_data, val_data = preprocess_data(data_dir)
    model = create_transfer_learning_model(img_size)

    history = train_model(model, train_data, val_data)
    print("Model trained and saved to ../models/asl_model_transfer_learning.keras")
