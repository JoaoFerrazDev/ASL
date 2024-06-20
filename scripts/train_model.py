from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from scripts.preprocess import preprocess_data


def create_base_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(26, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


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
    model.save('../models/asl_model.h5')
    return history


if __name__ == "__main__":
    data_dir = "../data/asl_alphabet_train"
    img_size = (64, 64, 3)

    train_data, val_data = preprocess_data(data_dir)

    use_transfer_learning = True  # Mude para False para usar o modelo base

    if use_transfer_learning:
        model = create_transfer_learning_model(img_size)
    else:
        model = create_base_model(img_size)

    history = train_model(model, train_data, val_data)
    print("Model trained and saved to ../models/asl_model.h5")
