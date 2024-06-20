import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model('../models/asl_model.h5')

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def preprocess_hand_image(image, hand_landmarks, img_size=(64, 64)):
    # Extrair a região da mão com base nos landmarks
    img_height, img_width, _ = image.shape
    bbox = [
        min([lm.x for lm in hand_landmarks.landmark]) * img_width,
        min([lm.y for lm in hand_landmarks.landmark]) * img_height,
        max([lm.x for lm in hand_landmarks.landmark]) * img_width,
        max([lm.y for lm in hand_landmarks.landmark]) * img_height
    ]
    bbox = [int(v) for v in bbox]
    hand_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # Redimensionar a imagem da mão
    hand_image = cv2.resize(hand_image, img_size[:2])
    hand_image = hand_image / 255.0  # Normalizar a imagem
    hand_image = np.expand_dims(hand_image, axis=0)  # Adicionar dimensão do lote
    return hand_image


# Abrir a câmera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converter a imagem de BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar a imagem para encontrar mãos
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Desenhar landmarks na imagem
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Pré-processar a imagem da mão
                hand_image = preprocess_hand_image(frame, hand_landmarks)

                # Prever o gesto
                prediction = model.predict(hand_image)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                # Mapear a classe para o gesto correspondente (A-Z)
                gesture = chr(65 + class_id)  # 65 é o código ASCII para 'A'

                # Exibir o gesto e a confiança na imagem
                cv2.putText(frame, f'{gesture} ({confidence:.2f})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Mostrar o quadro
        cv2.imshow('ASL Recognition', frame)

        # Sair do loop se a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
