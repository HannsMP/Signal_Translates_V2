# Flask == 3.0.3
from tabnanny import verbose

from flask import Flask, request, jsonify, render_template
# TensorFlow == 2.19.0
import tensorflow as tf
# numpy == 1.26.2
import numpy as np
# opencv-python == 4.11.0.86
# opencv-contrib-python == 4.11.0.86
# mediapipe == 0.10.14
# keras == 3.9.2
import cv2
import os
import pickle
import base64
import logging

# cvzone == 1.6.1
from cvzone.HandTrackingModule import HandDetector

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__, template_folder='Server/')

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class Signal_hands:
    def __init__(self, maxHands=1):
        self.detector = HandDetector(maxHands=maxHands)

        self.off_point = [0, 0, 0]
        self.off_hand = [self.off_point[:] for _ in range(21)]

        self.model = tf.keras.models.load_model('../model/v1_model_hands.h5')

        # Prueba de prediccion con un vector de tamano correcto
        self.model.predict(np.zeros((1, 129), dtype=np.float32), verbose=False)
        print("Modelo cargado y listo")

        with open('../model/v1_label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

    def relative_center(self, lmList, center):
        c0, c1, c2 = center
        return [[p[0] - c0, p[1] - c1, p[2] - c2] for p in lmList]

    def predict_from_image(self, img):
        hands, _ = self.detector.findHands(img, draw=False)

        if not hands:
            return img, False, "fuera de rango"

        hand_len = len(hands)

        center_left = self.off_point
        center_rigth = self.off_point
        absolute = self.off_point

        hand_left = self.off_hand
        hand_rigth = self.off_hand

        hand_1 = hands[0]
        lm = hand_1["lmList"]

        if len(lm) == 21:
            if hand_1["type"] == "Left":
                center_left = lm[0]
                hand_left = self.relative_center(lm, center_left)
            else:
                center_rigth = lm[0]
                hand_rigth = self.relative_center(lm, center_rigth)
        else:
            print("Advertencia: mano 1 incompleta (menos de 21 puntos)")

        if hand_len == 2:
            hand_2 = hands[1]
            lm = hand_2["lmList"]

            if len(lm) == 21:
                if hand_2["type"] == "Left" and hand_1["type"] == "Right":
                    center_left = lm[0]
                    hand_left = self.relative_center(lm, center_left)
                elif hand_2["type"] == "Right" and hand_1["type"] == "Left":
                    center_rigth = lm[0]
                    hand_rigth = self.relative_center(lm, center_rigth)

                # Calcular vector absoluto
                absolute = [
                    center_rigth[0] - center_left[0],
                    center_rigth[1] - center_left[1],
                    center_rigth[2] - center_left[2],
                ]
            else:
                print("Advertencia: mano 2 incompleta (menos de 21 puntos)")

        # Construir input
        input_data = np.concatenate([
            np.array(hand_left, dtype=np.float32).flatten(),
            np.array(hand_rigth, dtype=np.float32).flatten(),
            np.array(absolute, dtype=np.float32)
        ])

        # Rellenar con ceros si falta tamano
        if input_data.shape[0] < 129:
            input_data = np.pad(input_data, (0, 129 - input_data.shape[0]), mode='constant')

        input_data = np.expand_dims(input_data, axis=0)

        prediction = self.model.predict(input_data, verbose=False)
        predicted_index = np.argmax(prediction)
        predicted_label = self.label_encoder.inverse_transform([predicted_index])[0]

        return img, True, predicted_label


# Instancia del modelo de predicciÃ³n
translator = Signal_hands(maxHands=1)


@app.route('/')
def index():
    return render_template('camara.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img_out, has_hand, label = translator.predict_from_image(img)

        # Mostrar resultado en imagen
        if has_hand:
            cv2.putText(img_out, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', img_out)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_base64 = 'data:image/jpeg;base64,' + img_base64

        return jsonify({
            'image': img_base64,
            'hasHand': has_hand,
            'letter': label
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Error al procesar la imagen'})


if __name__ == '__main__':
    print("Servidor corriendo en https://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, ssl_context=('certificate/server.crt', 'certificate/server.key'))
