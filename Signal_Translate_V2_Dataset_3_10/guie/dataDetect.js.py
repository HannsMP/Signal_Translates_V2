import time
import math
import os
import json
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

# ==================== CONFIGURATION ====================
# Parámetros del recorte
offset = 20
imgSize = 300
imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)
# Creamos el directorio para guardar el JSON si no existe
json_dir = "json"

# ==================== INIT ====================
# Inicializamos la cámara y el detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
os.makedirs(json_dir, exist_ok=True)
# Contador de muestras y lista de almacenamiento
counter = 0
samples = []
label_name = "B"

# ==================== TEMPLATES ====================
xyz_template = [0, 0, 0]
hand_template = [xyz_template[:] for _ in range(21)]


# ==================== FUNCTIONS ====================
def convertir_a_relativo(lmList, center):
    """
    Transforma los landmarks de MediaPipe en coordenadas relativas,
    tomando como origen el punto 'center' (usualmente la muñeca).
    """
    return [[p[0] - center[0], p[1] - center[1], p[2] - center[2]] for p in lmList]


# Variable para mantener el texto de la muestra a mostrar de manera persistente
display_text = "fuera de rango"

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)

    # Valores por defecto (21 puntos en 3D) para cada mano: se asignan 0 si la mano no se detecta
    left_hand = hand_template[:]
    right_hand = hand_template[:]
    center_left = xyz_template[:]
    center_rigth = xyz_template[:]

    lenHands = len(hands)

    if hands:
        if len(hands) == 1:
            hand = hands[0]

            lm = hand['lmList']
            center = lm[0]  # Usamos la muñeca como center
            rel_coords = convertir_a_relativo(lm, center)

            if hand["type"] == "Left":
                left_hand = rel_coords
                center_left = center
            else:
                right_hand = rel_coords
                center_rigth = center

            x, y, w, h = hand['bbox']
        elif len(hands) == 2:
            hand1, hand2 = hands

            for hand in (hand1, hand2):
                lm = hand['lmList']
                center = lm[0]
                rel_coords = convertir_a_relativo(lm, center)

                if hand["type"] == "Left":
                    left_hand = rel_coords
                    center_left = center
                else:
                    right_hand = rel_coords
                    center_rigth = center

            # Se calcula la hitbox global a partir de las dos manos
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']
            x, y = min(x1, x2), min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

        try:
            # Extraer la imagen de la hitbox con un offset
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]
            # Rellenamos el plano blanco
            imgWhite[:] = 255
            aspecRadio = h / w

            if aspecRadio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Aquí se puede dibujar algo en el imgWhite si se quiere (opcional)

        except Exception as e:
            display_text = "fuera de rango"

    # Calcular la distancia absoluta entre centros (muñecas) de cada mano
    dx = center_rigth[0] - center_left[0]
    dy = center_rigth[1] - center_left[1]
    dz = center_rigth[2] - center_left[2]
    distancia_absoluta = [dx, dy, dz]

    # Overlay de texto en la parte inferior izquierda del frame de la cámara
    # Se usan constantes de fuente, tamaño y color
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (0, 255, 0)  # Verde
    thickness = 1
    # Se obtiene el tamaño de la imagen para posicionar el texto en la esquina inferior izquierda
    (text_width, text_height), _ = cv2.getTextSize(display_text, font, scale, thickness)
    pos_x = 10
    pos_y = img.shape[0] - 10
    cv2.putText(img, display_text, (pos_x, pos_y), font, scale, color, thickness, cv2.LINE_AA)

    # Mostrar la imagen de la cámara
    cv2.imshow("Camara", img)
    cv2.imshow("DetectorWhite", imgWhite)

    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        display_text = f"muestra numero: {counter}"

        # Guardamos la muestra en la lista
        sample = {
            "label_name": label_name,
            "left": left_hand,
            "right": right_hand,
            "absolute": distancia_absoluta
        }
        samples.append(sample)

    # Se rompe el loop al obtener 300 muestras
    if counter >= 300:
        break

# Guardamos el JSON en el directorio indicado
json_filename = f"json/{label_name}.json"
with open(json_filename, "w") as f:
    json.dump(samples, f, indent=2)

cap.release()
cv2.destroyAllWindows()
