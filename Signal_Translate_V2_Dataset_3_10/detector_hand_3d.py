from utils.render_3d import Render_3d, XYZ
# cvzone == 1.6.1
from cvzone.HandTrackingModule import HandDetector
#numpy == 1.26.2
import numpy as np
import json
import os
import cv2


class Signal_hands_3d:

    def __init__(self, limit=300, fps=1000, label_name="test", maxHands=1):
        self.limit = limit
        self.frame_delay = 1000 // fps

        self.label_name = label_name
        self.mk_check("../data")
        self.file_save = f"../data/{label_name}.json"

        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=maxHands)
        self.samples = []

        self.off_point = [0, 0, 0]
        self.off_hand = [self.off_point[:] for _ in range(21)]

        # guardar
        self.key_s = ord("s")
        # cerrar
        self.key_c = ord("c")
        # retirar el ultimo
        self.key_d = ord("d")
        # vaciar muestras
        self.key_v = ord("v")
        # pintar coordenadas
        self.key_a = ord("a")

        self.display_text = "fuera de rango"
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX
        self.display_scale = 0.6
        self.display_color = (0, 255, 0)
        self.display_width = 1

        self.render = Render_3d()

    def mk_check(self, folder):
        os.makedirs(folder, exist_ok=True)

    def save_json(self):
        with open(self.file_save, "w") as f:
            json.dump(self.samples, f, indent=2)

    def read_cap(self):
        return self.cap.read()

    def end_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def relative_center(self, lmList, center):
        c0, c1, c2 = center
        return [[
            p[0] - c0,
            p[1] - c1,
            p[2] - c2
        ] for p in lmList]

    def stamp_display_text(self, img):
        cv2.putText(
            img=img,
            text=self.display_text,
            org=(10, img.shape[0] - 10),
            fontFace=self.display_font,
            fontScale=self.display_scale,
            color=self.display_color,
            thickness=self.display_width,
            lineType=cv2.LINE_AA
        )

    def render_hands(self, hand_left, hand_rigth):
        self.render.clear()

        XYZ.mirror_YZ(hand_left)
        XYZ.mirror_YZ(hand_rigth)

        self.render.hand(hand_left)
        self.render.hand(hand_rigth)

    def process(self, key, img):
        hands, imgFind = self.detector.findHands(img)

        if (not hands):
            return img

        if (key != self.key_s and key != self.key_a):
            return imgFind

        hand_len = len(hands)

        center_left = self.off_point
        center_rigth = self.off_point
        absolute = self.off_point

        hand_left = self.off_hand
        hand_rigth = self.off_hand

        hand_1 = hands[0]

        if (hand_len == 1 or hand_len == 2):
            lm = hand_1["lmList"]

            if (hand_1["type"] == "Left"):
                center_left = lm[0]
                hand_left = self.relative_center(lm, center_left)

            else:
                center_rigth = lm[0]
                hand_rigth = self.relative_center(lm, center_rigth)

        if (hand_len == 2):
            hand_1 = hands[0]
            hand_2 = hands[1]
            lm = hand_2["lmList"]

            if (hand_2["type"] == "Left"):
                if (hand_1["type"] == "Right"):
                    center_left = lm[0]
                    hand_left = self.relative_center(lm, center_left)

            else:
                if (hand_1["type"] == "Left"):
                    center_rigth = lm[0]
                    hand_rigth = self.relative_center(lm, center_rigth)

            absolute[0] = center_rigth[0] - center_left[0]
            absolute[1] = center_rigth[1] - center_left[1]
            absolute[2] = center_rigth[2] - center_left[2]

        if (key == self.key_s):
            self.samples.append({
                "label_name": self.label_name,
                "left": hand_left,
                "rigth": hand_rigth,
                "absolute": absolute
            })

        if (key == self.key_a):
            self.render_hands(hand_left, hand_rigth)

        return imgFind

    def run(self):
        while (True):
            key = cv2.waitKey(self.frame_delay)
            success, img = self.read_cap()

            try:
                if (success):
                    img = self.process(key, img)
                self.display_text = f"'{self.label_name}': [{len(self.samples)}/{self.limit}]"

            except:
                self.display_text = f"'{self.label_name}': [{len(self.samples)}/{self.limit}] Error de procesamiento"

            self.stamp_display_text(img)
            cv2.imshow("Camara", img)

            if (len(self.samples) >= self.limit):
                break

            if (key == -1):
                continue

            elif (key == self.key_c):
                break

            elif (key == self.key_d):
                self.samples and self.samples.pop()

            elif (key == self.key_v):
                self.samples.clear()

        self.save_json()
        self.end_cap()


if (__name__ == "__main__"):
    translate = Signal_hands_3d(
        maxHands=1,
        limit=3000,
        fps=30,
        label_name="L"
    )

    translate.run()
