from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import os
import cv2


class Signal_hands:

    def __init__(self, fps=1000, maxHands=1):
        self.frame_delay = 1000 // fps

        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=maxHands)

        self.off_point = [0, 0, 0]
        self.off_hand = [self.off_point[:] for _ in range(21)]

        # cerrar
        self.key_c = ord("c")

        self.display_text = "fuera de rango"
        self.display_font = cv2.FONT_HERSHEY_SIMPLEX
        self.display_scale = 0.6
        self.display_color = (0, 255, 0)
        self.display_width = 1
        self.model = tf.keras.models.load_model('../model/model_hands_v1.h5')

    def mk_check(self, folder):
        os.makedirs(folder, exist_ok=True)

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

    def process(self, key, img):
        hands, imgFind = self.detector.findHands(img)

        if (not hands):
            return img

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

        print({
            "left": hand_left,
            "rigth": hand_rigth,
            "absolute": absolute
        })

        return imgFind

    def run(self):
        while (True):
            key = cv2.waitKey(self.frame_delay)
            success, img = self.read_cap()

            try:
                if (success):
                    img = self.process(key, img)
                self.display_text = f"test"

            except:
                self.display_text = f"test"

            self.stamp_display_text(img)
            cv2.imshow("Camara", img)

            if (key == -1):
                continue

            elif (key == self.key_c):
                break

        self.end_cap()


if (__name__ == "__main__"):
    translate = Signal_hands(
        maxHands=1,
        fps=50,
    )

    translate.run()
