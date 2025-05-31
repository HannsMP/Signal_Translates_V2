import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np


class XYZ:
    @staticmethod
    def rotate_axis_X(data, angle: int):
        theta = np.radians(angle)
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        data[:, 0:3] = data[:, 0:3] @ R.T

    @staticmethod
    def rotate_axis_Y(data, angle: int):
        theta = np.radians(angle)
        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])
        data[:, 0:3] = data[:, 0:3] @ R.T

    @staticmethod
    def rotate_axis_Z(data, angle: int):
        theta = np.radians(angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        data[:, 0:3] = data[:, 0:3] @ R.T

    @staticmethod
    def scale_X(data, scale: float = 1):
        if scale != 1:
            data[:, 0] *= scale

    @staticmethod
    def scale_Y(data, scale: float = 1):
        if scale != 1:
            data[:, 1] *= scale

    @staticmethod
    def scale_Z(data, scale: float = 1):
        if scale != 1:
            data[:, 2] *= scale

    @staticmethod
    def scale(data, scale: float = 1):
        XYZ.scale_X(data, scale)
        XYZ.scale_Y(data, scale)
        XYZ.scale_Z(data, scale)

    @staticmethod
    def mirror_YZ(data):
        XYZ.scale_X(data, -1)

    @staticmethod
    def mirror_ZX(data):
        XYZ.scale_Y(data, -1)

    @staticmethod
    def mirror_XY(data):
        XYZ.scale_Z(data, -1)

    @staticmethod
    def offset_X(data, offset):
        data[:, 0] += offset

    @staticmethod
    def offset_Y(data, offset):
        data[:, 1] += offset

    @staticmethod
    def offset_Z(data, offset):
        data[:, 2] += offset

    def offset(data, x, y, z):
        XYZ.offset_X(x)
        XYZ.offset_Y(y)
        XYZ.offset_Z(z)


class Render_3d:
    def __init__(self, scale_z=2.5, title="3D", ax=0, ay=110, aw=800, ah=600):
        self.scale_z = scale_z
        self.app = QApplication(sys.argv)

        self.window = gl.GLViewWidget()
        self.window.setWindowTitle(title)
        self.window.setGeometry(ax, ay, aw, ah)
        self.window.opts['distance'] = 600
        self.window.show()

        self.axes = gl.GLAxisItem()
        self.axes.setSize(150, 150, 150)
        self.window.addItem(self.axes)

        self.items = []
        self.handBounds = [
            # dedo pulgar
            (0, 1, 4), (1, 2, 4), (2, 3, 2), (3, 4, 2),
            # dedo índice
            (5, 6, 2), (6, 7, 2), (7, 8, 2),
            # dedo medio
            (9, 10, 2), (10, 11, 2), (11, 12, 2),
            # dedo anular
            (13, 14, 2), (14, 15, 2), (15, 16, 2),
            # dedo meñique
            (0, 17, 4), (17, 18, 2), (18, 19, 2), (19, 20, 2),
            # area de la mano
            (1, 5, 4), (2, 5, 4), (5, 9, 4), (9, 13, 4), (13, 17, 4)
        ]

    def _addItem(self, item):
        self.window.addItem(item)
        self.items.append(item)

    def setTimeOut(self, fun, time=10):
        QTimer.singleShot(time, fun)

    def hand(self, data, color=(0.99, 0.86, 0.79, 1), size=5, width=10):
        """
        Agrega los puntos y huesos de la mano
        """

        points = gl.GLScatterPlotItem(
            pos=data,
            color=(1, 1, 1, 1),
            size=size
        )
        self._addItem(points)

        for i, j, w in self.handBounds:
            line = gl.GLLinePlotItem(
                pos=np.array([data[i], data[j]]),
                color=color,
                width=width
            )
            self._addItem(line)

    def clear(self):
        """
        Elimina todos los objetos agregados
        """

        for item in self.items:
            self.window.removeItem(item)
        self.items.clear()

    def start(self):
        self.app.exec_()


if (__name__ == '__main__'):
    from PyQt5.QtCore import QTimer

    render = Render_3d()

    data = np.array([
        # X(blue), Y(yellow), Z(green)
        [0, 0, 0],
        [36, -16, -22],
        [64, -64, -30],
        [54, -106, -35],
        [20, -114, -40],
        [47, -123, -13],
        [67, -167, -24],
        [77, -195, -35],
        [86, -219, -43],
        [20, -133, -13],
        [29, -184, -22],
        [36, -215, -29],
        [42, -242, -34],
        [-6, -128, -17],
        [-7, -180, -28],
        [-3, -209, -34],
        [1, -232, -39],
        [-32, -112, -24],
        [-46, -148, -36],
        [-53, -176, -43],
        [-57, -203, -47]
    ], dtype=float)

    XYZ.rotate_axis_X(data, -90)
    XYZ.rotate_axis_Z(data, 135)

    render.hand(data)


    def update_scene():
        render.clear()
        XYZ.scale(data, 0.5)
        QTimer.singleShot(1000, lambda: render.hand(data))


    QTimer.singleShot(3000, update_scene)

    # Ejecuta la aplicación solo una vez
    render.start()
