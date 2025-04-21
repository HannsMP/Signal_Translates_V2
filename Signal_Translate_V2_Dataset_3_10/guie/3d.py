import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QApplication
import sys
import numpy as np

# ==== DATOS DE UNA MANO DE EJEMPLO ====
# Cada punto es (x, y, z) relativo a la muñeca (índice 0)
mano_coords = np.array([
    [
        0,
        0,
        0
    ],
    [
        36,
        -16,
        -22
    ],
    [
        64,
        -64,
        -30
    ],
    [
        54,
        -106,
        -35
    ],
    [
        20,
        -114,
        -40
    ],
    [
        47,
        -123,
        -13
    ],
    [
        67,
        -167,
        -24
    ],
    [
        77,
        -195,
        -35
    ],
    [
        86,
        -219,
        -43
    ],
    [
        20,
        -133,
        -13
    ],
    [
        29,
        -184,
        -22
    ],
    [
        36,
        -215,
        -29
    ],
    [
        42,
        -242,
        -34
    ],
    [
        -6,
        -128,
        -17
    ],
    [
        -7,
        -180,
        -28
    ],
    [
        -3,
        -209,
        -34
    ],
    [
        1,
        -232,
        -39
    ],
    [
        -32,
        -112,
        -24
    ],
    [
        -46,
        -148,
        -36
    ],
    [
        -53,
        -176,
        -43
    ],
    [
        -57,
        -203,
        -47
    ]
], dtype=float)

mano_coords[:, 2] *= 2.5

# ==== CONEXIONES ENTRE PUNTOS ====
# Índices de puntos conectados como si fueran huesos
huesos = [
    (0, 1, 4), (1, 2, 4), (2, 3, 2), (3, 4, 2),  # pulgar
    (0, 5, 4), (5, 6, 2), (6, 7, 2), (7, 8, 2),  # índice
    (0, 9, 4), (9, 10, 2), (10, 11, 2), (11, 12, 2),  # medio
    (0, 13, 4), (13, 14, 2), (14, 15, 2), (15, 16, 2),  # anular
    (0, 17, 4), (17, 18, 2), (18, 19, 2), (19, 20, 2),  # meñique
    (1, 5, 4), (2, 5, 4), (5, 9, 4), (9, 13, 4), (13, 17, 4)
]

app = QApplication(sys.argv)

window = gl.GLViewWidget()
window.setWindowTitle("Visualizador 3D de coordenadas")
window.setGeometry(0, 110, 800, 600)
window.opts['distance'] = 600  # Distancia inicial de la cámara
# window.setBackgroundColor('w')
window.show()

# Ejes de coordenadas
axes = gl.GLAxisItem()
axes.setSize(x=150, y=150, z=150)
window.addItem(axes)

# Ejemplo de puntos
puntos_item = gl.GLScatterPlotItem(pos=mano_coords, color=(1, 0, 0, 1), size=5)
window.addItem(puntos_item)

for i, j, w in huesos:
    linea = gl.GLLinePlotItem(
        pos=np.array([mano_coords[i], mano_coords[j]]),
        color=(0.2, 0.2, 0.2, 1),
        width=20
    )
    window.addItem(linea)

sys.exit(app.exec_())