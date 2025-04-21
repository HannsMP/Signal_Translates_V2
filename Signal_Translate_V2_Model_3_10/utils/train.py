import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

class TrainHand:
    def __init__(self, data_folder='data', model_output='model_hand.h5'):
        self.data_folder = data_folder
        self.model_output = model_output
        self.X = []
        self.y = []
        self.label_encoder = LabelEncoder()

    def load_data(self):
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_folder, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    label = data.get('label')
                    coords = data.get('coordinates')
                    if label and coords:
                        flat_coords = np.array(coords).flatten()  # Convert to 63 values (21 * 3)
                        self.X.append(flat_coords)
                        self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = self.label_encoder.fit_transform(self.y)
        self.y = to_categorical(self.y)
        print(f"Datos cargados: {self.X.shape[0]} muestras")

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X.shape[1],)),
            Dense(64, activation='relu'),
            Dense(self.y.shape[1], activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(self.model_output, monitor='val_accuracy', save_best_only=True, verbose=1)

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, callbacks=[checkpoint])

        print(f"Modelo guardado en {self.model_output}")

    def run(self):
        self.load_data()
        self.train()
