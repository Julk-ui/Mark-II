#!/usr/bin/env python3
# models/lstm_model.py
"""
Implementación de un modelo LSTM para predicción.
"""
# --- imports (reemplaza solo este bloque) ---
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# TensorFlow / Keras (robusto para distintas instalaciones)
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:  # fallback si alguien tiene keras standalone
    import keras     # type: ignore
    tf = None        # opcional

# Alias para mantener el resto del archivo igual
Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Adam = keras.optimizers.Adam

from .base_model import BaseModel

class LSTMModel(BaseModel):
    """Modelo LSTM para predicción de series temporales."""

    def _create_dataset(self, dataset, look_back=1):
        """Crea secuencias para el LSTM."""
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """Entrena un modelo LSTM y predice."""
        try:
            # Parámetros
            window = self.params.get("window", 30)
            units = self.params.get("units", 50)
            dropout = self.params.get("dropout", 0.2)
            epochs = self.params.get("epochs", 50)
            batch_size = self.params.get("batch_size", 32)
            lr = self.params.get("learning_rate", 0.001)

            # 1. Escalar datos
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(y_train.values.reshape(-1, 1))

            # 2. Crear secuencias
            X, y = self._create_dataset(scaled_data, window)
            if X.shape[0] == 0:
                self.logger.warning(f"LSTM: No se pudieron crear secuencias con window={window}. Datos insuficientes.")
                return [0] * len(X_test)

            # Reshape para LSTM [samples, time_steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            # 3. Construir el modelo
            self.model = Sequential([
                LSTM(units=units, return_sequences=True, input_shape=(window, 1)),
                Dropout(dropout),
                LSTM(units=units),
                Dropout(dropout),
                Dense(units=1)
            ])
            optimizer = Adam(learning_rate=lr)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')

            # 4. Entrenar
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            # 5. Predecir
            # Tomar los últimos `window` datos del training set para predecir el siguiente paso
            last_sequence = scaled_data[-window:]
            input_for_pred = last_sequence.reshape((1, window, 1))
            
            prediction_scaled = self.model.predict(input_for_pred, verbose=0)
            prediction = scaler.inverse_transform(prediction_scaled)

            return prediction.flatten().tolist() * len(X_test)

        except Exception as e:
            self.logger.error(f"LSTM Error: {e}")
            return [0] * len(X_test)
