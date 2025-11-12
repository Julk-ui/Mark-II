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

    def _create_dataset(self, X_data: np.ndarray, y_data: np.ndarray, look_back: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Crea secuencias para el LSTM."""
        dataX, dataY = [], []
        # Empezamos desde 'look_back' para tener suficientes datos pasados
        for i in range(look_back, len(X_data)):
            # La secuencia de features es desde i-look_back hasta i-1
            a = X_data[i-look_back:i, :]
            dataX.append(a)
            # El objetivo es el valor en el momento i
            dataY.append(y_data[i, 0])
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

            if X_train is None:
                self.logger.error("LSTM requiere features (X_train). No se puede entrenar.")
                return [0] * len(X_test)

            # 1. Escalar features y target por separado
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            target_scaler = MinMaxScaler(feature_range=(0, 1))

            scaled_X = feature_scaler.fit_transform(X_train)
            scaled_y = target_scaler.fit_transform(y_train.values.reshape(-1, 1))

            # 2. Crear secuencias
            X, y = self._create_dataset(scaled_X, scaled_y, window)
            if X.shape[0] == 0:
                self.logger.warning(f"LSTM: No se pudieron crear secuencias con window={window}. Datos insuficientes.")
                return [0] * len(X_test)

            # X ya tiene la forma correcta [samples, time_steps, n_features] desde _create_dataset

            # 3. Construir el modelo
            n_features = X.shape[2]
            self.model = Sequential([
                LSTM(units=units, return_sequences=True, input_shape=(window, n_features)),
                Dropout(dropout),
                LSTM(units=units),
                Dropout(dropout),
                Dense(units=1)
            ])
            
            optimizer = Adam(learning_rate=lr)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')

            # 4. Entrenar
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

            # 5. Predecir
            # Tomar las últimas `window` filas de features del set de entrenamiento para predecir
            last_sequence_features = scaled_X[-window:]
            input_for_pred = last_sequence_features.reshape((1, window, n_features))
            
            prediction_scaled = self.model.predict(input_for_pred, verbose=0)
            # Revertir la escala de la predicción usando el 'target_scaler'
            prediction = target_scaler.inverse_transform(prediction_scaled)

            # Replicar la predicción para el tamaño de X_test (normalmente 1 en backtest)
            return prediction.flatten().tolist() * len(X_test) if X_test is not None else prediction.flatten().tolist()

        except Exception as e:
            self.logger.error(f"LSTM Error: {e}")
            return [0] * len(X_test)
