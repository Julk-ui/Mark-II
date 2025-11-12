#!/usr/bin/env python3
# models/prophet_model.py
"""
Implementación del modelo Prophet.
"""
from __future__ import annotations
import pandas as pd
from prophet import Prophet
from .base_model import BaseModel

class ProphetModel(BaseModel):
    """Modelo Prophet para predicción de series temporales."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """Entrena un modelo Prophet y predice."""
        try:
            # Prophet requiere columnas 'ds' y 'y'
            df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

            self.model = Prophet(**self.params)

            # Añadir regresores si existen en X_train
            if X_train is not None and not X_train.empty:
                regressors = [col for col in X_train.columns if col in X_test.columns]
                for regressor in regressors:
                    self.model.add_regressor(regressor)
                df_train = pd.concat([df_train, X_train[regressors].reset_index(drop=True)], axis=1)

            self.model.fit(df_train)

            # Crear dataframe futuro para predicción
            df_future = pd.DataFrame({'ds': X_test.index})
            if X_train is not None and not X_train.empty:
                 df_future = pd.concat([df_future, X_test[regressors].reset_index(drop=True)], axis=1)

            forecast = self.model.predict(df_future)
            return forecast['yhat'].tolist()

        except Exception as e:
            self.logger.error(f"Prophet Error: {e}")
            return [0] * len(X_test)
