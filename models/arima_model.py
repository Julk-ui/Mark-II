#!/usr/bin/env python3
# models/arima_model.py
"""
Implementación del modelo ARIMA.
"""
from __future__ import annotations
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from .base_model import BaseModel

class ArimaModel(BaseModel):
    """Modelo ARIMA para predicción de series temporales."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """Entrena un modelo ARIMA y predice."""
        try:
            order = (
                self.params.get("p", 1),
                self.params.get("d", 0),
                self.params.get("q", 0)
            )
            
            # ARIMA en statsmodels es univariado, ignora X_train.
            model = ARIMA(y_train, order=order)
            model_fit = model.fit()
            
            prediction = model_fit.forecast(steps=len(X_test))
            return prediction.tolist()
        except Exception as e:
            self.logger.error(f"ARIMA Error: {e}")
            return [0] * len(X_test) # Fallback a 0 si hay error