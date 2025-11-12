#!/usr/bin/env python3
# models/random_walk_model.py
"""
Implementación del modelo Random Walk (línea base).
"""
from __future__ import annotations
import pandas as pd
from .base_model import BaseModel

class RandomWalkModel(BaseModel):
    """La predicción es el último valor conocido."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """La predicción para el siguiente paso es el último valor de y_train."""
        if y_train.empty:
            self.logger.warning("RW Model: y_train está vacío. Prediciendo 0.")
            return [0] * len(X_test)
        
        last_known_value = y_train.iloc[-1]
        return [last_known_value] * len(X_test)
