#!/usr/bin/env python3
# models/base_model.py
"""
Define la interfaz base para todos los modelos de predicción.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    """Clase abstracta para los modelos."""

    def __init__(self, params: dict, logger):
        self.params = params
        self.logger = logger
        self.model = None

    @abstractmethod
    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None, X_test: pd.DataFrame | None) -> list:
        """Entrena el modelo y devuelve las predicciones."""
        pass
