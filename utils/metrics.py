#!/usr/bin/env python3
# utils/metrics.py
"""
Módulo para calcular métricas de evaluación para modelos de trading.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula el Mean Absolute Error (MAE)."""
    return mean_absolute_error(y_true, y_pred)

def calculate_hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Hit Rate.
    Es el porcentaje de veces que el signo de la predicción fue correcto.
    """
    # np.sign(0) es 0. Para evitar que una predicción de 0 coincida con un valor real de 0
    # y se cuente como un acierto incorrecto, tratamos los signos.
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return np.mean(true_sign == pred_sign) * 100

def calculate_all_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> dict[str, float]:
    """
    Calcula un conjunto de métricas de evaluación y las devuelve en un diccionario.

    Args:
        y_true: Valores verdaderos.
        y_pred: Valores predichos por el modelo.

    Returns:
        Un diccionario con las métricas calculadas.
    """
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError("Los arrays de y_true y y_pred deben tener la misma forma.")

    return {
        "rmse": calculate_rmse(y_true_np, y_pred_np),
        "mae": calculate_mae(y_true_np, y_pred_np),
        "hit_rate": calculate_hit_rate(y_true_np, y_pred_np)
    }