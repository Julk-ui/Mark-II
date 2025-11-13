#!/usr/bin/env python3
# models/prophet_model.py
"""
Implementación del modelo Prophet para predicción de series temporales.
Incluye manejo de regresores externos con rezago para evitar data leakage.
"""
from __future__ import annotations
import pandas as pd
from prophet import Prophet
from .base_model import BaseModel

class ProphetModel(BaseModel):
    """Modelo Prophet para predicción de series temporales."""

    def train_and_predict(self, y_train: pd.Series, X_train: pd.DataFrame | None = None, X_test: pd.DataFrame | None = None) -> list:
        """
        Entrena un modelo Prophet y predice.

        Maneja los regresores de dos maneras para evitar data leakage:
        1. Si `use_lagged_regressors` es True: Usa los valores de los regresores del último día conocido (T-1)
           para predecir el día T. Esto simula un entorno real.
        2. Si es False (comportamiento por defecto y erróneo): Usa los regresores de X_test, causando data leakage.
        """
        try:
            # 1. Preparar el DataFrame para Prophet
            df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

            # Parámetros del modelo
            prophet_params = {k: v for k, v in self.params.items() if k != 'use_lagged_regressors'}
            use_lagged_regressors = self.params.get('use_lagged_regressors', False)

            self.model = Prophet(**prophet_params)

            # 2. Añadir regresores (features)
            regressors = []
            if X_train is not None and not X_train.empty:
                regressors = list(X_train.columns)
                # --- CORRECCIÓN ---
                # Se une X_train a df_train usando la columna 'ds' de df_train
                # y el índice (que son fechas) de X_train. Esto alinea
                # correctamente los datos.
                df_train = df_train.set_index('ds').join(X_train).reset_index()
                for regressor in regressors:
                    self.model.add_regressor(regressor)

            #print(df_train.head())  # Debug: Verificar el DataFrame de entrenamiento
            # 3. Entrenar el modelo
            self.model.fit(df_train)

            # 4. Crear el DataFrame futuro para la predicción
            horizon = len(X_test) if X_test is not None else 1
            future = self.model.make_future_dataframe(periods=horizon)

            # 5. Llenar el DataFrame futuro con los valores de los regresores
            if regressors:
                if use_lagged_regressors:
                    # --- LÓGICA CORRECTA: Evita Data Leakage ---
                    # Usamos el último valor conocido de los regresores (de X_train)
                    # para predecir el siguiente paso.
                    if not X_train.empty:
                        last_known_regressors = X_train.iloc[-1:]
                        for regressor in regressors:
                            # Asigna el último valor conocido a todas las filas del dataframe 'future'
                            future[regressor] = last_known_regressors[regressor].values[0]
                    else:
                        self.logger.warning("Prophet: X_train está vacío, no se pueden usar regresores rezagados.")

                else:
                    # --- LÓGICA INCORRECTA: Causa Data Leakage ---
                    # Esto usa los valores futuros de los regresores, lo que lleva a resultados perfectos.
                    self.logger.warning("Prophet: 'use_lagged_regressors' es False. ¡ALERTA DE DATA LEAKAGE!")
                    if X_test is not None:
                        # Se asegura que el índice coincida para la unión
                        future.set_index('ds', inplace=True)
                        future = future.join(X_test, how='left')
                        future.reset_index(inplace=True)
                        future.ffill(inplace=True) # Rellenar por si acaso

            # 6. Predecir
            forecast = self.model.predict(future)

            # Devolver solo la predicción para el horizonte deseado
            prediction = forecast['yhat'].iloc[-horizon:].tolist()
            return prediction

        except Exception as e:
            self.logger.error(f"Prophet Error: {e}")
            return [0] * (len(X_test) if X_test is not None else 1)