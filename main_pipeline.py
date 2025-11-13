#!/usr/bin/env python3
# main_pipeline.py
"""
Pipeline principal del proyecto de Trading Algorítmico.
Integra todos los módulos: Conexión, Limpieza, EDA y Modelos.
"""
from __future__ import annotations
import sys, os

# --- Supresión de Warnings de librerías ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from sklearn.model_selection import ParameterGrid

# Imports de módulos propios
from data.data_loader import DataLoader, DataValidator
from data.data_cleaner import DataCleaner, FeatureEngineer
from utils.metrics import calculate_all_metrics
from models.arima_model import ArimaModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel
from models.random_walk_model import RandomWalkModel
# Agrega aquí otros modelos que crees

from eda.exploratory_analysis import ExploratoryAnalysis


class TradingPipeline:
    """
    Orquestador principal del pipeline de trading
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config, self.config_path = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        
        # Componentes
        self.data_loader: DataLoader | None = None
        self.data_cleaner: DataCleaner | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.eda: ExploratoryAnalysis | None = None
    
    def _load_config(self, config_path: str) -> tuple[Dict[str, Any], str]:
        """Carga configuración desde YAML"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"El archivo de configuración no se encontró en: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuración cargada desde: {config_path}")
        return config, config_path
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        import logging
        
        log_config = self.config.get("logging", {})
        if not log_config.get("enabled", True):
            return
        
        level = getattr(logging, log_config.get("level", "INFO"))
        
        # Formato
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Handlers
        handlers = []
        
        if log_config.get("to_console", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            handlers.append(console_handler)
        
        if log_config.get("to_file", True):
            log_file = Path(log_config.get("file_path", "logs/trading.log"))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            handlers.append(file_handler)
        
        # Configurar logger
        logging.basicConfig(level=level, handlers=handlers)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("="*60)
        self.logger.info("TRADING PIPELINE INICIADO")
        self.logger.info("="*60)
    
    def _setup_directories(self) -> None:
        """Crea estructura de directorios necesaria"""
        dirs = [
            "data/cache",
            "outputs/eda",
            "outputs/models",
            "outputs/backtest",
            "outputs/predictions",
            "logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📁 Directorios de trabajo configurados")
    
    def run(self, mode: str = None) -> None:
        """
        Ejecuta el pipeline según el modo especificado
        
        Args:
            mode: "eda", "train", "backtest", "production"
                 Si es None, usa el modo del config
        """
        mode = mode or self.config.get("execution", {}).get("mode", "eda")
        
        self.logger.info(f"🚀 Ejecutando modo: {mode.upper()}")
        
        if mode == "eda":
            self._run_eda_mode()
        elif mode == "train":
            self._run_train_mode()
        elif mode == "backtest":
            self._run_backtest_mode()
        elif mode == "production":
            self._run_production_mode()
        else:
            raise ValueError(f"Modo no soportado: {mode}")
    
    def _run_eda_mode(self) -> None:
        """
        Modo EDA: Carga → Limpia → Analiza
        Genera reportes estadísticos y gráficos
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ANÁLISIS EXPLORATORIO (EDA)")
        self.logger.info("="*60 + "\n")
        
        # 1. Cargar datos
        df_raw = self._load_data()
        
        # 2. Limpiar datos
        df_clean = self._clean_data(df_raw)
        
        # 3. Generar features (opcional para EDA)
        df_features = self._generate_features(df_clean)
        
        # 4. Análisis exploratorio
        self._perform_eda(df_features)
        
        # 5. Guardar datos en diferentes formatos
        self._save_processed_data(df_features)
        self._save_dataframes_to_excel({
            "Raw Data": df_raw,
            "Cleaned Data": df_clean,
            "Features Data": df_features
        })
        
        self.logger.info("\n✅ MODO EDA COMPLETADO")
    
    def _run_train_mode(self) -> None:
        """
        Modo Train: Entrena modelos y guarda para producción
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: ENTRENAMIENTO DE MODELOS")
        self.logger.info("="*60 + "\n")
        
        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)
        
        # 4. Entrenar modelos
        self.logger.info("🔧 Entrenando modelos...")
        # TODO: Implementar entrenamiento de modelos
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: BACKTESTING")
        self.logger.info("="*60 + "\n")
        
        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)
        
        # 4. Preparar y ejecutar backtest
        self._run_hyperparameter_tuning(df_features)
        
        self.logger.info("\n✅ MODO TRAIN COMPLETADO")

    def _run_backtest_mode(self) -> None:
        """
        Modo Backtest: Evalúa modelos en histórico
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: BACKTESTING")
        self.logger.info("="*60 + "\n")
        
        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)
        
        # 4. Preparar y ejecutar backtest
        self._run_hyperparameter_tuning(df_features)
        
        self.logger.info("\n✅ MODO BACKTEST COMPLETADO")

    def _run_hyperparameter_tuning(self, df_features: pd.DataFrame) -> None:
        """Orquesta el backtesting con búsqueda de hiperparámetros."""
        self.logger.info("튜 PASO 4: INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
        self.logger.info("-" * 60)

        all_results = []
        models_config = self.config.get("models", [])

        for model_config in models_config:
            if not model_config.get("enabled", False):
                continue

            model_name = model_config["name"]
            self.logger.info(f"\n🔥 Procesando modelo: {model_name}")

            # Si se usan params fijos, crear una rejilla de 1
            if "params" in model_config:
                param_grid = model_config["params"]
            else:
                param_grid = model_config.get("param_grid", {})

            grid = ParameterGrid(param_grid)
            model_results = []

            for i, params in enumerate(grid):
                self.logger.info(f"  -> Probando combinación {i+1}/{len(grid)}: {params}")

                # Ejecutar walk-forward para esta combinación
                predictions, true_values = self._run_walk_forward_for_params(
                    df_features, model_name, params
                )

                # Calcular métricas
                if not predictions:
                    self.logger.warning("    No se generaron predicciones, saltando métricas.")
                    continue
                
                metrics = self._calculate_metrics(true_values, predictions)
                self.logger.info(f"    - Métricas: {metrics}")

                # Guardar resultado
                result_row = {"model": model_name, **params, **metrics}
                model_results.append(result_row)
                all_results.append(result_row)

            # Guardar reporte detallado para este modelo
            if model_results:
                self._save_model_report(model_name, model_results)

        # Guardar reporte consolidado
        if all_results:
            self._save_consolidated_summary(all_results)

            # Encontrar los mejores parámetros y guardar config optimizada
            self._find_and_save_best_params(all_results)

    def _find_and_save_best_params(self, all_results: list[dict]):
        """Encuentra los mejores hiperparámetros y guarda el archivo de config optimizado."""
        self.logger.info("\n" + "="*60)
        self.logger.info("🏆 ENCONTRANDO MEJORES HIPERPARÁMETROS")
        self.logger.info("="*60)

        df_results = pd.DataFrame(all_results)
        best_params_per_model = {}

        for model_name in df_results["model"].unique():
            model_df = df_results[df_results["model"] == model_name]
            # Ordenar por RMSE (menor es mejor)
            best_run = model_df.sort_values(by="rmse", ascending=True).iloc[0]
            
            params = {k: v for k, v in best_run.items() if k not in ['model', 'rmse', 'mae', 'hit_rate']}
            best_params_per_model[model_name] = params
            
            self.logger.info(f"  -> Mejor para {model_name}: RMSE={best_run['rmse']:.6f} con params={params}")

        # Crear la nueva configuración optimizada
        new_config = deepcopy(self.config)
        for i, model_conf in enumerate(new_config["models"]):
            model_name = model_conf["name"]
            if model_name in best_params_per_model:
                # Eliminar param_grid y añadir params fijos
                if "param_grid" in new_config["models"][i]:
                    del new_config["models"][i]["param_grid"]
                new_config["models"][i]["params"] = best_params_per_model[model_name]

        # Guardar el nuevo archivo YAML
        optimized_config_path = Path(self.config_path).parent / "config_optimizado.yaml"
        with open(optimized_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)
        self.logger.info(f"\n💾 Configuración optimizada guardada en: {optimized_config_path}")

    def _save_model_report(self, model_name: str, model_results: list[dict]) -> None:
        """Guarda el reporte detallado de un modelo en un archivo CSV."""
        if not model_results:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"report_{model_name}.csv"
        df_report = pd.DataFrame(model_results)
        
        # Ordenar por la métrica principal (RMSE)
        if "rmse" in df_report.columns:
            df_report = df_report.sort_values(by="rmse", ascending=True)
            
        df_report.to_csv(report_path, index=False)
        self.logger.info(f"    💾 Reporte para {model_name} guardado en: {report_path}")

    def _save_consolidated_summary(self, all_results: list[dict]) -> None:
        """Guarda un resumen consolidado de todos los modelos."""
        if not all_results:
            return

        output_dir = Path(self.config.get("output", {}).get("dir", "outputs")) / "backtest"
        summary_path = output_dir / "summary_best_runs.csv"
        
        df_summary = pd.DataFrame(all_results)
        # Agrupar por modelo y obtener la mejor ejecución para cada uno (menor RMSE)
        best_runs = df_summary.loc[df_summary.groupby('model')['rmse'].idxmin()]
        best_runs.to_csv(summary_path, index=False)
        self.logger.info(f"\n📄 Resumen consolidado de mejores ejecuciones guardado en: {summary_path}")

    def _run_walk_forward_for_params(self, df_features: pd.DataFrame, model_name: str, params: dict) -> tuple[list, list]:
        """Ejecuta un backtest Walk-Forward para una configuración de modelo específica."""
        backtest_config = self.config.get("backtest", {})
        initial_train_size = backtest_config.get("initial_train", 800)
        step = backtest_config.get("step", 20)
        target_col = backtest_config.get("target", "Return_1")

        # --- Mejora en el manejo de NaNs ---
        df_processed = df_features.dropna(subset=[target_col])

        features_cols = [col for col in df_features.columns if col != target_col]
        y = df_processed[target_col]
        X = df_processed[features_cols]

        # Validación de datos suficientes
        if initial_train_size >= len(X):
            self.logger.warning(f"    -> No hay suficientes datos para el backtest con initial_train_size={initial_train_size}. "
                                f"Datos disponibles después de limpiar NaNs: {len(X)}. Saltando combinación.")
            return [], []


        all_predictions = []
        all_true_values = []

        for i in range(initial_train_size, len(X), step):
            train_end = i
            test_end = i + 1  # Predecir un paso a la vez

            # --- CORRECCIÓN DATA LEAKAGE ---
            # El rellenado de NaNs debe hacerse DENTRO del bucle sobre el set de entrenamiento
            X_train_raw, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
            X_train = X_train_raw.bfill().ffill().fillna(0)  # Rellenar NaNs solo en train
            y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end] # y_test se usa para las métricas

            prediction = self._train_and_predict(model_name, params, X_train, y_train, X_test)
            
            if prediction is not None:
                all_predictions.extend(prediction)
                all_true_values.extend(y_test.values)

        return all_predictions, all_true_values

    def _train_and_predict(self, model_name: str, params: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> list | None:
        """Punto central para entrenar y predecir con un modelo específico."""
        
        model_map = {
            "RandomWalk": RandomWalkModel,
            "ARIMA": ArimaModel,
            "PROPHET": ProphetModel,
            "LSTM": LSTMModel,
            # "RandomForest": RandomForestModel # Podrías crear este archivo también
        }

        model_class = model_map.get(model_name)
        
        if not model_class:
            self.logger.warning(f"Modelo '{model_name}' no reconocido. Saltando.")
            return None
        
        try:
            self.logger.debug(f"Instanciando modelo {model_name} con params: {params}")
            model_instance = model_class(params=params, logger=self.logger)
            
            return model_instance.train_and_predict(y_train, X_train, X_test)

        except Exception as e:
            self.logger.error(f"Error al ejecutar {model_name}: {e}")
            return None

    def _calculate_metrics(self, y_true: list, y_pred: list) -> dict:
        """Calcula un conjunto de métricas de evaluación."""
        if not y_true or not y_pred:
            self.logger.warning("Listas de valores vacías para calcular métricas.")
            return {"rmse": np.nan, "mae": np.nan, "hit_rate": np.nan}
            
        metrics = calculate_all_metrics(y_true, y_pred)
        return {k: round(v, 6) for k, v in metrics.items()}

    def _run_production_mode(self) -> None:
        """
        Modo Producción: Carga los mejores modelos, obtiene datos en tiempo real
        y genera señales de trading.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("MODO: PRODUCCIÓN")
        self.logger.info("="*60 + "\n")
        
        # 1-3. Cargar, limpiar y features
        df = self._load_data()
        df_clean = self._clean_data(df)
        df_features = self._generate_features(df_clean)

        # 4. Cargar el mejor modelo y predecir
        self.logger.info("🔮 Generando predicción para producción...")
        
        # Identificar el mejor modelo general desde el archivo de config
        # (asumiendo que config_optimizado.yaml está siendo usado)
        best_model_config = self._get_best_model_from_config()
        if not best_model_config:
            self.logger.error("No se pudo determinar el mejor modelo desde la configuración. Ejecute el backtest primero.")
            return

        model_name = best_model_config["name"]
        params = best_model_config.get("params", {})

        self.logger.info(f"Usando el modelo '{model_name}' con parámetros: {params}")

        # Preparar datos: todo el dataset es para "entrenar" y predecir el siguiente paso
        target_col = self.config.get("backtest", {}).get("target", "Return_1")
        features_cols = [col for col in df_features.columns if col != target_col]
        df_processed = df_features.dropna(subset=[target_col] + features_cols)
        
        y_train = df_processed[target_col]
        X_train = df_processed[features_cols]

        # Crear un X_test dummy para el horizonte de predicción
        horizon = self.config.get("backtest", {}).get("horizon", 1)
        X_test_dummy = pd.DataFrame(index=range(horizon))

        prediction = self._train_and_predict(model_name, params, X_train, y_train, X_test_dummy)

        if prediction:
            self.logger.info(f"📈 Predicción generada: {prediction}")
            # Aquí iría la lógica para guardar la predicción o generar una señal
        else:
            self.logger.error("Falló la generación de la predicción.")

        self.logger.info("\n✅ MODO PRODUCCIÓN COMPLETADO")

    def _get_best_model_from_config(self) -> dict | None:
        """Identifica el mejor modelo de la config (el primero con 'params')."""
        for model_config in self.config.get("models", []):
            if model_config.get("enabled", False) and "params" in model_config:
                return model_config
        return None

    # --- MÉTODOS AUXILIARES DEL PIPELINE ---

    def _load_data(self) -> pd.DataFrame:
        """Paso 1: Cargar datos usando DataLoader."""
        self.logger.info("PASO 1: CARGANDO DATOS")
        self.logger.info("-" * 60)
        
        data_config = self.config.get("data", {})
        mt5_config = self.config.get("mt5", {})
        
        self.data_loader = DataLoader(mt5_config=mt5_config)
        df = self.data_loader.load_data(
            symbol=data_config.get("symbol", "EURUSD"),
            timeframe=data_config.get("timeframe", "D1"),
            n_bars=data_config.get("n_bars", 1000),
            use_cache=data_config.get("use_cache", True),
            cache_expiry_hours=data_config.get("cache_expiry_hours", 24)
        )
        
        # Mensajes para indicar de dónde vienen los parámetros
        if "symbol" in data_config:
            self.logger.info(f"  -> Símbolo '{data_config['symbol']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> Símbolo '{df.attrs['symbol']}' (por defecto) usado, no especificado en config/config.yaml")
        if "timeframe" in data_config:
            self.logger.info(f"  -> Timeframe '{data_config['timeframe']}' cargado desde config/config.yaml")
        else:
            self.logger.info(f"  -> Timeframe '{df.attrs['timeframe']}' (por defecto) usado, no especificado en config/config.yaml")
        self.logger.info(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paso 2: Limpiar datos usando DataCleaner."""
        self.logger.info("PASO 2: LIMPIANDO DATOS")
        self.logger.info("-" * 60)
        self.data_cleaner = DataCleaner(self.config.get("data_cleaning", {}))
        df_clean = self.data_cleaner.clean(df)
        self.logger.info(f"✓ Datos limpios: {df_clean.shape[0]} filas restantes.")
        return df_clean

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Paso 3: Generar features usando FeatureEngineer."""
        self.logger.info("PASO 3: GENERANDO FEATURES")
        self.logger.info("-" * 60)
        features_config = self.config.get("features", {})
        df_features = df.copy()

        # 1. Generar retornos
        if features_config.get("returns", {}).get("enabled", False):
            periods = features_config["returns"].get("periods", [1])
            df_features = FeatureEngineer.add_returns(df_features, periods=periods)
            self.logger.info(f"  -> Retornos agregados para períodos: {periods}")

        # 2. Generar indicadores técnicos
        if features_config.get("technical_indicators", {}).get("enabled", False):
            df_features = FeatureEngineer.add_technical_indicators(df_features)
            self.logger.info("  -> Indicadores técnicos agregados.")

        # 3. Generar features rezagados (lags)
        if features_config.get("lag_features", {}).get("enabled", False):
            lag_config = features_config["lag_features"]
            for col in lag_config.get("columns", []):
                if col in df_features.columns:
                    df_features = FeatureEngineer.add_lag_features(df_features, col=col, lags=lag_config.get("lags", []))
                    self.logger.info(f"  -> Lags agregados para la columna: '{col}'")

        self.logger.info(f"✓ Features generadas. Total columnas: {df_features.shape[1]}.")
        return df_features

    def _perform_eda(self, df: pd.DataFrame) -> None:
        """Ejecuta el análisis exploratorio."""
        if not self.config.get("eda", {}).get("enabled", False):
            self.logger.info("-> Análisis Exploratorio (EDA) deshabilitado en config. Saltando.")
            return
            
        self.logger.info("PASO 4: REALIZANDO ANÁLISIS EXPLORATORIO (EDA)")
        self.logger.info("-" * 60)
        self.eda = ExploratoryAnalysis(self.config)
        self.eda.run_analysis(df)
        self.logger.info("✓ Análisis exploratorio completado.")

    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Guarda el dataframe procesado en los formatos especificados."""
        output_config = self.config.get("output", {})
        if not output_config.get("save_predictions", False): return

        output_dir = Path(output_config.get("dir", "outputs"))
        formats = output_config.get("formats", ["csv"])
        
        if "csv" in formats:
            df.to_csv(output_dir / "processed_data.csv")
            self.logger.info(f"💾 Datos procesados guardados en: {output_dir / 'processed_data.csv'}")

    def _save_dataframes_to_excel(self, dataframes: dict[str, pd.DataFrame]):
        """Guarda múltiples dataframes en un solo archivo Excel."""
        output_config = self.config.get("output", {})
        if "excel" not in output_config.get("formats", []): return

        output_dir = Path(output_config.get("dir", "outputs"))
        excel_path = output_dir / "trading_data_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=True)
        self.logger.info(f"💾 Reporte de datos guardado en: {excel_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Trading Algorítmico.")
    parser.add_argument("--mode", type=str, default="eda", choices=["eda", "train", "backtest", "production"],
                        help="Modo de ejecución del pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuración YAML.")
    args = parser.parse_args()
    
    pipeline = TradingPipeline(config_path=args.config)
    pipeline.run(mode=args.mode)
