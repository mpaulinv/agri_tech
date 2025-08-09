### Pipeline para feature engineering 
### 1) carga el dataset limpio 
### 2) Ejecutamos las siguientes transformaciones:
###    a) Transformaciones contemporaneas: 
###       BoxCox
###       Logaritmo
###       Normalizacion (estandarizacion)
###       Cuadrado de la variable normalizada 
###       Cubo de la variable normalizada
###    b) Contempraneas polinomios:
###       Cuadrado de la variable
###       Cubo de la variable
###    c) Interacciones entre variables
###       Razon 
###       Producto      
###    d) Cambios respecto al historico o promedio: 
###       Cambio porcentual 
###       Cambio logaritmico (logaritmo natural de la razon)
###    e) One-hot encoding para la variable 'area'
### Nota: Basado en el analisis EDA_feature_engineering.py y EDA_features.py, he selecionado solo realizar interacciones para las variables transformadas con BoxCox.


## Cargamos las librerias necesarias
import pandas as pd
import os
import logging
import io 
from utils import transformaciones_contemporaneas, interacciones_contemporaneas, calcular_historico, cambio_porcentual, cambio_logaritmo, contemporaneas_polinomios,dummies_area



### Funcion para configurar el logger 
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/feature_engineering_pipeline.log"),
            logging.StreamHandler()
        ]
    )


### Funciones de carga, transformacion y guardado de datos: 
def load_clean_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def apply_feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Definimos las variables que se van a transformar, estas son las variables continuas fuera del target
    continuous_columns = ['rain_fall_mean', 'pesticides_mean', 'avg_temp_median']
    # Transformaciones contemporaneas
    feature_engineered = transformaciones_contemporaneas(data, continuous_columns)
    logging.info(f"Después de transformaciones_contemporaneas: columnas={list(feature_engineered.columns)}, shape={feature_engineered.shape}, nulos={feature_engineered.isnull().sum().sum()}")
    # Aplicamos polinomios a las transformaciones box-cox contemporaneas
    boxcox_columns = ['rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox']
    feature_engineered = contemporaneas_polinomios(feature_engineered, boxcox_columns)
    logging.info(f"Después de contemporaneas_polinomios: columnas={list(feature_engineered.columns)}, shape={feature_engineered.shape}, nulos={feature_engineered.isnull().sum().sum()}")
    # Interacciones contemporaneas entre las variables transformadas
    interaction_columns = [
        'rain_fall_mean_boxcox', 'pesticides_mean_boxcox',
        'avg_temp_median_boxcox'
    ]
    feature_engineered = interacciones_contemporaneas(feature_engineered, interaction_columns)
    logging.info(f"Después de interacciones_contemporaneas: columnas={list(feature_engineered.columns)}, shape={feature_engineered.shape}, nulos={feature_engineered.isnull().sum().sum()}")
    # One-hot encoding para Area
    feature_engineered = dummies_area(feature_engineered)
    logging.info(f"Después de dummies_area: columnas={list(feature_engineered.columns)}, shape={feature_engineered.shape}, nulos={feature_engineered.isnull().sum().sum()}")
    return feature_engineered

def add_historical_features(data: pd.DataFrame, continuous_columns: list) -> pd.DataFrame:
    # Cambios respecto al historico o promedio
    # Esta funcion calcula el promedio historico por area y año para las variables continuas especificadas 
    # Si no hay datos para años anteriores para el pais, se usa el promedio global del año. Para el primer año de datos, el valor es NaN. 
    historicos = calcular_historico(data, continuous_columns)
    logging.info(f"Después de calcular_historico: columnas={list(historicos.columns)}, shape={historicos.shape}, nulos={historicos.isnull().sum().sum()}")
    for column in continuous_columns:
        # calculamos el cambio porcentual respecto a los historicos
        cambio_porcentual(historicos, column, f'{column}_hist_mean')
    logging.info(f"Después de cambio_porcentual: columnas={list(historicos.columns)}, shape={historicos.shape}, nulos={historicos.isnull().sum().sum()}")
    for column in continuous_columns:
        # calculamos el cambio logaritmico respecto a los historicos
        cambio_logaritmo(historicos, column, f'{column}_hist_mean')
    logging.info(f"Después de cambio_logaritmo: columnas={list(historicos.columns)}, shape={historicos.shape}, nulos={historicos.isnull().sum().sum()}")
    return historicos

def save_featured_data(data: pd.DataFrame, path: str) -> None:
    data.to_csv(path, index=False)


# Ejecutamos el pipeline
def main():
    setup_logger()
    continuous_columns = ['rain_fall_mean', 'pesticides_mean', 'avg_temp_median']
    cleaned_data = load_clean_data("data/maize_cleaned.csv")
    logging.info(f"Datos limpios cargados: columnas={list(cleaned_data.columns)}, shape={cleaned_data.shape}, nulos={cleaned_data.isnull().sum().sum()}")
    feature_engineered_data = apply_feature_engineering(cleaned_data)
    historicos = add_historical_features(feature_engineered_data, continuous_columns)
    save_featured_data(historicos, "data/feature_engineered_data.csv")
    logging.info("Pipeline de feature engineering finalizado.")

if __name__ == "__main__":
    main()

