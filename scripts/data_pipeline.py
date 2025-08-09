### Data pipeline. Elaborado por Mario Paulín como parte del proyecto agritech  
### Este script se utiliza para 
# 1) cargar el dataset desde kaggle
# 2) Mantener solo los datos de los cultivos de maíz
# 3) Agregar duplicados de temperatura promedio para el mismo país y año utilizando la media
# 4) Guardar el dataset limpio en un archivo para generar feature engineering. 

### Nota respecto a missings e imputacion: Basado en el analisis exploratorio de datos (EDA.py), existen valores faltantes en el sentido en que 
### hay paises que no reportan datos para ciertos años. Por la estructura de los faltantes en que el pais puede comenzar a tener data a partir de cierto año 
### por esta razon, que no son años aleatorios sin data o data faltante solo en alguna variable, me parece que la mejor estrategia es no hacer imputacion.

### Para hacer tracking de los cambios en el dataset, se utiliza DVC (Data Version Control)
#
# Recomendación de uso:
# 1. Inicializa DVC en tu proyecto (solo una vez):
#    dvc init
# 2. Versiona el archivo original descargado:
#    dvc add data/yield_df.csv
# 3. Versiona el archivo limpio generado:
#    dvc add data/maize_cleaned.csv
# 4. Opcional: Crea un pipeline reproducible con DVC:
#    dvc run -n clean_maize_data \
#      -d scripts/data_pipeline.py -d data/yield_df.csv \
#      -o data/maize_cleaned.csv \
#      -p <parametros_si_los_hay> \
#      python scripts/data_pipeline.py
# 5. Haz commit de los archivos .dvc y dvc.yaml a git.
# 6. Para reproducir el pipeline en otro entorno:
#    dvc pull  # descarga los datos versionados
#    dvc repro # ejecuta el pipeline completo

### Iniciamos importando las librerías necesarias

import numpy as np 
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
# import dvc 
import logging
import io 

### Funcion para configurar el logger 
def setup_logger():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/data_pipeline.log"),
            logging.StreamHandler()
        ]
    )

### 1) Funcion para descargar datos desde Kaggle y hacer unzip. 
def download_data():
    os.makedirs("data", exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files("patelris/crop-yield-prediction-dataset", path="data", unzip=True)
    except Exception as e:
        logging.error(f"Error al descargar datos: {e}")

### Cargamos el csv a pandas 
def load_data(path):
    ### Descriptivos basicos del dataset
    data = pd.read_csv(path)
    buffer = io.StringIO()
    print("Info:")
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    logging.info("Info del DataFrame:\n" + info_str)
    print("Descripcion:")
    describe_str = data.describe().to_string()
    logging.info("Descripción estadística:\n" + describe_str)
    
    ## El dataset se ha cargado correctamente. Tiene 28242 filas y 8 columnas. Las columnas son non-null pero vamos a verficar. 
    # Revision de valores faltantes 
    missing_summary = pd.DataFrame({
        'Missing_Count': data.isnull().sum(),
        'Missing_Percentage': (data.isnull().sum() / len(data)) * 100
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    logging.info("Resumen de valores faltantes:")
    if missing_summary.empty:
        logging.info("No hay valores faltantes en el dataset.")
    else:
        logging.info(missing_summary)
    
    # Revisamos si hay duplicados
    duplicate_count = data.duplicated().sum()
    logging.info(f"Duplicados encontrados: {duplicate_count}")

    # Tipo de datos de cada columna
    logging.info("Resumen de tipos de datos:")
    logging.info(data.dtypes.value_counts())

    return data

### 2) Filtramos los datos de cultivos para quedarnos solo con maiz
def filter_maize(data):
    return data[data['Item'] == 'Maize']

### 3) Agrega duplicados de temperatura promedio para el mismo país y año utilizando la mediana. Esto lo documente en 
### El archivo de EDA. Basado en los resultados de analisis exploratorio, la mediana y la media son similares, sin embargo 
### preferi utilizar la mediana para hacer la medida mas resistente a outliers en caso de que las repeticiones se deban a errores de medicion o calidad de datos 
def aggregate_maize(maize_data):
    maize_agg = maize_data.groupby(['Area', 'Year'], as_index=False).agg({
        'hg/ha_yield': 'mean',
        'average_rain_fall_mm_per_year': 'mean',
        'pesticides_tonnes': 'mean',
        'avg_temp': ['median']
    })
    maize_agg.columns = ['Area', 'Year', 'yield_mean', 'rain_fall_mean', 'pesticides_mean', 'avg_temp_median']
    return maize_agg

###  4) Guardar el dataset limpio en un archivo para generar feature engineering.  
def save_cleaned(maize_agg, path):
    maize_agg.to_csv(path, index=False)

if __name__ == "__main__":
    setup_logger()
    logging.info("Iniciando pipeline de datos...")
    download_data()
    data = load_data("data/yield_df.csv")
    logging.info(f"Dataset cargado: {data.shape}")

    maize_data = filter_maize(data)
    logging.info(f"Filtrado maíz: {maize_data.shape}")
    maize_agg = aggregate_maize(maize_data)
    logging.info(f"Agregado: {maize_agg.shape}")
    save_cleaned(maize_agg, "data/maize_cleaned.csv")
    logging.info("Pipeline finalizado.")
