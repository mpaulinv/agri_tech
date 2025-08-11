### Tests. Elaborado por Mario Paulín como parte del proyecto agritech  
### Este script se utiliza para ejecutar pruebas unitarias en el modelo de predicción de rendimiento. 


import pandas as pd
import os
import requests

# test de carga de datos limpios con colunas esperadas
def test_load_clean_data():
    """Test de carga de datos limpios"""
    assert os.path.exists("data/maize_cleaned.csv")
    df = pd.read_csv("data/maize_cleaned.csv")
    assert not df.empty
    expected_cols = ['Area', 'Year', 'yield_mean', 'rain_fall_mean', 'pesticides_mean', 'avg_temp_median']
    for col in expected_cols:
        assert col in df.columns

# test de feature engineering
def test_feature_engineering_output():
    """Test de salida de feature engineering"""
    assert os.path.exists("data/feature_engineered_data.csv")
    df = pd.read_csv("data/feature_engineered_data.csv")
    transformed_cols = ['rain_fall_mean_boxcox',
     'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
    'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
    'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
    'rain_fall_mean_boxcox_times_pesticides_mean_boxcox',
    'rain_fall_mean_boxcox_over_avg_temp_median_boxcox',
    'pesticides_mean_boxcox_over_avg_temp_median_boxcox'
    ]

    for col in transformed_cols:
        assert col in df.columns

# test de generación de modelo
def test_model_file_exists():
    """Test de existencia del modelo entrenado"""
    assert os.path.exists("model/model_rf_final.joblib")

