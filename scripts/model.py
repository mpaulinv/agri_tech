### Script para entrenar el modelo en toda la data para la api 

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm
import mlflow 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from itertools import product
import random
import shap
import joblib

os.makedirs('model', exist_ok=True)


historicos = pd.read_csv("data/feature_engineered_data.csv")
print(historicos.head())
print(historicos.info)
print(historicos.columns)

### Tenemos 2028 filas. 
### Entrenamos el modelo con todo el dataset.

features_subset = ['rain_fall_mean_boxcox',
     'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
    'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
    'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
    'rain_fall_mean_boxcox_times_pesticides_mean_boxcox',
    'rain_fall_mean_boxcox_over_avg_temp_median_boxcox',
    'pesticides_mean_boxcox_over_avg_temp_median_boxcox'
    ]

area_dummy_cols = [col for col in historicos.columns if col.startswith('Area_')]
features_rf_area = features_subset + area_dummy_cols

# Random Forest final con hiperparámetros del tunning 
X_train_rf = historicos[features_rf_area]
y_train_rf = historicos['yield_mean']
rf_final = RandomForestRegressor(n_estimators=550, max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, max_features=2, random_state=42, n_jobs=-1)
rf_final.fit(X_train_rf, y_train_rf)

# Guardar el modelo entrenado
joblib.dump(rf_final, "model/model_rf_final.joblib")
