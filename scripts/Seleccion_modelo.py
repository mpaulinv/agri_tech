
### Desarrollo y selección del modelo predictivo de rendimiento de maiz
### Este script realiza:
# 1. Carga de los datos procesados con feature engineering
# 2. Split en train y test (test = 2009-2013, datos mas recientes)
# 3. Definición de folds para validación cruzada temporal
# 4. Entrenamiento y validación de modelos:
#    - Regresión lineal (full y subset de variables)
#    - Modelos de efectos aleatorios 
#    - Modelos de efectos fijos (dummies por país)
#    - Random Forest (con y sin variables logarítmicas)
#    - Tuneo de hiperparámetros para Random Forest
# 5. Tracking de experimentos y artefactos con MLflow
# 6. Cálculo y reporte de métricas (RMSE, MAE, R2) por fold y globales
# 7. Visualización de resultados por país y año (2008 y 2009-2013)
# 8. Comparación visual y numérica de modelos para interpretación y selección final



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
from utils import run_linear_regression_cv, run_mixedlm_cv, run_fixed_effects_cv, run_random_forest_cv


historicos = pd.read_csv("data/feature_engineered_data.csv")
print(historicos.head())
print(historicos.info)
print(historicos.columns)

### Tenemos 2028 filas. 
### Estrategia de validacion. Guardamos los ultimos años como parte del test set, en este caso 
### los ultimos 5 años (2009-2013)
train = historicos[historicos['Year'] <= 2008]
test = historicos[historicos['Year'] > 2008]    

#### Comenzamos partiendo los datos en 5 folds para validacion cruzada temporal.
def temporal_cv_folds(df: pd.DataFrame, year_col: str = 'Year', n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera índices de train/val para validación cruzada temporal en panel de años.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        year_col (str): Nombre de la columna de año.
        n_folds (int): Número de folds.

    Returns:
        list of tuples: Cada tupla contiene (train_idx, val_idx) para cada fold.
    """
    years = sorted(df[year_col].unique())
    fold_size = len(years) // n_folds
    folds = []
    for i in range(n_folds):
        # Selecciona los años de validación para este fold
        if i < n_folds - 1:
            val_years = years[i*fold_size:(i+1)*fold_size]
        else:
            val_years = years[i*fold_size:]
        # Los años de entrenamiento son todos los demás
        train_years = [y for y in years if y not in val_years]
        # Índices de entrenamiento y validación
        train_idx = df[df[year_col].isin(train_years)].index.values
        val_idx = df[df[year_col].isin(val_years)].index.values
        folds.append((train_idx, val_idx))
    return folds

# Crear folds
folds = temporal_cv_folds(train, year_col='Year', n_folds=5)

# Mostrar los años de validación de cada fold
for i, (train_idx, val_idx) in enumerate(folds):
    val_years = sorted(train.loc[val_idx, 'Year'].unique())
    print(f"Fold {i+1} - Años de validación: {val_years}")

### Empezamos con el modelo mas simple. Incluimos las variables que escogimos en EDA
### Y lo incluimos en una regresion lineal. Para validar el modelo usamos RMSE, R2 y MAE.
### Trackeamos los experimentos con MLflow

features_full = [
    'rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
    'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
    'pesticides_mean_boxcox_square', 'pesticides_mean_boxcox_cube',
    'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
    'rain_fall_mean_boxcox_times_pesticides_mean_boxcox',
    'rain_fall_mean_boxcox_over_avg_temp_median_boxcox',
    'pesticides_mean_boxcox_over_avg_temp_median_boxcox',
    'rain_fall_mean_cambio_logaritmico',
    'pesticides_mean_cambio_logaritmico',
    'avg_temp_median_cambio_logaritmico'
]

run_linear_regression_cv(
    train_df=train,
    folds=folds,
    features=features_full,
    experiment_name="Yield_Prediction_Linear_full",
    run_name="linear_regression_cv",
    model_type="LinearRegression",
    tag_experiment="linear_regression_cv",
    data_version="v1.0",
    results_csv="cv_results.csv"
)


# Las interacciones pesticida_quadrado y pesticida_cubo no son significativas. Ejecuto el modelo sin ellas.

features_subset = ['rain_fall_mean_boxcox',
     'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
    'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
    'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
    'rain_fall_mean_boxcox_times_pesticides_mean_boxcox',
    'rain_fall_mean_boxcox_over_avg_temp_median_boxcox',
    'pesticides_mean_boxcox_over_avg_temp_median_boxcox',
    'rain_fall_mean_cambio_logaritmico',
    'pesticides_mean_cambio_logaritmico',
    'avg_temp_median_cambio_logaritmico'
]
run_linear_regression_cv(
    train_df=train,
    folds=folds,
    features=features_subset,
    experiment_name="Yield_Prediction_Linear_subset",
    run_name="linear_regression_cv",
    model_type="LinearRegression",
    tag_experiment="linear_regression_cv",
    data_version="v1.0",
    results_csv="cv_results_subset.csv"
)

### Incluimos un modelo de effectos aleatorios para capturar la variabilidad entre paises
run_mixedlm_cv(
    train_df=train,
    folds=folds,
    features=features_subset,
    group_col="Area",
    experiment_name="Yield_Prediction_Linear_random_effects_subset",
    run_name="linear_regression_cv",
    model_type="MixedLM_Area",
    tag_experiment="mixedlm_random_effects_cv",
    data_version="v1.0",
    results_csv="cv_results_mixedlm.csv"
)


# Modelo de efectos fijos (dummies para Area). Aqui debo eliminar rainfall ya que es constante por pais/año
features_fixed = [ 'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
        'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
        'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
        'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 
        'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 
        'pesticides_mean_boxcox_over_avg_temp_median_boxcox']

run_fixed_effects_cv(
    train_df=train,
    folds=folds,
    features_base=features_fixed,
    group_col="Area",
    experiment_name="Yield_Prediction_Linear_fixed_effects_subset",
    run_name="linear_regression_cv",
    model_type="FixedEffects_Area",
    tag_experiment="mixedlm_fixed_effects_cv",
    data_version="v1.0",
    results_csv="cv_results_fixed_effects.csv"
)

### Ahora probamos un modelo de Random Forest, que puede capturar relaciones no lineales y interacciones
### Las variables de cambio logarimico deberian mejorar el rendimiento en el random forest 
run_random_forest_cv(
    train_df=train,
    folds=folds,
    features=features_subset,
    experiment_name="Yield_Prediction_RandomForest_subset",
    run_name="random_forest_cv",
    model_type="RandomForestRegressor",
    tag_experiment="random_forest_cv",
    data_version="v1.0",
    results_csv="cv_results_rf.csv",
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

### Probamos sin las variables de cambio logaritmico.
features_no_log_change = ['rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
        'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
        'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
        'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 
        'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 
        'pesticides_mean_boxcox_over_avg_temp_median_boxcox']

run_random_forest_cv(
    train_df=train,
    folds=folds,
    features=features_no_log_change,
    experiment_name="Yield_Prediction_RandomForest_no_log_change",
    run_name="random_forest_cv_no_log_change",
    model_type="RandomForestRegressor_no_log_change",
    tag_experiment="random_forest_cv_no_log_change",
    data_version="v1.0",
    results_csv="cv_results_rf_no_log_change.csv",
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

## Nota: Las variables de cambio con respecto a historicos no aportan mucho en terminos de las metricas. Esto puede deberse a que 
## la variable de lluvia no presenta variabilidad dentro de cada pais en el tiempo y por tanto no aporta informacion adicional.
## El calculo de estas variables puede introducir ruido, sobretodo al introducir nuevos paises en la data. Por tanto prefiero simplificar el modelo. 

### Experimento: Random Forest con Area como one-hot encoding (dummies)

# Features: originales + dummies de Area (ya presentes en el dataset)
area_dummy_cols = [col for col in train.columns if col.startswith('Area_')]
features_rf_area = features_no_log_change + area_dummy_cols
print("Features used for Random Forest (including Area dummies):", features_rf_area)
run_random_forest_cv(
    train_df=train,
    folds=folds,
    features=features_rf_area,
    experiment_name="Yield_Prediction_RandomForest_AreaDummies",
    run_name="random_forest_cv_areadummies",
    model_type="RandomForestRegressor_AreaDummies",
    tag_experiment="random_forest_cv_areadummies",
    data_version="v1.0",
    results_csv="cv_results_rf_areadummies.csv",
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Tuneo de hiperparámetros usando folds temporales (Random Search manual)

mlflow.set_experiment("Yield_Prediction_RandomForest_temporal_tuning_AreaDummies")
with mlflow.start_run(run_name="random_forest_temporal_tuning_cv_areadummies"):
    features = features_rf_area
    param_grid = {
        'n_estimators': [500, 550, 600, 650, 700],
        'max_depth': [20, 25, None],
        'min_samples_split': [ 2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'max_features': [2, 3, 4, 5, 6]
    }
    all_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['max_features']
    ))
    random.seed(42)
    sampled_combinations = random.sample(all_combinations, min(60, len(all_combinations)))

    best_score = float('inf')
    best_params = None
    all_results = []
    for params in sampled_combinations:
        param_dict = {
            'n_estimators': params[0],
            'max_depth': params[1],
            'min_samples_split': params[2],
            'min_samples_leaf': params[3],
            'max_features': params[4],
            'random_state': 42,
            'n_jobs': -1
        }
        fold_mae = []
        for train_idx, val_idx in folds:
            train_fold = train.loc[train_idx]
            val_fold = train.loc[val_idx]
            train_fold = train_fold.dropna(subset=features + ['yield_mean'])
            val_fold = val_fold.dropna(subset=features + ['yield_mean'])
            X_train, y_train = train_fold[features], train_fold['yield_mean']
            X_val, y_val = val_fold[features], val_fold['yield_mean']
            rf = RandomForestRegressor(**param_dict)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            fold_mae.append(mae)
        mean_mae = np.mean(fold_mae)
        all_results.append({**param_dict, 'mean_cv_mae': mean_mae})
        if mean_mae < best_score:
            best_score = mean_mae
            best_params = param_dict.copy()

    print("Mejores hiperparámetros (temporal CV, AreaDummies):", best_params)
    mlflow.log_params(best_params)

    # Validación cruzada temporal con el mejor modelo
    rmse_scores, mae_scores, r2_scores = [], [], []
    results = []
    for i, (train_idx, val_idx) in enumerate(folds):
        train_fold = train.loc[train_idx]
        val_fold = train.loc[val_idx]
        train_fold = train_fold.dropna(subset=features + ['yield_mean'])
        val_fold = val_fold.dropna(subset=features + ['yield_mean'])
        X_train, y_train = train_fold[features], train_fold['yield_mean']
        X_val, y_val = val_fold[features], val_fold['yield_mean']
        rf_best = RandomForestRegressor(**best_params)
        rf_best.fit(X_train, y_train)
        y_pred = rf_best.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        row = {'fold': i+1, 'rmse': rmse, 'mae': mae, 'r2': r2}
        for f, imp in zip(features, rf_best.feature_importances_):
            row[f'featimp_{f}'] = imp
        results.append(row)
        print(f"Fold {i+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}")

    # Log model, parameters, and tags for versioning
    mlflow.sklearn.log_model(rf_best, "model")
    mlflow.log_param("features", features)
    mlflow.log_param("model_type", "RandomForestRegressor_temporal_tuned_AreaDummies")
    mlflow.set_tag("experiment", "random_forest_temporal_tuning_cv_areadummies")
    mlflow.set_tag("data_version", "v1.0")

    results_df = pd.DataFrame(results)
    print("\nTabla de resultados por fold (con importancias de variables):")
    print(results_df)

    # Guardar la tabla como artefacto en MLflow
    results_df.to_csv("cv_results_rf_temporal_tuned_areadummies.csv", index=False)
    mlflow.log_artifact("cv_results_rf_temporal_tuned_areadummies.csv")

    # Log de métricas promedio en MLflow
    mlflow.log_metric("cv_rmse_mean", np.mean(rmse_scores))
    mlflow.log_metric("cv_mae_mean", np.mean(mae_scores))
    mlflow.log_metric("cv_r2_mean", np.mean(r2_scores))


### El random forest tuneado tiene mejores resultados en las tres metricas, comparado con el modelo de efectos fijos. 
### si la interpretabilidad es clave eso puede ser una ventaja para el modelo de efectos fijos 
