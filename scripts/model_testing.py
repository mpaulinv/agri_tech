### Script para entrenar en el periodo de entrenamiento y probar el modelo en el set de prueba. 
### Añade interpretabilidad y métricas de rendimiento año por año 



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

os.makedirs('results', exist_ok=True)


historicos = pd.read_csv("data/feature_engineered_data.csv")
print(historicos.head())
print(historicos.info)
print(historicos.columns)

### Tenemos 2028 filas. 
### Estrategia de validacion. Guardamos los ultimos años como parte del test set, en este caso 
### los ultimos 5 años (2009-2013)
train = historicos[historicos['Year'] <= 2008]
test = historicos[historicos['Year'] > 2008]    


### Validamos el modelo final en el test de prueba

features_subset = ['rain_fall_mean_boxcox',
     'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
    'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
    'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube',
    'rain_fall_mean_boxcox_times_pesticides_mean_boxcox',
    'rain_fall_mean_boxcox_over_avg_temp_median_boxcox',
    'pesticides_mean_boxcox_over_avg_temp_median_boxcox'
    ]

area_dummy_cols = [col for col in train.columns if col.startswith('Area_')]
features_rf_area = features_subset + area_dummy_cols

# Random Forest final con hiperparámetros del tunning 
X_train_rf = train[features_rf_area]
y_train_rf = train['yield_mean']
X_test_rf = test[features_rf_area]
rf_final = RandomForestRegressor(n_estimators=550, max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, max_features=2, random_state=42, n_jobs=-1)
rf_final.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_final.predict(X_test_rf)

# Métricas globales en test (2009-2013)
y_true_test = test['yield_mean'].values
y_pred_rf_test = y_pred_rf[:len(y_true_test)]

rmse_rf = root_mean_squared_error(y_true_test, y_pred_rf_test)
mae_rf = mean_absolute_error(y_true_test, y_pred_rf_test)
r2_rf = r2_score(y_true_test, y_pred_rf_test)

print("\nMétricas globales en test (2009-2013):")
print(f"Random Forest (Area dummies, tuned) - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R2: {r2_rf:.3f}")
import os
os.makedirs('results', exist_ok=True)
importances = rf_final.feature_importances_
feature_names = features_rf_area
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df, palette='viridis')
plt.title('Top 10 Feature Importances - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/top10_feature_importances_rf.png')
plt.show()


# SHAP summary plot para las top 10 features
print("\nCalculando valores SHAP para interpretabilidad global...")
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test_rf)
# Seleccionar las top 10 features por importancia media absoluta SHAP
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_shap_idx = np.argsort(mean_abs_shap)[::-1][:10]
top_shap_features = [feature_names[i] for i in top_shap_idx]
print("Top 10 features por importancia SHAP:")
for f in top_shap_features:
    print(f)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_rf, feature_names=feature_names, plot_type="bar", max_display=10)
plt.title('Top 10 SHAP Feature Importances - Random Forest')
plt.tight_layout()
plt.savefig('results/top10_shap_feature_importances_rf.png')
plt.show()

# Importancia por permutación
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
perm_result = permutation_importance(rf_final, X_test_rf, y_true_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp_df = pd.DataFrame({'feature': feature_names, 'importance': perm_result.importances_mean})
perm_imp_df = perm_imp_df.sort_values('importance', ascending=False).head(10)
print("\nTop 10 features por importancia de permutación:")
print(perm_imp_df)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=perm_imp_df, palette='mako')
plt.title('Top 10 Permutation Importances - Random Forest')
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('results/top10_permutation_importances_rf.png')
plt.show()

# PDP para las top 10 de permutación
print("\nGraficando PDP para las top 10 features de permutación...")
top_perm_features = perm_imp_df['feature'].tolist()
fig, ax = plt.subplots(figsize=(16, 18))
PartialDependenceDisplay.from_estimator(rf_final, X_test_rf, top_perm_features, ax=ax)
plt.tight_layout()
plt.savefig('results/top10_pdp_rf.png')
plt.show()

# Graficar por año
for year in sorted(test['Year'].unique()):
    df_plot = test[test['Year'] == year][['Area', 'yield_mean']].copy()
    idxs = test['Year'] == year
    df_plot['RF'] = y_pred_rf_test[idxs.values]
    df_plot = df_plot.sort_values('Area')
    # Calcular métricas por año
    y_true_year = df_plot['yield_mean'].values
    y_rf_year = df_plot['RF'].values
    rmse_rf_year = root_mean_squared_error(y_true_year, y_rf_year)
    mae_rf_year = mean_absolute_error(y_true_year, y_rf_year)
    r2_rf_year = r2_score(y_true_year, y_rf_year)
    print(f"\nAño {year}:")
    print(f"  Random Forest (Area dummies, tuned) - RMSE: {rmse_rf_year:.2f}, MAE: {mae_rf_year:.2f}, R2: {r2_rf_year:.3f}")
    plt.figure(figsize=(14, 8))
    plt.plot(df_plot['Area'], df_plot['yield_mean'], marker='o', label='Real Yield', color='green')
    plt.plot(df_plot['Area'], df_plot['RF'], marker='o', label='RF Prediction', color='blue')
    plt.title(f'Predicciones por País - Random Forest (Area dummies, tuned) ({year})')
    plt.xlabel('País')
    plt.ylabel('Predicción de Rendimiento')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"results/predicciones_por_pais_rf_{year}.png")
    plt.show()

