### EDA para feature engineering. 
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
#import dvc 
from utils import describe_categorical, describe_continuous, transformaciones_contemporaneas, interacciones_contemporaneas, calcular_historico, cambio_porcentual, cambio_logaritmo, contemporaneas_polinomios

### Revisar la correlacion entre variables.
### No quiero usar el target para no hacer inferencia antes de generar data de entrenamiento y test.

cleaned_data = pd.read_csv("data/maize_cleaned.csv")
print(cleaned_data.head())

continuous_columns = ['rain_fall_mean', 'pesticides_mean', 'avg_temp_mean', 'avg_temp_median']

### Descriptivos de las variables continuas en el dataset limpio
#for column in continuous_columns:
#    describe_continuous(cleaned_data, column)  

#Correlación entre variables continuas
#correlation_matrix = cleaned_data[continuous_columns].corr()
#plt.figure(figsize=(10, 6))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
#plt.title("Matriz de Correlación entre Variables Continuas")
#plt.show()

feature_engineered_data = transformaciones_contemporaneas(cleaned_data, continuous_columns)

### Descriptivos, plots y correlaciones con las nuevas variables
excluded = ['Area', 'Year', 'yield_mean']
all_continuous_columns = feature_engineered_data.columns.drop(excluded)

#for column in all_continuous_columns:
#    describe_continuous(feature_engineered_data, column)

#Correlación entre variables continuas
#correlation_matrix = feature_engineered_data[all_continuous_columns].corr()
#plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
#plt.title("Matriz de Correlación entre Variables Continuas (con transformaciones)")
#plt.show()

### De estas transformaciones las que mas se ajustan a normalizar la distribucion son las box-cox. Además, 
### cuadrados y cubos pueden ser útiles para capturar relaciones no lineales.

### Otra opcion es generar interacciones con las box-cox. Propongo utilizar razones y productos.
### Puede tener sentido que las variables de lluvia, pesticidas y temperatura promedio tengan interacciones entre ellas.  

interaction_columns = ['rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_mean_boxcox', 'avg_temp_median_boxcox']

feature_engineered_data = interacciones_contemporaneas(feature_engineered_data, interaction_columns)

### Descriptivos, plots y correlaciones con las nuevas variables
excluded = ['Area', 'Year', 'yield_mean']
all_continuous_columns = feature_engineered_data.columns.drop(excluded)

#for column in all_continuous_columns:
#    describe_continuous(feature_engineered_data, column)

#Correlación entre variables continuas
#correlation_matrix = feature_engineered_data[all_continuous_columns].corr()
#plt.figure(figsize=(12, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
#plt.title("Matriz de Correlación entre Variables Continuas (con transformaciones)")
#plt.show()

# añadimos medianas de años anteriores. la idea es usar una metrica robusta a outliers para calcular 
# cambios porcentuales en las variables de lluvia, pesticidas y temperatura promedio respecto a los históricos.
# esto impedirá incluiir el primer año en el dataset de entrenamiento. para los siguientes años, si un país no tiene datos, se usará la mediana de los años anteriores para el dataset en general.

historicos = calcular_historico(feature_engineered_data, ['rain_fall_mean', 'pesticides_mean', 'avg_temp_mean', 'avg_temp_median'])

## descriptivos para historicos 
historical_columns = ['rain_fall_mean_hist_mean', 'pesticides_mean_hist_mean', 'avg_temp_mean_hist_mean', 'avg_temp_median_hist_mean']


for column in historical_columns:
    describe_continuous(historicos, column)

### Ahora calculamos el cambio porcentual respecto a los historicos
for column in continuous_columns:
    cambio_porcentual(historicos, column, f'{column}_hist_mean')

### Añadimos cambio logaritmico respecto a los historicos
for column in continuous_columns:
    cambio_logaritmo(historicos, column, f'{column}_hist_mean')


print(historicos.columns)

change_columns  = ['rain_fall_mean_cambio_porcentual', 'pesticides_mean_cambio_porcentual',
       'avg_temp_mean_cambio_porcentual', 'avg_temp_median_cambio_porcentual']

for column in change_columns:
    describe_continuous(historicos, column)

change_columns  = ['rain_fall_mean_cambio_logaritmico', 'pesticides_mean_cambio_logaritmico',
       'avg_temp_mean_cambio_logaritmico', 'avg_temp_median_cambio_logaritmico']


for column in change_columns:
    describe_continuous(historicos, column)


### Cuadrados y cubos para box-cox 
boxcox_columns = ['rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox']
historicos =  contemporaneas_polinomios(historicos, boxcox_columns)

boxcox_columns_polinomios = [col for col in historicos.columns if any(key in col for key in ['boxcox_square', 'boxcox_cube'])]

for column in boxcox_columns_polinomios:
    describe_continuous(historicos, column)

print(historicos.columns)

### Conclusiones: 
# 1) Las variables continuas tienen distribuciones sesgadas a la derecha, lo que puede afectar el rendimiento del modelo. Una buena transformacione es la box-cox. Otra posibilidad es utilizar las estandarizadas 
#    especialmente si se quieren incluir los términos cuadráticos y cúbicos.
# 2) Creo que las variables avg_temp_mean y avg_temp_median son redundantes, ya que tienen una correlacion muy alta. Preferiria quedarme con la mediana al no ser tan sensible a outliers.
# 3) Las interacciones entre las variables de lluvia, pesticidas y temperatura pueden capturar relaciones no lineales importantes.
# 4) No creo que incluir cambio porcentual respecto a uso de pesticidas sea muy intuitivo. Pero si quiero mantenter cambio porcentual respecto a lluiva y temperatura mediana. 

## Bajo estas consideraciones, quedan los siguientes features candidatos para el modelo:
# 1)  (9) 'rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox', 'rain_fall_mean_boxcox_squared', 'rain_fall_mean_boxcox_cubed', 'pesticides_mean_boxcox_squared', 'pesticides_mean_boxcox_cubed', 'avg_temp_median_boxcox_squared', 'avg_temp_median_boxcox_cubed'
# 2)  (9) rain_fall_mean_norm', 'rain_fall_mean_square_norm', 'rain_fall_mean_cube_norm', 'pesticides_mean_norm', 'pesticides_mean_square_norm', 'pesticides_mean_cube_norm', 'avg_temp_median_norm', 'avg_temp_median_square_norm', 'avg_temp_median_cube_norm'
# 3)  (6) 'rain_fall_mean_boxcox_over_pesticides_mean_boxcox', 'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 'rain_fall_mean_boxcox_times_avg_temp_median_boxcox', 'pesticides_mean_boxcox_over_avg_temp_median_boxcox', 'pesticides_mean_boxcox_times_avg_temp_median_boxcox',
# 4)   (3) 'rain_fall_mean_cambio_porcentual', 'pesticides_mean_cambio_porcentual', 'avg_temp_median_cambio_porcentual'
# 5)   (4) 'rain_fall_mean_cambio_logaritmico', 'pesticides_mean_cambio_logaritmico', 'avg_temp_median_cambio_logaritmico'

#donde 1 y 2 son excluyentes entre si. 3 interacciones 4. cambios porcentuales respecto a historicos. 5 cambios logaritmicos respecto a historicos. 4 y 5 son excluyentes entre si.
# esto da un total de 24 variables candidato para el modelo aunque algunas son excluyentes entre si. 

### guardar la base. Nota, esto eventualmente se debe hacer en el feature_engineering con transformaciones candidatas. 

print(historicos.isnull().sum())

historicos.to_csv("data/historicos.csv", index=False)
