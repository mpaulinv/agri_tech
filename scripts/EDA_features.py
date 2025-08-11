#### Exploracion de la relacion entre las variables y el target 
#### Este script se utiliza para realizar un analsis exploratorio entre las variables candidatas y el target 
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from utils import association_continuous

historicos = pd.read_csv("data/historicos.csv")
print(historicos.head())
print(historicos.info)

### Tenemos 2028 filas. 
### Estrategia de validacion. Guardamos los ultimos años como parte del test set, en este caso 
### los ultimos 5 años (2009-2013)
train = historicos[historicos['Year'] <= 2008]
test = historicos[historicos['Year'] > 2008]    
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
## Esto nos deja un set de prueba de alreadedor del 22%

### Exploremos la relacion entre las variables y el target
### Del EDA feature engineering, tenemos las siguientes variables candidatas: 

# 1)  (9) 'rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox', 'rain_fall_mean_boxcox_squared', 'rain_fall_mean_boxcox_cubed', 'pesticides_mean_boxcox_squared', 'pesticides_mean_boxcox_cubed', 'avg_temp_median_boxcox_squared', 'avg_temp_median_boxcox_cubed'
# 2)  (9) rain_fall_mean_norm', 'rain_fall_mean_square_norm', 'rain_fall_mean_cube_norm', 'pesticides_mean_norm', 'pesticides_mean_square_norm', 'pesticides_mean_cube_norm', 'avg_temp_median_norm', 'avg_temp_median_square_norm', 'avg_temp_median_cube_norm'
# 3)  (6) 'rain_fall_mean_boxcox_over_pesticides_mean_boxcox', 'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 'rain_fall_mean_boxcox_times_avg_temp_median_boxcox', 'pesticides_mean_boxcox_over_avg_temp_median_boxcox', 'pesticides_mean_boxcox_times_avg_temp_median_boxcox',
# 4)   (3) 'rain_fall_mean_cambio_porcentual', 'pesticides_mean_cambio_porcentual', 'avg_temp_median_cambio_porcentual'
# 5)   (4) 'rain_fall_mean_cambio_logaritmico', 'pesticides_mean_cambio_logaritmico', 'avg_temp_median_cambio_logaritmico'


boxcox_columns = ['rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox',
                  'rain_fall_mean_boxcox_square', 'rain_fall_mean_boxcox_cube',
                  'pesticides_mean_boxcox_square', 'pesticides_mean_boxcox_cube',
                  'avg_temp_median_boxcox_square', 'avg_temp_median_boxcox_cube']
norm_columns = ['rain_fall_mean_norm', 'rain_fall_mean_square_norm', 'rain_fall_mean_cube_norm',
                'pesticides_mean_norm', 'pesticides_mean_square_norm', 'pesticides_mean_cube_norm',
                'avg_temp_median_norm', 'avg_temp_median_square_norm', 'avg_temp_median_cube_norm']
interaction_columns = ['rain_fall_mean_boxcox_over_pesticides_mean_boxcox', 
                       'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 
                       'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 
                       'rain_fall_mean_boxcox_times_avg_temp_median_boxcox', 
                       'pesticides_mean_boxcox_over_avg_temp_median_boxcox', 
                       'pesticides_mean_boxcox_times_avg_temp_median_boxcox']
cambio_porcentual_columns = ['rain_fall_mean_cambio_porcentual', 
                              'pesticides_mean_cambio_porcentual', 
                              'avg_temp_median_cambio_porcentual']
cambio_logaritmo_columns = ['rain_fall_mean_cambio_logaritmico', 
                                'pesticides_mean_cambio_logaritmico', 
                                'avg_temp_median_cambio_logaritmico']

# Empezamos con las variables box-cox
for column in boxcox_columns:
    association_continuous(historicos, column, 'yield_mean')
### Se observa que las relaciones son no lineales, por tanto creo que el modelo debe incorporar las transfomaciones

### Variables normalizadas
for column in norm_columns:
    association_continuous(historicos, column, 'yield_mean')

### Interacciones 
for column in interaction_columns:
    association_continuous(historicos, column, 'yield_mean')

### Cambio porcentual
for column in cambio_porcentual_columns:
    association_continuous(historicos, column, 'yield_mean')

### Cambio logaritmico
for column in cambio_logaritmo_columns:
    association_continuous(historicos, column, 'yield_mean')

### Aqui hay algo de feedback entre el analisis y el feature engineering. 
### Las transformaciones boxcox parecen tener un mejor rango contra yield, pero es importante incluir terminos no lineales 
### por otro lado, los cambios porcentuales no funcionan tan bien ya que hay muchos outliers. Tal vez sea mejor incluir un logaritmo de la razon entre las variables.  
       
### Basado en el analisis, creo que las siguientes variables son las mas prometedoras:
### 1)  (9) 'rain_fall_mean_boxcox', 'pesticides_mean_boxcox', 'avg_temp_median_boxcox', 'rain_fall_mean_boxcox_squared', 'rain_fall_mean_boxcox_cubed', 'pesticides_mean_boxcox_squared', 'pesticides_mean_boxcox_cubed', 'avg_temp_median_boxcox_squared', 'avg_temp_median_boxcox_cubed'
### 2)  (3)     'rain_fall_mean_boxcox_times_pesticides_mean_boxcox', 'rain_fall_mean_boxcox_over_avg_temp_median_boxcox', 'pesticides_mean_boxcox_over_avg_temp_median_boxcox'
### 3)  (3) 'rain_fall_mean_cambio_logaritmico', 'pesticides_mean_cambio_logaritmico', 'avg_temp_median_cambio_logaritmico'

#### Correlaciones entre variables predicivas 

correlation_matrix = historicos[boxcox_columns + interaction_columns + cambio_logaritmo_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Matriz de Correlación entre Variables Predictivas")
plt.show()