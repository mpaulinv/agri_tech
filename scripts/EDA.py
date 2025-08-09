### Data pipeline. Elaborado por Mario Paulín como parte del proyecto agritech  
### Este script se utiliza para 
# 1) cargar el dataset desde kaggle
# 2) realizar un análisis exploratorio de datos (EDA) 
# 3) Identificar si existen valores faltantes, duplicados, o potenciales inconsistencias o outliers 
# 4) Analizar la estructura del dataset para entender si hay múltiples registros por país/año
# 5) Conclusiones a implementar durante el data pipeline y feature engineering

### Para hacer tracking de los cambios en el dataset, se utilizara DVC en el data pipeline y siguientes modulos (Data Version Control)
### Iniciamos importando las librerías necesarias
import numpy as np 
import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import matplotlib.pyplot as plt 
import seaborn as sns
#import dvc 
from utils import transformaciones_contemporaneas, interacciones_contemporaneas

# 1) cargar el dataset desde kaggle
### Importamos la data directamente desde Kaggle 
### genera el directorio "data" si no existe 
os.makedirs("data", exist_ok=True)
api = KaggleApi()
api.authenticate()
api.dataset_download_files("patelris/crop-yield-prediction-dataset", path="data", unzip=True)

data = pd.read_csv("data/yield_df.csv")


### imprimimos las primeras filas del dataset para verificar que se ha cargado correctamente
data.head()

### Descriptivos basicos del dataset
print("Info:")
data.info()
print("Descripcion:")
data.describe()

## El dataset se ha cargado correctamente. Tiene 28242 filas y 8 columnas. Las columnas son non-null pero vamos a verficar. 
# Revision de valores faltantes 
missing_summary = pd.DataFrame({
    'Missing_Count': data.isnull().sum(),
    'Missing_Percentage': (data.isnull().sum() / len(data)) * 100
})
missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print("Resumen de valores faltantes:")
if missing_summary.empty:
    print("No hay valores faltantes en el dataset.")
else:
    print(missing_summary)

# Revisamos si hay duplicados
duplicate_count = data.duplicated().sum()
print(duplicate_count)

# Tipo de datos de cada columna
print("Resumen de tipos de datos:")
print(data.dtypes.value_counts())


## Vamos a analizar el numero de valores únicos por columna para entender 
## Que variables son categóricas y cuales son continuas
unique_values = data.apply(lambda col: col.nunique())

# Display the result
print("Numero de valores únicos por columna:")
print(unique_values)

## Tenemos 101 Areas, 10 cultivos, 23 años, yield, lluvia, pesticidas y temperatura promedio son continuas.

# 2) y 3) realizar un análisis exploratorio de datos (EDA) 
#### EDA ##### 

# Comienzo definiendo algunas funciones que se utilizaran para analizar los datos. 
def describe_categorical(data, column_name):
    """
    Muestra una tabla de conteos y un gráfico de barras para una variable categórica.

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - column_name: Nombre de la variable categórica a explorar
    """
    # Conteo y proporción de categorías
    counts = data[column_name].value_counts()
    proportions = data[column_name].value_counts(normalize=True) * 100

    # Crear un DataFrame para mostrar los resultados
    summary_table = pd.DataFrame({
        "Count": counts,
        "Proportion (%)": proportions
    })

    # Imprimir la tabla de resumen
    print(f"Conteo y Proporción de Categorías para '{column_name}':")
    print(summary_table)
    print("\n")

    # Gráfico de barras
    sns.barplot(x=counts.index, y=counts.values, palette="Set2", hue=counts.index, legend=False)
    plt.title(f"Proporción de Categorías en '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel("Conteo")
    plt.show()

# Ahora lo ejecutamos para las columnas categóricas
categorical_columns = ['Area', 'Item', 'Year']

for column in categorical_columns:
    describe_categorical(data, column)

### No todas las áreas tienen el mismo número de cultivos
### Por ello, puede haber valores faltantes en algunas áreas para ciertos cultivos.


# Función para describir variables continuas
def describe_continuous(data, column_name):
    """
    Muestra estadísticas descriptivas y un boxplot para una variable continua.

    Parameters:
    - data: Pandas DataFrame que contiene los datos
    - column_name: Nombre de la columna continua a explorar
    """
    # Calculo de estadísticas descriptivas
    stats = {
        "Min": data[column_name].min(),
        "Max": data[column_name].max(),
        "Mean": data[column_name].mean(),
        "Median": data[column_name].median()
    }

    # Imprimir estadísticas descriptivas
    print(f"Estadísticas Descriptivas para '{column_name}':")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print("\n")

    # Plot boxplot
    sns.boxplot(x=data[column_name], color="skyblue")
    plt.title(f"Boxplot of '{column_name}'")
    plt.xlabel(column_name)
    plt.show()

    # Plot histograma
    plt.figure(figsize=(10, 4))
    sns.histplot(data[column_name], kde=True, color="steelblue", bins=20)
    plt.title(f"Histograma de '{column_name}'")
    plt.xlabel(column_name)
    plt.ylabel("Frecuencia")
    plt.show()


# Ahora lo ejecutamos para las columnas continuas
continuous_columns = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

for column in continuous_columns:
    describe_continuous(data, column)   

### Las variables continuas son todas positivas pero la distribucion es en general sesgada a la derecha.

### Verifiquemos si el dataset esta completo por pais para el cultivo de maiz que es el que buscaremos predecir
maize_data = data[data['Item'] == 'Maize']

for column in categorical_columns:
    describe_categorical(maize_data, column)

print(maize_data[maize_data['Area'] == 'India']['Year'].value_counts())
print(maize_data[maize_data['Area'] == 'India'].head())
print(maize_data.columns)

# 4) Identificar si existen valores faltantes, duplicados, o potenciales inconsistencias o outliers
### Se ve que hay paises con años sin datos por ejemplo sudan, dinamarca y montenegro. Por otro lado, hay paises 
### como India con varias observaciones por año. Quiero explorar esto con mas detalle. 
### La idea es enumerar cuantas observaciones unicas de las variables hay por pais y año.

variables = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
unique_counts = maize_data.groupby(['Area', 'Year'])[variables].nunique().reset_index()
print(unique_counts)
unique_counts.to_csv('outputs/maize_country_year_counts.csv', index=False)

for var in variables:
    dist = unique_counts[var].value_counts().sort_index()
    print(f"Distribucion de valores unicos por {var}:")
    print(dist)

### Se observa que por pais/año hay en general 1 valor unico por variable para hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes
### pero, puede haber 2 o mas valores unicos para avg_temp. Esto puede ser porque hay varios registros por año en algunos paises.
### en realidad no tengo manera de saber si esto es correcto o no. Lo que puedo hacer es agregar estos valores para tener solo una 
### fila por pais/año.    

### Conclusiones:
### 1) Si bien el dataset no tiene valores faltantes, si tiene años sin datos para ciertos paises.
### al ser completamente faltante el dato, no tiene mucho sentido imputarlo. Una decision para el modelo 
### es como incluir la data del pais en entrenamiento y prediccion. Una opcion es indicadores por pais pero eso tiene limitantes 
### ya que se incluirian muchas variables categoricas y necesitariamos una categoria como otro pais para aquellos con pocos datos y para predecir 
### fuera de la data de entrenamiento. Otra opcion puede ser incluir latitud y longitud promedio del pais. Otra opcion es buscar generar clusters de paises. 

### 2) Para aquellos paises con varios registros por año se puede generar la temperatura promedio o mediana por ejemplo. 
### 3) El dataset no tiene duplicados.
### 4) El dataset tiene algunas variables con distribuciones sesgadas a la derecha. Se puede considerar transformaciones logaritmicas o Box-Cox para estas variables durante el feature engineering.
### en particular, hay valores muy altos (alredor de 12 veces la mediana) para el yield. Esto puede afectar al modelo. 

