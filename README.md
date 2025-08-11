# agri_tech

## Descripción del Proyecto

AgriTech es un sistema de predicción de rendimiento de cultivos de maíz basado en machine learning que utiliza datos climáticos y agrícolas para generar predicciones. El proyecto implementa una API FastAPI que permite realizar predicciones en basadas en variables como país, precipitación, uso de pesticidas y temperatura promedio.
Incluyo algunos detalles encontrados durante el projecto en el archivo hallazgos. 

## Características Principales

- **Predicción de Rendimiento**: Modelo de machine learning para predecir el rendimiento de maíz
- **Pipeline Automatizado**: Procesamiento completo de datos desde la descarga hasta el entrenamiento
- **API REST**: Interfaz FastAPI para predicciones 
- **Feature Engineering**: Transformaciones BoxCox, interacciones y cambios con respecto a tendencias históricas
- **Validación Temporal**: Cross-validation respetando la secuencia temporal de los datos

## Estructura del Proyecto

```
proyecto-agritech/
│
├── data/                          # Datos del proyecto
│   ├── yield_df.csv              # Datos raw descargados de Kaggle
│   ├── maize_cleaned.csv         # Datos limpios de maíz
│   └── feature_engineered_data.csv # Datos con feature engineering
│
├── model/                         # Modelo entrenado
│   ├── model_rf_final.joblib     # Modelo Random Forest final
│   └── features_rf_area.txt      # Lista de features del modelo
│
├── logs/                          # Archivos de log
│   ├── data_pipeline.log         # Logs del pipeline de datos
│   └── feature_engineering_pipeline.log # Logs de feature engineering
│
├── __init__.py                    # Inicialización del paquete
├── api_model.py                   # API FastAPI para predicciones
├── config.py                      # Configuraciones del proyecto
├── data_pipeline.py               # Pipeline de procesamiento de datos
├── model.py                       # Entrenamiento del modelo final
├── pipeline_feature_engineering.py # Pipeline de feature engineering
├── utils.py                       # Funciones utilitarias
└── README.md                      # Este archivo
```

## Instalación

### Prerrequisitos

- Python 3.13
- pip
- Cuenta de Kaggle (para descarga de datos)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd proyecto-agritech
```

2. **Crear entorno virtual**
```bash
python -m venv agritech_env
source agritech_env/bin/activate  # En Windows: agritech_env\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Configurar Kaggle API**
   - Descargar `kaggle.json` desde tu cuenta de Kaggle
   - Colocarlo en `~/.kaggle/` (Linux/Mac) o `C:\Users\<username>\.kaggle\` (Windows)
   - Establecer permisos: `chmod 600 ~/.kaggle/kaggle.json`

## Uso del Proyecto

### 1. Ejecutar Pipeline Completo

```bash
# 1. Procesar datos raw
python data_pipeline.py

# 2. Generar features
python pipeline_feature_engineering.py

# 3. Entrenar modelo final
python model.py
```

### 2. Iniciar API

```bash
uvicorn src.api_model:app --host 0.0.0.0 --port 8000 --reload
```

La API estará disponible en `http://localhost:8000`

### 3. Realizar Predicciones

**Endpoint de salud:**
```bash
curl http://localhost:8000/health
```

**Realizar predicción:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Area": "Argentina",
       "rain_fall_mean": 1200,
       "pesticides_mean": 45,
       "avg_temp_median": 22
     }'
```

**Respuesta esperada:**
```json
{
  "prediction": 15513.9
}
```

## Metodología

### Pipeline de Datos

1. **Descarga y Limpieza**: Obtención de datos desde Kaggle y filtrado para cultivos de maíz
2. **Agregación**: Manejo de duplicados utilizando mediana para temperatura y media para otras variables
3. **Feature Engineering**: Creación de variables derivadas y transformaciones

### Feature Engineering

- **Transformaciones BoxCox**: Normalización de distribuciones
- **Polinomios**: Variables cuadráticas y cúbicas
- **Interacciones**: Productos y razones entre variables
- **Variables Históricas**: Cambios respecto a promedios históricos por área
- **One-Hot Encoding**: Para la variable categórica 'Area'

### Modelado

- **Validación Temporal**: Cross-validation respetando el orden cronológico
- **Modelos Evaluados**: 
  - Regresión Lineal
  - Efectos Aleatorios (MixedLM)
  - Efectos Fijos
  - Random Forest (modelo final)

## Parámetros del Modelo

El modelo Random Forest final utiliza los siguientes hiperparámetros:

```python
n_estimators = 550
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = 2
random_state = 42
```

## Variables de Entrada

- **Area** (str): País o región en Inglés y con la primera letra en mayúscula (ej: "Argentina", "Brazil")
- **rain_fall_mean** (float): Precipitación media anual (mm)
- **pesticides_mean** (float): Uso medio de pesticidas (toneladas)
- **avg_temp_median** (float): Temperatura media (°C)

**Restricciones**: Todas las variables numéricas deben ser > 0 para la transformación BoxCox.

## Logging y Monitoreo

El proyecto incluye logging detallado en:
- `logs/data_pipeline.log`: Procesamiento de datos
- `logs/feature_engineering_pipeline.log`: Creación de features

## Configuración

Todas las configuraciones están centralizadas en `config.py`:

```python
# Rutas de archivos
DATA_RAW_PATH = "data/yield_df.csv"
MODEL_PATH = "model/model_rf_final.joblib"

# Parámetros BoxCox (Estimados durante la generación de features)
LAMBDA_RAIN = 0.46461002838902316
LAMBDA_PEST = 0.09023987209681697
LAMBDA_TEMP = 1.2687660894233703
```

## Otros Archivos
Bajo la carpeta script se encuentran otros archivos que justifican decisiones técnicas como la selección de variables (EDA, EDA_Feature_engineering, EDA_features) 
seleccion y comparacion de modelos (Seleccion_modelo), validación del modelo seleccionado (model_testing)

## Limitaciones

- El modelo está entrenado específicamente para cultivos de maíz
- Requiere valores positivos para todas las variables numéricas

## Próximos Pasos

- [ ] Validar datos de lluvía por país y Yield en países outliers
- [ ] Container para el proyecto 
- [ ] Exploración de features de cambio vs históricos más complejos 
- [ ] Otro tipo de modelos (Gradient Boosting, Redes Neuronales)
- [ ] Integrar más fuentes de datos climáticos, condiciones de suelos, etc...
