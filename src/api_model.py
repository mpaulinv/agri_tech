from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from src.config import MODEL_PATH, FEATURES_PATH, LAMBDA_RAIN, LAMBDA_PEST, LAMBDA_TEMP

# Cargar modelo y features al iniciar la API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]
    yield

app = FastAPI(lifespan=lifespan)

# Esquema de entrada para el endpoint 
class PredictionInput(BaseModel):
    Area: str
    rain_fall_mean: float
    pesticides_mean: float
    avg_temp_median: float

# Transformación BoxCox
def boxcox_transform(x, lmbda):
    if lmbda == 0:
        return np.log(x)
    else:
        return (x ** lmbda - 1) / lmbda


# Construye el vector de features a partir de la entrada del usuario 
def build_features(input_dict, feature_names):
    rain = input_dict['rain_fall_mean']
    pest = input_dict['pesticides_mean']
    temp = input_dict['avg_temp_median']
    rain_boxcox = boxcox_transform(rain, LAMBDA_RAIN)
    pest_boxcox = boxcox_transform(pest, LAMBDA_PEST)
    temp_boxcox = boxcox_transform(temp, LAMBDA_TEMP)
    rain_boxcox_square = rain_boxcox ** 2
    rain_boxcox_cube = rain_boxcox ** 3
    temp_boxcox_square = temp_boxcox ** 2
    temp_boxcox_cube = temp_boxcox ** 3
    rain_times_pest = rain_boxcox * pest_boxcox
    rain_over_temp = rain_boxcox / (temp_boxcox + 1e-6)
    pest_over_temp = pest_boxcox / (temp_boxcox + 1e-6)
    area = input_dict['Area']
    area_dummies = {col: 0 for col in feature_names if col.startswith('Area_')}
    area_col = f'Area_{area}'
    if area_col in area_dummies:
        area_dummies[area_col] = 1
    features = {
        'rain_fall_mean_boxcox': rain_boxcox,
        'pesticides_mean_boxcox': pest_boxcox,
        'avg_temp_median_boxcox': temp_boxcox,
        'rain_fall_mean_boxcox_square': rain_boxcox_square,
        'rain_fall_mean_boxcox_cube': rain_boxcox_cube,
        'avg_temp_median_boxcox_square': temp_boxcox_square,
        'avg_temp_median_boxcox_cube': temp_boxcox_cube,
        'rain_fall_mean_boxcox_times_pesticides_mean_boxcox': rain_times_pest,
        'rain_fall_mean_boxcox_over_avg_temp_median_boxcox': rain_over_temp,
        'pesticides_mean_boxcox_over_avg_temp_median_boxcox': pest_over_temp,
    }
    features.update(area_dummies)
    X = pd.DataFrame([features], columns=feature_names)
    return X

# Realiza la predicción usando el modelo cargado
def predict_yield(input_data, feature_names, model):
    input_dict = input_data.dict()
    X = build_features(input_dict, feature_names)
    pred = model.predict(X)[0]
    return float(pred)

# Endpoint de salud para verificar que la API está activa
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint principal para realizar predicciones
@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Por ahora no se permiten valores negativos o cero para las variables de entrada. 
        if (
            input_data.rain_fall_mean <= 0 or
            input_data.pesticides_mean <= 0 or
            input_data.avg_temp_median <= 0
        ):
            raise HTTPException(
                status_code=400,
                detail="All input variables (rain_fall_mean, pesticides_mean, avg_temp_median) must be strictly positive for BoxCox transformation."
            )
        pred = predict_yield(input_data, feature_names, model)
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
