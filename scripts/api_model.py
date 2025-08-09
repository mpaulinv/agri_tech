from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Cargar el modelo entrenado y las features
MODEL_PATH = "results/model_rf_final.joblib"
FEATURES_PATH = "results/features_rf_area.txt"

app = FastAPI()

class PredictionInput(BaseModel):
    Area: str
    rain_fall_mean: float
    pesticides_mean: float
    avg_temp_median: float

@app.on_event("startup")
def load_model():
    global model, feature_names
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

def boxcox_transform(x):
    return np.log1p(x)

def build_features(input_dict, feature_names):
    rain = input_dict['rain_fall_mean']
    pest = input_dict['pesticides_mean']
    temp = input_dict['avg_temp_median']
    rain_boxcox = boxcox_transform(rain)
    pest_boxcox = boxcox_transform(pest)
    temp_boxcox = boxcox_transform(temp)
    rain_boxcox_square = rain_boxcox ** 2
    rain_boxcox_cube = rain_boxcox ** 3
    pest_boxcox_square = pest_boxcox ** 2
    pest_boxcox_cube = pest_boxcox ** 3
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

def predict_yield(input_data, feature_names, model):
    input_dict = input_data.dict()
    X = build_features(input_dict, feature_names)
    pred = model.predict(X)[0]
    return float(pred)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        pred = predict_yield(input_data, feature_names, model)
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
