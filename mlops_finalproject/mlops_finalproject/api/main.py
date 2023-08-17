import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from fastapi import FastAPI
from starlette.responses import JSONResponse

from predictor.predict import ModelPredictor
from .models.models import Titanic

app = FastAPI()

@app.get('/', status_code=200)
async def healthcheck():
    return 'Titanic classifier is all ready to go!'

@app.post('/predict')
def extract_name(titanic_features: Titanic):
    predictor = ModelPredictor("/Users/carloslme/Documents/GitHub/itesm-mlops/module-3/session-11/itesm_mlops/itesm_mlops/models/logistic_regression_output.pkl")
    X = [titanic_features.pclass_nan,
        titanic_features.age_nan,
        titanic_features.sibsp_nan,
        titanic_features.parch_nan,
        titanic_features.fare_nan,
        titanic_features.sex_male,
        titanic_features.cabin_Missing,
        titanic_features.cabin_rare,
        titanic_features.embarked_Q,
        titanic_features.embarked_S,
        titanic_features.title_Mr,
        titanic_features.title_Mrs,
        titanic_features.title_rar]
    prediction = predictor.predict([X])
    return JSONResponse(f"Resultado predicción: {prediction}")
