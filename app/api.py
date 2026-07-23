from fastapi import APIRouter
from app.schemas import PredictionInput, PredictionOutput
from predict import make_prediction

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    prediction = make_prediction(input_data.dict())
    return PredictionOutput(predicted_price=prediction)