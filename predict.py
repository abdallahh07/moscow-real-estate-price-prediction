import joblib
import pandas as pd
from pathlib import Path
from config.config import config

PIPELINE_PATH = Path(__file__).parent / config.pipeline_save_file
pipeline = joblib.load(PIPELINE_PATH)

def make_prediction(input_data: dict) -> float:
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)
    return prediction[0]