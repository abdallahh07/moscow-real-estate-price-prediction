from pathlib import Path
import pandas as pd 
import joblib 
from config.config import config

def load_dataset(filename: str) -> pd.DataFrame:
    data_path = Path(__file__).parent.parent / config.data_folder / filename
    return pd.read_csv(data_path)

def save_pipeline(pipeline):
    save_path = Path(__file__).parent.parent / config.pipeline_save_file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)