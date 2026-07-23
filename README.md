# Moscow Real Estate Price Prediction

Predicts new-build property prices in Moscow using listing, developer, and location features.

## Tech Stack

**Language & Core Libraries**

- Python 3.11
- pandas, NumPy — data manipulation
- scikit-learn — pipeline, preprocessing, train/test split
- CatBoost — gradient boosting regressor (best-performing model, selected after comparing 9 algorithms)

**Config & Validation**

- PyYAML — configuration management
- Pydantic — schema validation for config and API I/O

**Deployment**

- FastAPI — REST API layer
- Uvicorn — ASGI server
- Docker — containerization
- Railway — cloud hosting

**Model Persistence**

- joblib — pipeline serialization

## Project Structure

moscow-real-estate/
├── config/
│ ├── config.yml # paths, target, featurelists, hyperparameters
│ └── config.py # loads and validates config.yml via Pydantic
├── data/ # raw CSVs
├── processing/
│ ├── data_manager.py # load_dataset(), save_pipeline()
│ └── features.py # custom sklearn transformers
├── pipeline.py # sklearn Pipeline: encoding → scaling → CatBoost
├── train_pipeline.py # trains and saves the model
├── predict.py # loads trained pipeline, makes predictions
├── app/
│ ├── schemas.py # PredictionInput / PredictionOutput
│ ├── api.py # /predict endpoint
│ └── main.py # FastAPI app, /health endpoint
├── Dockerfile
└── requirements.txt

## Model Performance

- Best model: CatBoostRegressor (selected via R² comparison against RandomForest, GradientBoosting, XGBoost, LightGBM, KNN, Linear/Ridge/Lasso, plus voting/stacking ensembles)
- Validated with RepeatedKFold cross-validation

## Running Locally

```bash
pip install -r requirements.txt
python train_pipeline.py
uvicorn app.main:app --reload --port 8001
```

Visit `http://localhost:8001/docs` for interactive API documentation.

## Running with Docker

```bash
docker build -t moscow-real-estate .
docker run -p 8001:8001 moscow-real-estate
```

## API

**`GET /health`** — service status check

**`POST /predict`** — returns predicted price given property features (developer, complex class, district, area, distance to center, etc.)

\*\* requirements.txt
fastapi
uvicorn
pandas
scikit-learn
catboost
joblib
pyyaml
pydantic
