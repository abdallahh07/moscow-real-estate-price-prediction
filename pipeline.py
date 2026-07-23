from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from config.config import config

moscow_pipeline = Pipeline([
    ("encoder", ColumnTransformer([
        ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), config.features)
    ], remainder="passthrough")),
    ("scaler", StandardScaler(with_mean=False)),
    ("model", CatBoostRegressor(verbose=0))
])