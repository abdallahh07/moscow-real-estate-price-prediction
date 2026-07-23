from sklearn.model_selection import train_test_split
from processing.data_manager import load_dataset, save_pipeline
from pipeline import moscow_pipeline
from config.config import config

def run_training():
    df = load_dataset(config.new_builds)
    
    X = df.drop(columns=config.features_to_drop)
    y = df[config.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    
    moscow_pipeline.fit(X_train, y_train)
    save_pipeline(moscow_pipeline)

if __name__ == "__main__":
    run_training()