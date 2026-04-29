import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "sales_forecasting"

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "rolling_mean_7"
]

TARGET_COLUMN = "sales"