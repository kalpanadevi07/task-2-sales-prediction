import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Config imports
from config import (
    DATA_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI
)

from preprocess import load_data, preprocess
from feature_engineering import create_features
from evaluate import evaluate


def train_model():

    # -----------------------------
    # 1. MLflow Setup
    # -----------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # -----------------------------
    # 2. Load + Preprocess Data
    # -----------------------------
    df = load_data(DATA_PATH)
    df = preprocess(df)
    df = create_features(df)

    # -----------------------------
    # 3. Split Features & Target
    # -----------------------------
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        shuffle=False  # important for time-series
    )

    # -----------------------------
    # 4. Start MLflow Run
    # -----------------------------
    with mlflow.start_run():

        # -------------------------
        # Model Training
        # -------------------------
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)

        # -------------------------
        # Predictions
        # -------------------------
        predictions = model.predict(X_test)

        # -------------------------
        # Evaluation
        # -------------------------
        mae, rmse = evaluate(y_test, predictions)

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")

        # -------------------------
        # MLflow Logging
        # -------------------------
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # Log model in MLflow
        mlflow.sklearn.log_model(model, "model")

        # -------------------------
        # Save model locally
        # -------------------------
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved at: {MODEL_PATH}")

    print("Training completed successfully 🚀")


if __name__ == "__main__":
    train_model()