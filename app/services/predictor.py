import pandas as pd
from app.services.model_loader import load_model

model = load_model()

def predict_sales(data):

    input_df = pd.DataFrame([{
        "lag_1": data.features.lag_1,
        "lag_7": data.features.lag_7,
        "rolling_mean_7": data.features.rolling_mean_7
    }])

    prediction = model.predict(input_df)[0]

    return float(prediction)