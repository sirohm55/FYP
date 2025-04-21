import joblib
import pandas as pd
from datetime import datetime

def predict_pcu(model_path, date_str, hour):
    # === Load model ===
    model = joblib.load(model_path)
    
    # === Prepare complete input features ===
    date = pd.to_datetime(date_str)
    
    # Calculate all features used in training
    features = {
        "hour": hour,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek in [5, 6]),
        "week_of_year": date.isocalendar().week,
        "month": date.month
    }
    
    # Create DataFrame with same column order as training
    input_df = pd.DataFrame([features], columns=[
        "hour", 
        "day_of_week", 
        "is_weekend", 
        "week_of_year", 
        "month"
    ])
    
    # === Predict ===
    pcu_pred = model.predict(input_df)[0]
    return pcu_pred

# Example Usage
model_path = "Without date encoded/saved_models/Optuna/XGBoost/ccc2617079d5_Lane_2_TotalWeighted.pkl"
prediction = predict_pcu(
    model_path=model_path,
    date_str="2025-09-22",  # YYYY-MM-DD format
    hour=8                  # 0-23
)

print(f"Predicted PCU: {prediction:.2f}")