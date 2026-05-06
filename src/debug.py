import pandas as pd
import xgboost as xgb

# Load your last model
model = xgb.Booster()
model.load_model('models/v2_deep_ranker.json')

# Get feature importance (Weight)
importance = model.get_score(importance_type='weight')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n--- THE LEAK DETECTOR ---")
for feature, score in importance:
    print(f"{feature}: {score}")