import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import os

def train_iteration_one():
    print("Loading training matrix...")
    df = pd.read_csv('data/training_matrix.csv')
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Define categorical columns
    cat_cols = ['course', 'going', 'dist', 'class']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    features = [
        'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc', 
        'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5', 
        'days_since_last', 'last_pos', 'class_diff', 'avg_class_3'
    ] + cat_cols
    
    target = 'target_win'

    # Time-based split: Train on pre-2025, Test on 2025+
    train_df = df[df['date'] < '2025-01-01']
    test_df = df[df['date'] >= '2025-01-01']
    
    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"Training on {len(X_train):,} rows | Testing on {len(X_test):,} rows...")

    # XGBoost GPU Configuration
    model = xgb.XGBClassifier(
        n_estimators=1000,          # Increased since GPU is fast
        max_depth=6,
        learning_rate=0.05,
        # GPU PARAMETERS
        tree_method="hist",         # Histogram-based algorithm
        device="cuda",              # Use the NVIDIA GPU
        enable_categorical=True,    # Handle course/going/class automatically
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="logloss"
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, preds)
    print(f"\nModel AUC Score: {auc_score:.4f}")

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- Top Predictive Features ---")
    print(importance.head(10))

    os.makedirs('models', exist_ok=True)
    model.save_model('models/v1_xgboost_gpu.json')
    print("\nModel saved to models/v1_xgboost_gpu.json")

if __name__ == "__main__":
    train_iteration_one()