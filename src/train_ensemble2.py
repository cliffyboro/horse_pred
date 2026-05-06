import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

def train_ensemble():
    print("Loading training matrix...")
    df = pd.read_csv('data/training_matrix.csv')
    df['date'] = pd.to_datetime(df['date'])

    cat_cols = ['course', 'going', 'dist', 'class']
    num_features = ['wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc',
                    'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5',
                    'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
                    'gear_change', 'or_change', 'true_wgt', 'log_prize']

    features = num_features + cat_cols

    # Preprocessing
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    scaler = StandardScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    X = df[features].values.astype(np.float32)
    y = df['target_win'].values

    train_mask = df['date'] < '2025-01-01'
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

    models = {}
    test_preds = {}

    # 1. XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        tree_method='hist', device='cuda', eval_metric='auc', random_state=42
    )
    xgb_model.fit(X_train, y_train)
    models['xgb'] = xgb_model
    test_preds['xgb'] = xgb_model.predict_proba(X_test)[:, 1]

    # 2. RandomForest
    print("Training RandomForest...")
    rf_model = RandomForestClassifier(
        n_estimators=600, max_depth=12, n_jobs=-1, 
        random_state=42, class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    models['rf'] = rf_model
    test_preds['rf'] = rf_model.predict_proba(X_test)[:, 1]

    # 3. ExtraTrees (Diversity addition)
    print("Training ExtraTrees...")
    et_model = ExtraTreesClassifier(
        n_estimators=600, 
        max_depth=12, 
        n_jobs=-1,
        random_state=42,
        bootstrap=True
    )
    et_model.fit(X_train, y_train)
    models['et'] = et_model
    test_preds['et'] = et_model.predict_proba(X_test)[:, 1]

    # 4. LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=9,
        num_leaves=256, n_jobs=-1, verbose=-1, random_state=42
    )
    lgb_model.fit(X_train, y_train)
    models['lgb'] = lgb_model
    test_preds['lgb'] = lgb_model.predict_proba(X_test)[:, 1]

    # ====================== ENSEMBLE ======================
    ensemble_pred = np.mean(list(test_preds.values()), axis=0)

    print("\n" + "="*70)
    print("FINAL ENSEMBLE RESULTS (2025 Test Set)")
    print("="*70)
    for name, pred in test_preds.items():
        auc = roc_auc_score(y_test, pred)
        print(f"{name.upper():<4} AUC: {auc:.4f}")

    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    print(f"ENSEMBLE AUC: {ensemble_auc:.4f}   ← Final Score")

    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    for name, model in models.items():
        joblib.dump(model, f'models/{name}_model.pkl')

    print("\n✅ All 4 models saved!")
    return ensemble_auc

if __name__ == "__main__":
    train_ensemble()