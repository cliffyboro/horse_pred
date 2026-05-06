import pandas as pd
import xgboost as xgb
import os

def train_refined_ranker():
    df = pd.read_csv('data/training_matrix.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['race_id'] = df.groupby(['date', 'course', 'dist']).ngroup()
    df = df.sort_values('race_id')
    
    cat_cols = ['course', 'going', 'dist', 'class']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    features = [
        'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc', 
        'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5', 
        'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
        'gear_change', 'or_change', 'true_wgt', 'log_prize'  # THE NEW ALPHA
    ] + cat_cols
    
    train_df = df[df['date'] < '2025-01-01']
    test_df = df[df['date'] >= '2025-01-01']
    
    train_groups = train_df.groupby('race_id').size().values
    test_groups = test_df.groupby('race_id').size().values

    # ADVANCED HYPERPARAMETERS FOR PATTERN RECOGNITION
    ranker = xgb.XGBRanker(
        n_estimators=2000,          # More trees to catch smaller residuals
        learning_rate=0.01,         # Much slower learning rate to find subtle patterns
        max_depth=8,                # Deeper trees to see higher-order interactions
        min_child_weight=5,         # Prevents the model from focusing on "fluke" single rows
        subsample=0.8,              # Randomly samples rows to ensure robustness
        colsample_bytree=0.8,       # Randomly samples features per tree
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        objective="rank:ndcg",
        early_stopping_rounds=100   # More patience for the slow learning rate
    )

    ranker.fit(
        train_df[features], train_df['target_win'],
        group=train_groups,
        eval_set=[(test_df[features], test_df['target_win'])],
        eval_group=[test_groups],
        verbose=100
    )

    ranker.save_model('models/v2_deep_ranker.json')
    print("\nDeep Detail Ranker saved.")

if __name__ == "__main__":
    train_refined_ranker()