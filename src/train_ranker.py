import pandas as pd
import xgboost as xgb
import numpy as np
import os

def train_ranker():
    print("Loading matrix for Ranking Model...")
    df = pd.read_csv('data/training_matrix.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a unique Race ID by combining date, course, and distance
    # This is vital for the Ranker to know who is competing against whom
    df['race_id'] = df.groupby(['date', 'course', 'dist']).ngroup()
    
    # Sort by race_id so all competitors are grouped together
    df = df.sort_values('race_id')
    
    cat_cols = ['course', 'going', 'dist', 'class']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    features = [
        'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc', 
        'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5', 
        'days_since_last', 'last_pos', 'class_diff', 'avg_class_3'
    ] + cat_cols
    
    # Split into Train/Test
    train_df = df[df['date'] < '2025-01-01']
    test_df = df[df['date'] >= '2025-01-01']
    
    # XGBRanker needs the "group" sizes (how many horses in each race)
    train_groups = train_df.groupby('race_id').size().values
    test_groups = test_df.groupby('race_id').size().values

    X_train, y_train = train_df[features], train_df['target_win']
    X_test, y_test = test_df[features], test_df['target_win']

    print(f"Training Ranker on {len(train_groups)} races...")

    # XGBRanker setup
    ranker = xgb.XGBRanker(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        objective="rank:ndcg", # Optimizes for putting the winner at the top
        early_stopping_rounds=50
    )

    ranker.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_test, y_test)],
        eval_group=[test_groups],
        verbose=100
    )

    # Save Ranker
    os.makedirs('models', exist_ok=True)
    ranker.save_model('models/v2_ranker_gpu.json')
    print("\nRanking Model saved.")

if __name__ == "__main__":
    train_ranker()