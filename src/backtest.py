import pandas as pd
import xgboost as xgb
import numpy as np

def run_backtest():
    # 1. Load data and model
    df = pd.read_csv('data/training_matrix.csv')
    df['date'] = pd.to_datetime(df['date'])
    model = xgb.XGBClassifier()
    model.load_model('models/v1_xgboost_gpu.json')

    # 2. Filter for the "Out-of-Sample" period (2025+)
    test_set = df[df['date'] >= '2025-01-01'].copy()
    
    features = [
        'wgt_lbs', 'draw', 'combo_runs', 'combo_win_pc', 
        'h_course_runs', 'h_course_wins', 'avg_btn_3', 'avg_btn_5', 
        'days_since_last', 'last_pos', 'class_diff', 'avg_class_3',
        'course', 'going', 'dist', 'class'
    ]
    
    for col in ['course', 'going', 'dist', 'class']:
        test_set[col] = test_set[col].astype('category')

    # 3. Get Probabilities
    print("Generating predictions for 2025...")
    test_set['prob'] = model.predict_proba(test_set[features])[:, 1]

    # 4. Pick the "Top Rated" horse per race
    # We use 'id' or 'date'+'course' to group races
    # Assuming 'id' in your CSV refers to the individual result, 
    # we need a way to group horses into the same race.
    # Let's use 'date' and 'course' as a proxy for 'race_id' if you don't have one.
    
    # Sort by probability descending
    test_set = test_set.sort_values(['date', 'course', 'prob'], ascending=[True, True, False])
    
    # Take the horse with the highest prob in each race
    # (Using date/course/dist as a unique race identifier)
    top_picks = test_set.groupby(['date', 'course', 'dist']).head(1).copy()

    # 5. Calculate ROI
    total_bets = len(top_picks)
    winners = top_picks['target_win'].sum()
    win_rate = (winners / total_bets) * 100
    
    # Note: This is a "blind" ROI because we don't have the Odds (SP) in the matrix yet.
    # We can estimate ROI if we assume an average winning price, 
    # but let's look at the Win Rate vs. Random first.
    
    print(f"\n--- BACKTEST RESULTS (2025) ---")
    print(f"Total Races Bet: {total_bets}")
    print(f"Winners Found:   {winners}")
    print(f"Strike Rate:     {win_rate:.2f}%")
    
    # A random strike rate in 10-horse fields is ~10%. 
    # If you are > 20%, the model is doing something significant.
    
    if win_rate > 15:
        print("Status: Positive Signal. Model is outperforming the 'Random' baseline.")
    else:
        print("Status: Weak Signal. Needs more features or a 'Ranking' approach.")

if __name__ == "__main__":
    run_backtest()