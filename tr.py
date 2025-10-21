import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timezone
import joblib

DB_PATH = "raceform.db"
TABLE = "data"

print(f"📦 Loading from {DB_PATH} ...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
conn.close()
print(f"✅ Loaded {len(df):,} rows (no deletions).")

# -------------------------------------
# 1. Parse dates safely
# -------------------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df["year"] = df["date"].dt.year.fillna(0).astype(int)

now_utc = pd.Timestamp.now(tz="UTC")
df["days_ago"] = (now_utc - df["date"]).dt.days.fillna(df["date"].notna().sum())

# -------------------------------------
# 2. Convert numeric-like columns
# -------------------------------------
numeric_cols = ["or", "rpr", "ts", "btn", "ovr_btn", "pos", "age"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace NaNs with median (or 0 if all missing)
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median() if not df[col].isna().all() else 0
        df[col] = df[col].fillna(median_val)

# -------------------------------------
# 3. Add cumulative features
# -------------------------------------
def add_cum_features(df, group_col, cols, prefix):
    grp = df.groupby(group_col, sort=False)
    df[f"{prefix}_count"] = grp.cumcount()
    for c in cols:
        if c not in df.columns:
            continue
        cumsum_shifted = grp[c].transform(lambda x: x.cumsum().shift(1).fillna(0))
        df[f"{prefix}_{c}_sum"] = cumsum_shifted
        df[f"{prefix}_{c}_avg"] = (
            cumsum_shifted / df[f"{prefix}_count"].replace(0, np.nan)
        ).fillna(0)
    return df

df = add_cum_features(df, "horse", ["rpr", "ts", "btn"], prefix="horse")
df = add_cum_features(df, "trainer", ["rpr", "ts", "btn"], prefix="trainer")
df = add_cum_features(df, "jockey", ["rpr", "ts", "btn"], prefix="jockey")

# -------------------------------------
# 4. Optional — going+distance winrate
# -------------------------------------
if all(c in df.columns for c in ["horse", "going", "dist", "pos"]):
    df["horse_going_dist_count"] = df.groupby(["horse", "going", "dist"]).cumcount()
    df["horse_going_dist_win"] = df.groupby(["horse", "going", "dist"])["pos"].transform(
        lambda x: (x == 1).cumsum().shift(1).fillna(0)
    )
    df["horse_going_dist_winrate"] = (
        df["horse_going_dist_win"]
        / df["horse_going_dist_count"].replace(0, np.nan)
    ).fillna(0)
else:
    df["horse_going_dist_winrate"] = 0.0

# -------------------------------------
# 5. Feature list
# -------------------------------------
features = [
    "or", "rpr", "ts", "age", "btn", "ovr_btn",
    "horse_rpr_avg", "horse_ts_avg", "horse_btn_avg",
    "trainer_rpr_avg", "trainer_ts_avg", "trainer_btn_avg",
    "jockey_rpr_avg", "jockey_ts_avg", "jockey_btn_avg",
    "days_ago", "horse_going_dist_winrate"
]
features = [f for f in features if f in df.columns]

print(f"\n🧩 Using {len(features)} features:")
print(features)

# Replace NaNs in feature columns with 0
df[features] = df[features].fillna(0)

# -------------------------------------
# 6. Define target (Win = 1, else 0)
# -------------------------------------
df["pos"] = pd.to_numeric(df["pos"], errors="coerce").fillna(0)
y = np.where(df["pos"] == 1, 1, 0)
X = df[features]

# -------------------------------------
# 7. Train/test split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------
# 8. Train model
# -------------------------------------
print("\n🏇 Training RandomForest model (keeping all rows)...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------------
# 9. Evaluate
# -------------------------------------
preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"\n✅ R²: {r2:.4f}")
print(f"✅ RMSE: {rmse:.4f}")

# -------------------------------------
# 10. Save model + feature importances
# -------------------------------------
joblib.dump(model, "race_predictor.pkl")
print("💾 Model saved -> race_predictor.pkl")

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🏁 Top Feature Importances:")
print(imp.head(20))

print("\n✅ Training complete (no rows removed).")
