#!/usr/bin/env python3
"""
train_clean.py
Train a clean, no-leakage RandomForest model using raceform.db.
Features use only pre-race info (no ovr_btn, rpr, ts, etc.)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

DB_PATH = "raceform.db"
MODEL_PATH = "race_predictor_clean.pkl"

print(f"📦 Loading from {DB_PATH} ...")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM data", conn)
conn.close()
print(f"✅ Loaded {len(df):,} rows.")

# -------------------------------------
# 🧹 Preprocess
# -------------------------------------
df = df.drop_duplicates(subset=["date", "race_id", "num"], keep="last")
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
df = df[df["date"].notna()]

now_utc = pd.Timestamp.now(tz="UTC")
df["days_ago"] = (now_utc - df["date"]).dt.days

# Convert numeric-like fields
for col in ["or", "age", "rpr", "ts", "btn", "ovr_btn", "pos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["pos"] > 0]

# -------------------------------------
# 📈 Utility: cumulative averages
# -------------------------------------
def add_cum_features(df, group_col, cols, prefix):
    grp = df.groupby(group_col, sort=False)
    df[f"{prefix}_count"] = grp.cumcount()
    for c in cols:
        cumsum_shifted = grp[c].transform(lambda x: x.cumsum().shift(1).fillna(0))
        df[f"{prefix}_{c}_sum"] = cumsum_shifted
        df[f"{prefix}_{c}_avg"] = (
            cumsum_shifted / df[f"{prefix}_count"].replace(0, np.nan)
        ).fillna(0)
    return df

df = add_cum_features(df, "horse", ["rpr", "ts", "btn"], "horse")
df = add_cum_features(df, "trainer", ["rpr", "ts", "btn"], "trainer")
df = add_cum_features(df, "jockey", ["rpr", "ts", "btn"], "jockey")

# -------------------------------------
# 🧠 Derived combo stats
# -------------------------------------
def combo_winrate(df, cols):
    """Compute winrate for unique combo of columns."""
    sub = df.copy()
    sub["is_win"] = (sub["pos"] == 1).astype(int)
    grp = sub.groupby(cols)["is_win"].mean().reset_index()
    name = "_".join(cols) + "_winrate"
    df = df.merge(grp, on=cols, how="left", suffixes=("", f"_{name}"))
    df.rename(columns={"is_win": name}, inplace=True)
    return df

df = combo_winrate(df, ["horse", "jockey"])
df = combo_winrate(df, ["trainer", "course"])
df = combo_winrate(df, ["jockey", "going"])
df = combo_winrate(df, ["horse", "going", "dist"])
df.fillna(0, inplace=True)

# -------------------------------------
# 🎯 Target variable
# -------------------------------------
y = np.where(df["pos"] == 1, 1, 0)

# -------------------------------------
# 🔍 Feature set
# -------------------------------------
features = [
    "or", "age", "days_ago",
    "horse_rpr_avg", "horse_ts_avg", "horse_btn_avg",
    "trainer_rpr_avg", "trainer_ts_avg", "trainer_btn_avg",
    "jockey_rpr_avg", "jockey_ts_avg", "jockey_btn_avg",
    "horse_jockey_winrate_winrate",
    "trainer_course_winrate_winrate",
    "jockey_going_winrate_winrate",
    "horse_going_dist_winrate_winrate"
]

X = df[features].fillna(0)

# -------------------------------------
# ⚙️ Train model
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏇 Training clean RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# -------------------------------------
# 📊 Evaluate
# -------------------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# -------------------------------------
# 💾 Save model
# -------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved -> {MODEL_PATH}")

# -------------------------------------
# 🔍 Feature importance
# -------------------------------------
imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("\n🏁 Top Feature Importances:")
print(imp.head(15))
