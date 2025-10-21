#!/usr/bin/env python3
"""
predict.py
- Load racecards.json (daily scraped racecards)
- Show meetings table (Rich) with globally numbered races
- Ask user which races to predict (e.g., 1,2,3 or all)
- For each selected race, fetch historical aggregates from raceform.db,
  compute recent-form feature from racecards.json, create feature vector,
  predict with race_predictor.pkl, and show Top 3 + an "outsider".
"""

import json
import sqlite3
import re
from pathlib import Path
import math
import pandas as pd
import numpy as np
import joblib
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from sklearn.ensemble import RandomForestRegressor

console = Console()

# ---------- CONFIG ----------
DB_PATH = "raceform.db"
RACECARD_PATH = "racecards.json"
MODEL_PATH = "race_predictor.pkl"

FEATURES = [
    "or", "rpr", "ts", "age", "btn", "ovr_btn",
    "horse_rpr_avg", "horse_ts_avg", "horse_btn_avg",
    "trainer_rpr_avg", "trainer_ts_avg", "trainer_btn_avg",
    "jockey_rpr_avg", "jockey_ts_avg", "jockey_btn_avg",
    "days_ago", "horse_going_dist_winrate"
]

ALPHA = 0.75  # model weight
RECENT_FORM_WEIGHT = 0.6
DIST_GOING_WEIGHT = 0.4
OUTSIDER_SCORE_MARGIN = 0.08
OUTSIDER_OR_DELTA = -5

# ---------------- utilities ----------------
def load_racecards(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_meetings_table(meetings):
    cols = []
    races_by_meeting = []
    max_rows = 0
    for m in meetings:
        course = m.get("course", "Unknown")
        races = sorted(m.get("races", []), key=lambda r: r.get("off",""))
        races_by_meeting.append(races)
        formatted = [f"({i+1}) {r.get('off','')} - {r.get('dist','')}" for i, r in enumerate(races)]
        cols.append((course, formatted))
        max_rows = max(max_rows, len(formatted))

    # pad columns
    for idx, (course, formatted) in enumerate(cols):
        cols[idx] = (course, formatted + [""] * (max_rows - len(formatted)))

    table = Table(title="Races by Meeting", show_header=True, header_style="bold magenta")
    for course, _ in cols:
        table.add_column(course, style="green")

    mapping = {}
    global_counter = 1
    for row in range(max_rows):
        row_vals = []
        for m_idx, (course, formatted) in enumerate(cols):
            val = formatted[row]
            if val:
                mapping[global_counter] = (m_idx, row)
                row_vals.append(val)
                global_counter += 1
            else:
                row_vals.append("")
        table.add_row(*row_vals)

    return table, mapping, races_by_meeting

def parse_form_to_score(form_str, max_positions=6):
    if not form_str or not isinstance(form_str, str):
        return 0.5
    tokens = re.findall(r"\d+|[A-Za-z]{1,3}", form_str)
    vals = []
    for t in tokens:
        if t.isdigit():
            v = int(t)
            if v == 0:
                v = 10
            vals.append(v)
        else:
            vals.append(12)
    vals = vals[-max_positions:]
    avg_pos = sum(vals)/len(vals)
    avg_pos = max(1.0, min(12.0, avg_pos))
    score = 1.0 - (avg_pos-1)/(12-1)
    return float(score)

def get_scalar(conn, query, params=()):
    cur = conn.cursor()
    cur.execute(query, params)
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None

def avg_metric(conn, field, name_value):
    if field not in ("rpr", "ts", "btn", "ovr_btn", "or"):
        return None
    cur = conn.cursor()
    q = f"SELECT AVG([{field}]) FROM data WHERE horse LIKE ?"
    cur.execute(q, (f"%{name_value}%",))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None

def avg_metric_by_role(conn, field, role_col, role_name):
    if field not in ("rpr", "ts", "btn", "ovr_btn", "or"):
        return None
    cur = conn.cursor()
    q = f"SELECT AVG([{field}]) FROM data WHERE {role_col} LIKE ?"
    cur.execute(q, (f"%{role_name}%",))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None

def horse_days_since_last_run(conn, horse_name):
    cur = conn.cursor()
    q = "SELECT MAX(date) FROM data WHERE horse LIKE ?"
    cur.execute(q, (f"%{horse_name}%",))
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    last = pd.to_datetime(row[0], errors="coerce", utc=True)
    if pd.isna(last):
        return None
    now = pd.Timestamp.now(tz="UTC")
    return (now - last).days

def horse_going_dist_winrate(conn, horse_name, going, dist):
    cur = conn.cursor()
    q_count = "SELECT COUNT(*) FROM data WHERE horse LIKE ? AND going = ? AND dist = ?"
    q_wins = "SELECT COUNT(*) FROM data WHERE horse LIKE ? AND going = ? AND dist = ? AND pos = 1"
    cur.execute(q_count, (f"%{horse_name}%", going, dist))
    c = cur.fetchone()[0] or 0
    if c==0:
        return 0.0
    cur.execute(q_wins, (f"%{horse_name}%", going, dist))
    w = cur.fetchone()[0] or 0
    return float(w)/float(c)

def safe_float(x, fallback=0.0):
    try:
        return float(x)
    except Exception:
        return float(fallback)

# ---------------- Model loading ----------------
def train_dummy_model():
    X = pd.DataFrame(np.random.rand(200, len(FEATURES)), columns=FEATURES)
    y = np.random.rand(200)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    console.print("[yellow]⚠️ race_predictor.pkl not found — using a dummy model for demonstration.[/yellow]")
    return model

# ---------------- main ----------------
def main():
    # Load model or dummy fallback
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        console.print(f"[green]✅ Loaded trained model from {MODEL_PATH}[/green]")
        console.print(f"[blue]Features used in model: {FEATURES}[/blue]")
    else:
        model = train_dummy_model()

    # Load racecards
    rc = load_racecards(RACECARD_PATH)
    meetings = rc.get("meetings", rc) if isinstance(rc, dict) else rc
    if not meetings:
        console.print("[red]No meetings found in racecards.json[/red]")
        return

    # Build table
    table, mapping, races_by_meeting = build_meetings_table(meetings)
    console.print(table)

    # Prompt user
    choices = Prompt.ask("Enter row numbers to predict (e.g., 1,2,3 or 'all') - rows numbered left-to-right across all meetings")
    choices = choices.strip()
    if choices.lower() == "all":
        selected_indices = list(mapping.keys())
    else:
        parts = re.split(r"[,\s]+", choices)
        selected_indices = []
        for p in parts:
            if "-" in p:
                try:
                    a,b = p.split("-",1)
                    selected_indices.extend(range(int(a), int(b)+1))
                except: pass
            elif p.isdigit():
                selected_indices.append(int(p))
    selected_indices = sorted(set([i for i in selected_indices if i in mapping]))
    if not selected_indices:
        console.print("[yellow]No valid races selected. Exiting.[/yellow]")
        return

    # Connect DB
    conn = sqlite3.connect(DB_PATH)

    # Predict each selected race
    for idx in selected_indices:
        m_idx, r_idx = mapping[idx]
        meeting = meetings[m_idx]
        race = races_by_meeting[m_idx][r_idx]

        course = meeting.get("course", meeting.get("name","Unknown"))
        going = race.get("going") or meeting.get("going") or ""
        off = race.get("off", "")
        dist = race.get("dist", "")

        console.print()
        console.print(Panel(f"[bold]Predicting for race:[/bold] {off} at {course} (Going: {going})", style="bold green"))

        horses = race.get("horses", [])
        predictions = []

        # median OR
        or_list = [safe_float(h.get("[or]") or h.get("or") or np.nan) for h in horses if h.get("[or]") or h.get("or")]
        median_or = float(np.median(or_list)) if or_list else None

        for h in horses:
            name = h.get("horse")
            jockey = h.get("jockey") or ""
            trainer = h.get("trainer") or ""
            age = safe_float(h.get("age",0))
            or_val = safe_float(h.get("[or]") or h.get("or") or np.nan)

            recent_form = parse_form_to_score(h.get("form",""))
            horse_rpr_avg = avg_metric(conn, "rpr", name) or 0.0
            horse_ts_avg = avg_metric(conn, "ts", name) or 0.0
            horse_btn_avg = avg_metric(conn, "btn", name) or 0.0

            trainer_rpr_avg = avg_metric_by_role(conn, "rpr", "trainer", trainer) or 0.0
            trainer_ts_avg = avg_metric_by_role(conn, "ts", "trainer", trainer) or 0.0
            trainer_btn_avg = avg_metric_by_role(conn, "btn", "trainer", trainer) or 0.0

            jockey_rpr_avg = avg_metric_by_role(conn, "rpr", "jockey", jockey) or 0.0
            jockey_ts_avg = avg_metric_by_role(conn, "ts", "jockey", jockey) or 0.0
            jockey_btn_avg = avg_metric_by_role(conn, "btn", "jockey", jockey) or 0.0

            days_ago = horse_days_since_last_run(conn, name) or 365
            h_gd_winrate = horse_going_dist_winrate(conn, name, going, dist)

            rpr_val = safe_float(h.get("rpr") or horse_rpr_avg or 0.0)
            ts_val = safe_float(h.get("ts") or horse_ts_avg or 0.0)
            btn_val = safe_float(h.get("btn") or horse_btn_avg or 0.0)
            ovr_btn_val = safe_float(h.get("ovr_btn") or 0.0)

            feat = {f: locals()[f] if f in locals() else 0.0 for f in FEATURES}
            x = np.array([feat[f] for f in FEATURES], dtype=float).reshape(1,-1)
            model_score = float(model.predict(x)[0])
            short_term = RECENT_FORM_WEIGHT*recent_form + DIST_GOING_WEIGHT*h_gd_winrate
            final_score = ALPHA*model_score + (1-ALPHA)*short_term

            predictions.append({
                "horse": name, "jockey": jockey, "trainer": trainer,
                "model_score": model_score, "recent_form": recent_form,
                "short_term": short_term, "final_score": final_score,
                "or": or_val, "feat": feat, "raw": h
            })

        predictions.sort(key=lambda x: x["final_score"], reverse=True)

        # debug
        debug_lines = [f"{p['horse']} -> final:{p['final_score']:.3f} model:{p['model_score']:.3f} recent:{p['recent_form']:.3f} or:{p['or']}" for p in predictions[:6]]
        console.print(Panel("\n".join(debug_lines), title="Debug summaries (top 6 horses)", style="cyan"))

        # Top 3
        t = Table(title="Top 3 Predictions")
        t.add_column("Position", style="bold")
        t.add_column("Horse")
        t.add_column("Score")
        t.add_column("Jockey")
        t.add_column("Trainer")
        t.add_column("Form")
        for pos,p in enumerate(predictions[:3],1):
            t.add_row(str(pos), p["horse"], f"{p['final_score']:.3f}", p["jockey"], p["trainer"], p["raw"].get("form",""))
        console.print(t)

        # Outsider
        outsider = None
        third_score = predictions[2]["final_score"] if len(predictions)>=3 else (predictions[-1]["final_score"] if predictions else 0)
        for p in predictions[3:]:
            cond_nearmiss = p["final_score"] >= (third_score - OUTSIDER_SCORE_MARGIN)
            cond_low_or = median_or is not None and p["or"] <= (median_or + OUTSIDER_OR_DELTA)
            if cond_nearmiss or cond_low_or:
                outsider = p
                break

        if outsider:
            console.print(Panel(f"💡 Outsider: {outsider['horse']} (score: {outsider['final_score']:.3f}) — jockey: {outsider['jockey']}, trainer: {outsider['trainer']}", style="yellow"))
        else:
            console.print(Panel("💡 No strong outsider found", style="yellow"))

    conn.close()
    console.print("\n[green]Done.[/green]")

if __name__=="__main__":
    main()
