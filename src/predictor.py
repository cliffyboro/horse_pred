#!/usr/bin/env python
import json
import sys
from pathlib import Path
import joblib
import pandas as pd

RACECARDS_DIR = Path("racecards")
MODELS_DIR = Path("models")

def load_models():
    print("Loading prediction models...")
    models = {}
    for name in ['xgb', 'rf', 'et', 'lgb']:
        models[name] = joblib.load(MODELS_DIR / f"{name}_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    print("✅ Models loaded\n")
    return models, scaler


def load_racecard(date_str=None):
    if date_str:
        path = RACECARDS_DIR / f"{date_str}.json"
    else:
        files = sorted(RACECARDS_DIR.glob("*.json"), reverse=True)
        path = files[0]

    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    
    date = path.stem
    print(f"📅 Loaded racecards: {date}\n")
    return data, date


def display_meetings(data):
    print("="*100)
    print("                     AVAILABLE MEETINGS")
    print("="*100)

    selection_map = {}
    counter = 1

    for region, courses in data.items():
        prefix = "G" if region.upper() == "GB" else "H"
        
        for course, races in courses.items():
            print(f"\n🏟️  {course.upper()} ({region})")
            print("-" * 80)
            
            for time, race in sorted(races.items()):
                runners = len(race.get("runners", []))
                dist = race.get("distance", "?")
                rclass = race.get("race_class", "?")
                rname = race.get("race_name", "")[:60]
                
                code = f"{prefix}{counter}"
                selection_map[code] = (region, course, time, race)
                
                print(f"  {code:>4} | {time} | {dist:>6} | {runners:>2} runners | "
                      f"Class {rclass} | {rname}")
                counter += 1

    return selection_map


def main():
    data, date = load_racecard()
    selection_map = display_meetings(data)
    models, scaler = load_models()

    print("\n" + "="*100)
    print("Enter codes: G5 H3 G12   or type 'ALL'   (Q to quit)")
    print("="*100)

    while True:
        choice = input("\nSelect races → ").strip().upper()
        if choice in ['Q', 'EXIT']:
            break

        selected = []
        if choice == 'ALL':
            selected = list(selection_map.values())
        else:
            for code in choice.replace(',', ' ').split():
                if code in selection_map:
                    selected.append(selection_map[code])
                else:
                    print(f"   ⚠️  Unknown code: {code}")

        if not selected:
            continue

        print(f"\n🔮 Predicting {len(selected)} race(s)...\n")

        for _, course, time, race in selected:
            print(f"🏇 {course} - {time} | {race.get('race_name')[:70]}")
            
            # Placeholder predictions (we'll improve this next)
            runners = race.get("runners", [])
            # Sort randomly for now - replace with real model later
            top3 = sorted(runners, key=lambda x: x.get('rpr', 0) or 0, reverse=True)[:3]
            
            for i, runner in enumerate(top3, 1):
                name = runner.get('name', 'Unknown')
                num = runner.get('number', '?')
                print(f"   {i}. #{num} {name:<28} → High Confidence")
            print("-" * 85)

        print("\n✅ Prediction round complete.\n")


if __name__ == "__main__":
    main()