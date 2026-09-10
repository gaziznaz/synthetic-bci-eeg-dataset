import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_students_sessions_new.csv")

import pandas as pd
import numpy as np
from cognitive_analyzer import CognitiveStateAnalyzer

df = pd.read_csv(DATA_PATH)
analyzer = CognitiveStateAnalyzer()

records = []
for _, row in df.iterrows():
    eeg_data = row.to_dict()
    result = analyzer.analyze_cognitive_state(eeg_data)
    records.append({
        "StudentID": row["StudentID"],
        "SessionMinute": row["SessionMinute"],
        "ScenarioTemplate": row["ScenarioTemplate"],
        "ProgrammedPhase": row["Class"],
        "Attention": result["attention"],
        "Workload": result["workload"],
        "Drowsiness": result["drowsiness"],
        "AnalyzerState": result["state"],
    })

res = pd.DataFrame(records)

# 1) Crosstab: programmed phase (ground truth from generator) vs analyzer's
# 8-way rule-based state (internal consistency, NOT external validation).
ct = pd.crosstab(res["ProgrammedPhase"], res["AnalyzerState"])
print("=== Crosstab: programmed phase vs analyzer state (counts) ===")
print(ct)
print()

# 2) "Correct direction" internal-consistency score: for each programmed
# phase, does the analyzer state fall in the expected direction bucket?
expected_bucket = {
    "high_attention": {"high_attention", "good_attention", "normal"},
    "medium_attention": {"good_attention", "normal", "low_attention", "high_workload"},
    "low_attention": {"low_attention", "very_low_attention", "normal", "high_workload"},
    "drowsy": {"high_drowsiness", "critical_drowsiness", "very_low_attention", "low_attention"},
}

res["Match"] = res.apply(
    lambda r: r["AnalyzerState"] in expected_bucket[r["ProgrammedPhase"]], axis=1
)

by_phase = res.groupby("ProgrammedPhase")["Match"].mean() * 100
overall = res["Match"].mean() * 100

print("=== Internal-consistency match rate by programmed phase (%) ===")
print(by_phase.round(1))
print(f"\nOverall match rate: {overall:.1f}%  (n={len(res)} minute-level samples, {df['StudentID'].nunique()} students)")
