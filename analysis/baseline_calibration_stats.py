import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_students_sessions_new.csv")

import pandas as pd
import numpy as np
from scipy import stats
from cognitive_analyzer import CognitiveStateAnalyzer

BASELINE_MINUTES = 3

df = pd.read_csv(DATA_PATH)
analyzer = CognitiveStateAnalyzer()
PROFILE_MAP = {"good_profile": "good", "moderate_profile": "moderate", "fatigue_profile": "fatigue"}

records = []
for student_id, g in df.groupby("StudentID"):
    g = g.sort_values("SessionMinute")

    # Baseline: mean RAW score over the student's own first BASELINE_MINUTES.
    baseline_att, baseline_drow = [], []
    for _, row in g.head(BASELINE_MINUTES).iterrows():
        feats = analyzer._extract_features(row.to_dict())
        baseline_att.append(analyzer._calculate_attention_formula(feats))
        baseline_drow.append(analyzer._calculate_drowsiness_formula(feats))
    baseline_att = np.mean(baseline_att)
    baseline_drow = np.mean(baseline_drow)

    for _, row in g.iterrows():
        feats = analyzer._extract_features(row.to_dict())
        raw_att = analyzer._calculate_attention_formula(feats)
        raw_drow = analyzer._calculate_drowsiness_formula(feats)
        records.append({
            "StudentID": student_id,
            "SessionMinute": row["SessionMinute"],
            "Profile": PROFILE_MAP[row["ScenarioTemplate"]],
            "RawAttention": raw_att,
            "RawDrowsiness": raw_drow,
            # Signed, UNCLIPPED deviation from the student's own baseline --
            # this is what we report descriptively (no artificial floor/ceiling).
            "DeltaAttention": raw_att - baseline_att,
            "DeltaDrowsiness": raw_drow - baseline_drow,
        })

res = pd.DataFrame(records)
first_min, last_min = res["SessionMinute"].min(), res["SessionMinute"].max()

def start_end(d, col):
    s = d[d.SessionMinute == first_min][col]
    e = d[d.SessionMinute == last_min][col]
    return s.mean(), s.std(ddof=1), e.mean(), e.std(ddof=1)

print(f"=== Individual-baseline deviation (Delta = raw - own first-{BASELINE_MINUTES}-min mean), signed, unclipped ===\n")
print("--- Pooled (n=10), end of session only (start is ~0 by construction) ---")
for label, col in [("Attention", "DeltaAttention"), ("Drowsiness", "DeltaDrowsiness")]:
    _, _, em, es = start_end(res, col)
    print(f"Delta {label} at end of session: {em:+.1f} ± {es:.1f} percentage points from own baseline")

r, p = stats.pearsonr(res["DeltaAttention"], res["DeltaDrowsiness"])
print(f"Correlation (DeltaAttention vs DeltaDrowsiness, all points): r = {r:.2f}, p = {p:.2e}")

print("\n--- By profile (end-of-session delta from own baseline) ---")
for profile, n in [("good", 3), ("moderate", 3), ("fatigue", 4)]:
    d = res[res.Profile == profile]
    _, _, em_a, es_a = start_end(d, "DeltaAttention")
    _, _, em_d, es_d = start_end(d, "DeltaDrowsiness")
    print(f"{profile} (n={n}): Delta Attention {em_a:+.1f}±{es_a:.1f} pts; Delta Drowsiness {em_d:+.1f}±{es_d:.1f} pts")

print("\n--- For reference, raw absolute values (already in article) ---")
for profile, n in [("good", 3), ("moderate", 3), ("fatigue", 4)]:
    d = res[res.Profile == profile]
    sm, ss, em, es = start_end(d, "RawAttention")
    print(f"{profile} raw Attention: {sm:.1f}±{ss:.1f} -> {em:.1f}±{es:.1f}")
