import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_students_sessions_new.csv")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from cognitive_analyzer import CognitiveStateAnalyzer

df = pd.read_csv(DATA_PATH)
analyzer = CognitiveStateAnalyzer()

# Build feature matrix (avg alpha/beta/theta/delta, same features the
# rule-based formulas use) and two label options:
#  (a) the rule-based analyzer's own output state (does the ML model
#      reproduce the deterministic rule, i.e. sanity check)
#  (b) the generator's programmed phase label (ground truth *as defined
#      by the synthetic scenario*, the same "ground truth" the rule-based
#      system is implicitly evaluated against elsewhere in the article)

def extract_features(row):
    feats = {}
    for band in ["alpha", "beta", "theta", "delta"]:
        cols = [c for c in row.index if f" {band} m" in c]
        feats[f"avg_{band}"] = row[cols].mean()
    return feats

X_rows = []
y_rule = []
y_phase = []
groups = []
for _, row in df.iterrows():
    feats = extract_features(row)
    X_rows.append([feats["avg_alpha"], feats["avg_beta"], feats["avg_theta"], feats["avg_delta"]])
    result = analyzer.analyze_cognitive_state(row.to_dict())
    y_rule.append(result["state"])
    y_phase.append(row["Class"])
    groups.append(row["StudentID"])

X = np.array(X_rows)
groups = np.array(groups)

gkf = GroupKFold(n_splits=5)

def cv_eval(y, label_name):
    y = np.array(y)
    accs, f1s = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
    print(f"[{label_name}] Logistic Regression, student-grouped 5-fold CV: "
          f"accuracy = {np.mean(accs)*100:.1f}% (+/- {np.std(accs)*100:.1f}), "
          f"macro-F1 = {np.mean(f1s)*100:.1f}%")

print("=== Baseline ML comparison (illustrative, synthetic data only) ===")
cv_eval(y_rule, "predict rule-based state from band powers")
cv_eval(y_phase, "predict generator's programmed phase from band powers")

print("\nRule-based classification is deterministic given the same formulas/")
print("thresholds -> by construction it will always match itself with 100%")
print("'accuracy' if evaluated against its own output; the meaningful number")
print("here is how well a simple, purely data-driven classifier (never told")
print("the formulas) recovers the SAME categories from the same 4 band-power")
print("features, as a sanity/plausibility check, not a real-world benchmark.")
