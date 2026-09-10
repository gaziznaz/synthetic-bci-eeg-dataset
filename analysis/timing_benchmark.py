import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_students_sessions_new.csv")

import time
import pandas as pd
from cognitive_analyzer import CognitiveStateAnalyzer

df = pd.read_csv(DATA_PATH)
analyzer = CognitiveStateAnalyzer()
rows = [row.to_dict() for _, row in df.iterrows()]

# Warm-up
for r in rows[:20]:
    analyzer.analyze_cognitive_state(r)

# Measure single-point processing time (software layer only: feature
# extraction + formulas + rule classification -- NOT signal acquisition,
# filtering, or network/UI transport).
N_REPEATS = 200
t0 = time.perf_counter()
for _ in range(N_REPEATS):
    for r in rows:
        analyzer.analyze_cognitive_state(r)
t1 = time.perf_counter()

total_calls = N_REPEATS * len(rows)
elapsed = t1 - t0
per_call_ms = (elapsed / total_calls) * 1000

print(f"Total calls: {total_calls}")
print(f"Total time: {elapsed:.4f} s")
print(f"Per-call (per data point) processing time: {per_call_ms:.5f} ms")

# Throughput / scaling estimate: how many students' points can be processed
# per second on this machine, and what that implies for the interface's
# INTERFACE_UPDATE_MS = 3000 refresh cadence with N students.
per_call_s = per_call_ms / 1000
max_points_per_sec = 1 / per_call_s
print(f"\nMax data points processable per second (single core, this machine): {max_points_per_sec:,.0f}")

for n_students in [10, 50, 100, 500, 1000]:
    time_for_batch_ms = per_call_ms * n_students
    budget_ms = 3000
    ratio = budget_ms / time_for_batch_ms
    print(f"N={n_students:5d} students: batch compute time = {time_for_batch_ms:.3f} ms "
          f"(vs {budget_ms} ms UI refresh budget -> headroom x{ratio:,.0f})")
