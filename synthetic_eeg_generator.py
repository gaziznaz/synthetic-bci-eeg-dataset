from __future__ import annotations

import argparse
import json
import math
import platform
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

N_STUDENTS = 10
SESSION_MINUTES = 30

# One CSV row represents one simulated minute.
SIMULATION_STEP_SECONDS = 60

# Existing interface playback interval.
# This is NOT the temporal resolution of the synthetic EEG data.
INTERFACE_UPDATE_MS = 3000

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

BANDS = ["delta", "theta", "alpha", "beta"]

DATASET_REFERENCE = (
    "EEG Dataset for Emotion Classification Using Low-Cost and High-End "
    "Equipment, Zenodo, 2025, DOI: 10.5281/zenodo.14787743. "
    "Used only as a structural / approximate-scale reference for the "
    "synthetic Emotiv EPOC+-shaped data; no emotion labels are used."
)


# ============================================================
# WORKING RANGES
# ============================================================
#
# These are simulation working ranges, not physiological norms.
#
# They are simulation working ranges selected so the unchanged
# CognitiveStateAnalyzer and existing UI can still exercise the intended
# interface states. They are NOT physiological ground-truth intervals and
# are NOT claimed to be statistics extracted from the public dataset.
# The generator never calls the analyzer and never accepts/rejects a run
# based on analyzer output.
# ============================================================

STATE_RANGES = {
    "high_attention": {
        "alpha": (14.0, 17.5),
        "beta": (42.5, 46.0),
        "theta": (8.0, 11.0),
        "delta": (2000.0, 3000.0),
    },
    "medium_attention": {
        "alpha": (20.0, 24.0),
        "beta": (39.0, 43.0),
        "theta": (12.0, 16.0),
        "delta": (2000.0, 3000.0),
    },
    "low_attention": {
        "alpha": (27.0, 30.0),
        "beta": (36.0, 40.0),
        "theta": (19.0, 24.0),
        "delta": (2000.0, 3000.0),
    },
    "drowsy": {
        # Recalibrated so the Attention formula (2)-(3) does not saturate at
        # the 0 floor for every subject in this state: an earlier version of
        # these ranges (alpha 32-40, beta 22-28, theta 42-50) pushed
        # Attention below 0 for essentially all students, collapsing all
        # individual variability to an identical 0.0 +/- 0.0 at session end.
        # These ranges keep Drowsiness clearly the highest of the four
        # states (theta/alpha and theta/beta both elevated) while leaving
        # Attention a small positive band with real spread.
        "alpha": (28.0, 32.0),
        "beta": (28.0, 34.0),
        "theta": (36.0, 44.0),
        "delta": (2000.0, 3000.0),
    },
}

GLOBAL_RANGES = {
    "alpha": (10.0, 75.0),
    "beta": (15.0, 85.0),
    "theta": (5.0, 85.0),
    "delta": (1800.0, 3200.0),
}


# ============================================================
# PROFILE FAMILIES
# ============================================================
#
# The original prototype had 3 broad patterns:
#   1-3: mostly good attention
#   4-6: moderate decline
#   7-10: fatigue / drowsiness
#
# Here we preserve the same overall class composition (3/3/4), but shuffle
# the profile families between StudentIDs on every unseeded run.
# Thus student_10 is NOT permanently tied to the same trajectory.
# ============================================================

PROFILE_POOL = (
    ["good_profile"] * 3
    + ["moderate_profile"] * 3
    + ["fatigue_profile"] * 4
)

# Nominal phase lengths. They are jittered per student while keeping 30 min.
NOMINAL_PHASES = {
    "good_profile": [
        # Was high -> medium -> LOW, which meant this profile actually
        # ended the session classified as "low_attention" by the analyzer
        # (see review notes) -- indistinguishable from moderate_profile's
        # ending and contradicting the "relatively stable attention"
        # description in the article. Now dips to medium mid-session and
        # recovers to high by the end, so it stays a genuinely distinct,
        # non-declining trajectory.
        ("high_attention", 16),
        ("medium_attention", 9),
        ("high_attention", 5),
    ],
    "moderate_profile": [
        ("high_attention", 8),
        ("medium_attention", 12),
        ("low_attention", 10),
    ],
    "fatigue_profile": [
        ("medium_attention", 5),
        ("low_attention", 15),
        ("drowsy", 10),
    ],
}

PHASE_JITTER_MINUTES = 2
MIN_PHASE_MINUTES = 3


# ============================================================
# TEMPORAL / INDIVIDUAL VARIABILITY
# ============================================================

# Student-level band scaling, fixed for the whole session.
PROFILE_BAND_SCALE_SIGMA = 0.035
PROFILE_BAND_SCALE_CLIP = (0.92, 1.08)

# Channel-specific scaling, fixed for the whole session.
CHANNEL_SCALE_SIGMA = 0.015
CHANNEL_SCALE_CLIP = (0.96, 1.04)

# Slight per-student/per-state center variation.
STATE_CENTER_JITTER_FRAC = 0.035

# Shared temporal correlation for each band.
TEMPORAL_RHO_RANGE = (0.60, 0.75)

# Small slow common variation shared by all 14 channels of a band.
COMMON_NOISE_FRAC_RANGE = (0.004, 0.009)

# Small channel-specific correlated residual.
CHANNEL_NOISE_FRAC_RANGE = (0.002, 0.006)
CHANNEL_NOISE_RHO_RANGE = (0.45, 0.70)

# Maximum change per minute, relative to the student's baseline for a band.
MAX_COMMON_STEP_FRAC = 0.100
MAX_CHANNEL_STEP_FRAC = 0.120

# Local within-segment sample used to calculate m and std.
LOCAL_SAMPLE_SIZE = 20
WITHIN_SEGMENT_SIGMA_FRAC_RANGE = (0.020, 0.045)


@dataclass
class StudentProfile:
    student_id: int
    profile_type: str
    phases: list[tuple[str, int]]
    state_centers: dict
    band_scale: dict
    channel_scale: dict
    temporal_rho: float
    common_noise_frac: float
    channel_noise_frac: float
    channel_noise_rho: float
    within_segment_sigma_frac: float


# ============================================================
# HELPERS
# ============================================================

def clip(value: float, limits: tuple[float, float]) -> float:
    return float(np.clip(value, limits[0], limits[1]))


def state_midpoint(state: str, band: str) -> float:
    low, high = STATE_RANGES[state][band]
    return (low + high) / 2.0


def state_span(state: str, band: str) -> float:
    low, high = STATE_RANGES[state][band]
    return high - low


def clamp_step(previous: float, proposed: float, max_step: float) -> float:
    return float(np.clip(
        proposed,
        previous - max_step,
        previous + max_step
    ))


def assign_profile_types(rng: np.random.Generator) -> dict[int, str]:
    pool = list(PROFILE_POOL)
    rng.shuffle(pool)
    return {
        student_id: pool[student_id - 1]
        for student_id in range(1, N_STUDENTS + 1)
    }


def randomize_phases(
    rng: np.random.Generator,
    profile_type: str
) -> list[tuple[str, int]]:
    """
    Slightly varies phase boundaries while keeping:
      - the same phase order,
      - every phase >= MIN_PHASE_MINUTES,
      - total duration exactly 30 minutes.
    """
    nominal = NOMINAL_PHASES[profile_type]

    if len(nominal) != 3:
        raise ValueError("This generator expects exactly 3 phases per profile.")

    while True:
        d1 = nominal[0][1] + int(
            rng.integers(-PHASE_JITTER_MINUTES, PHASE_JITTER_MINUTES + 1)
        )
        d2 = nominal[1][1] + int(
            rng.integers(-PHASE_JITTER_MINUTES, PHASE_JITTER_MINUTES + 1)
        )
        d3 = SESSION_MINUTES - d1 - d2

        durations = [d1, d2, d3]

        if all(d >= MIN_PHASE_MINUTES for d in durations):
            return [
                (nominal[0][0], d1),
                (nominal[1][0], d2),
                (nominal[2][0], d3),
            ]


def expand_state_schedule(phases: list[tuple[str, int]]) -> list[str]:
    schedule: list[str] = []

    for state, duration in phases:
        schedule.extend([state] * duration)

    if len(schedule) != SESSION_MINUTES:
        raise RuntimeError(
            f"Schedule length is {len(schedule)}, expected {SESSION_MINUTES}."
        )

    return schedule


# ============================================================
# PROFILE CREATION
# ============================================================

def build_student_profile(
    rng: np.random.Generator,
    student_id: int,
    profile_type: str,
) -> StudentProfile:

    phases = randomize_phases(rng, profile_type)

    band_scale = {
        band: clip(
            rng.normal(1.0, PROFILE_BAND_SCALE_SIGMA),
            PROFILE_BAND_SCALE_CLIP
        )
        for band in BANDS
    }

    channel_scale = {
        channel: {
            band: clip(
                rng.normal(1.0, CHANNEL_SCALE_SIGMA),
                CHANNEL_SCALE_CLIP
            )
            for band in BANDS
        }
        for channel in CHANNELS
    }

    # Individual state centers.
    # Centers are selected once per student and reused throughout the session.
    state_centers = {}

    states_used = list(dict.fromkeys(state for state, _ in phases))

    for state in states_used:
        state_centers[state] = {}

        for band in BANDS:
            midpoint = state_midpoint(state, band)
            span = state_span(state, band)

            jitter = rng.normal(
                0.0,
                span * STATE_CENTER_JITTER_FRAC
            )

            value = (midpoint + jitter) * band_scale[band]

            # Keep personalized center inside the state's working range.
            state_centers[state][band] = clip(
                value,
                STATE_RANGES[state][band]
            )

    return StudentProfile(
        student_id=student_id,
        profile_type=profile_type,
        phases=phases,
        state_centers=state_centers,
        band_scale=band_scale,
        channel_scale=channel_scale,
        temporal_rho=float(rng.uniform(*TEMPORAL_RHO_RANGE)),
        common_noise_frac=float(rng.uniform(*COMMON_NOISE_FRAC_RANGE)),
        channel_noise_frac=float(rng.uniform(*CHANNEL_NOISE_FRAC_RANGE)),
        channel_noise_rho=float(rng.uniform(*CHANNEL_NOISE_RHO_RANGE)),
        within_segment_sigma_frac=float(
            rng.uniform(*WITHIN_SEGMENT_SIGMA_FRAC_RANGE)
        ),
    )


# ============================================================
# SMOOTH BAND PATH
# ============================================================

def build_smoothed_targets(
    profile: StudentProfile,
    schedule: list[str],
    band: str,
) -> np.ndarray:
    """
    Build a smooth target trajectory through the centres of successive phases.

    The categorical simulation label can change at a phase boundary, while
    the spectral trajectory changes gradually across several minutes. This
    avoids abrupt state-to-state jumps and is closer to the behaviour expected
    from a slowly evolving EEG-derived feature stream.
    """
    phase_centres = []
    cursor = 0

    for state, duration in profile.phases:
        centre_t = cursor + (duration - 1) / 2.0
        phase_centres.append(
            (centre_t, profile.state_centers[state][band])
        )
        cursor += duration

    anchors = [
        (0.0, phase_centres[0][1]),
        *phase_centres,
        (float(SESSION_MINUTES - 1), phase_centres[-1][1]),
    ]

    # Remove duplicate anchor times while preserving order.
    clean_anchors = []
    for t, value in anchors:
        if clean_anchors and abs(t - clean_anchors[-1][0]) < 1e-9:
            clean_anchors[-1] = (t, value)
        else:
            clean_anchors.append((t, value))

    targets = np.zeros(SESSION_MINUTES, dtype=float)

    for minute in range(SESSION_MINUTES):
        t = float(minute)

        if t <= clean_anchors[0][0]:
            targets[minute] = clean_anchors[0][1]
            continue

        if t >= clean_anchors[-1][0]:
            targets[minute] = clean_anchors[-1][1]
            continue

        for i in range(len(clean_anchors) - 1):
            t0, v0 = clean_anchors[i]
            t1, v1 = clean_anchors[i + 1]

            if t0 <= t <= t1:
                x = (t - t0) / max(t1 - t0, 1e-9)
                smooth = x * x * (3.0 - 2.0 * x)  # smoothstep
                targets[minute] = v0 + (v1 - v0) * smooth
                break

    return targets


def build_common_band_path(
    rng: np.random.Generator,
    profile: StudentProfile,
    band: str,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Shared band-level trajectory for all 14 channels.

    High temporal correlation + small innovation gives a smooth curve.
    """
    series = np.zeros(SESSION_MINUTES, dtype=float)

    baseline = targets[0]
    rho = profile.temporal_rho
    sigma = max(
        baseline * profile.common_noise_frac,
        1e-9
    )

    series[0] = targets[0] + rng.normal(0.0, sigma * 0.5)

    for t in range(1, SESSION_MINUTES):
        innovation = rng.normal(0.0, sigma)

        proposed = (
            rho * series[t - 1]
            + (1.0 - rho) * targets[t]
            + innovation
        )

        max_step = max(
            abs(series[t - 1]) * MAX_COMMON_STEP_FRAC,
            sigma * 2.0,
            1e-9
        )

        proposed = clamp_step(
            series[t - 1],
            proposed,
            max_step
        )

        series[t] = clip(
            proposed,
            GLOBAL_RANGES[band]
        )

    return series


def build_channel_path(
    rng: np.random.Generator,
    profile: StudentProfile,
    band: str,
    channel: str,
    common_path: np.ndarray,
) -> np.ndarray:
    """
    Adds a small stable channel offset and a small correlated channel residual.

    Channels follow the same overall band trajectory but are not copies.
    """
    coeff = profile.channel_scale[channel][band]
    baseline = common_path[0]

    noise_sigma = max(
        baseline * profile.channel_noise_frac,
        1e-9
    )

    residual = np.zeros(SESSION_MINUTES, dtype=float)
    residual[0] = rng.normal(0.0, noise_sigma)

    rho = profile.channel_noise_rho

    for t in range(1, SESSION_MINUTES):
        residual[t] = (
            rho * residual[t - 1]
            + rng.normal(
                0.0,
                noise_sigma * math.sqrt(max(1.0 - rho ** 2, 1e-6))
            )
        )

    raw = common_path * coeff + residual

    output = np.zeros(SESSION_MINUTES, dtype=float)
    output[0] = clip(raw[0], GLOBAL_RANGES[band])

    for t in range(1, SESSION_MINUTES):
        max_step = max(
            abs(output[t - 1]) * MAX_CHANNEL_STEP_FRAC,
            noise_sigma * 2.0,
            1e-9
        )

        value = clamp_step(
            output[t - 1],
            raw[t],
            max_step
        )

        output[t] = clip(
            value,
            GLOBAL_RANGES[band]
        )

    return output


# ============================================================
# m / std
# ============================================================

def sample_m_std(
    rng: np.random.Generator,
    central_value: float,
    band: str,
    sigma_fraction: float,
) -> tuple[float, float]:

    sigma = max(
        abs(central_value) * sigma_fraction,
        1e-9
    )

    samples = rng.normal(
        central_value,
        sigma,
        size=LOCAL_SAMPLE_SIZE
    )

    low, high = GLOBAL_RANGES[band]
    samples = np.clip(samples, low, high)

    return (
        float(np.mean(samples)),
        float(np.std(samples, ddof=1)),
    )


# ============================================================
# STUDENT DATAFRAME
# ============================================================

def generate_student_dataframe(
    rng: np.random.Generator,
    profile: StudentProfile,
) -> pd.DataFrame:

    schedule = expand_state_schedule(profile.phases)

    # Build the complete column dictionary first. This preserves the exact CSV
    # schema expected by the existing interface while avoiding DataFrame
    # fragmentation warnings from repeatedly inserting >100 columns.
    columns = {
        "StudentID": [profile.student_id] * SESSION_MINUTES,
        "SessionMinute": list(range(1, SESSION_MINUTES + 1)),
        "SimulatedTimeSec": [
            (minute - 1) * SIMULATION_STEP_SECONDS
            for minute in range(1, SESSION_MINUTES + 1)
        ],
        "ScenarioTemplate": [profile.profile_type] * SESSION_MINUTES,

        # Backward compatibility with the existing interface/data loader.
        # Values remain the original labels used by data_simulator.py.
        "Class": schedule,

        # Explicit simulation-state field.
        "SimulationScenario": schedule,
    }

    for band in BANDS:
        targets = build_smoothed_targets(
            profile,
            schedule,
            band
        )

        common_path = build_common_band_path(
            rng,
            profile,
            band,
            targets
        )

        for channel in CHANNELS:
            channel_path = build_channel_path(
                rng,
                profile,
                band,
                channel,
                common_path
            )

            m_values = []
            std_values = []

            for value in channel_path:
                m, std = sample_m_std(
                    rng,
                    value,
                    band,
                    profile.within_segment_sigma_frac
                )

                m_values.append(m)
                std_values.append(std)

            columns[f"{channel} {band} m"] = m_values
            columns[f"{channel} {band} std"] = std_values

    return pd.DataFrame(columns)


# ============================================================
# VALIDATION
# ============================================================

def validate_student(df: pd.DataFrame, student_id: int) -> list[str]:
    warnings = []

    if len(df) != SESSION_MINUTES:
        raise RuntimeError(
            f"student_{student_id}: expected {SESSION_MINUTES} rows, got {len(df)}"
        )

    if df["SessionMinute"].tolist() != list(range(1, SESSION_MINUTES + 1)):
        raise RuntimeError(
            f"student_{student_id}: SessionMinute must be 1..{SESSION_MINUTES}"
        )

    if df.isna().any().any():
        raise RuntimeError(
            f"student_{student_id}: NaN values detected"
        )

    allowed_classes = {
        "high_attention",
        "medium_attention",
        "low_attention",
        "drowsy",
    }

    if not set(df["Class"].unique()).issubset(allowed_classes):
        raise RuntimeError(
            f"student_{student_id}: unexpected Class value"
        )

    for channel in CHANNELS:
        for band in BANDS:
            m_col = f"{channel} {band} m"
            std_col = f"{channel} {band} std"

            if m_col not in df.columns or std_col not in df.columns:
                raise RuntimeError(
                    f"student_{student_id}: missing {channel}/{band} columns"
                )

            if (df[std_col] < 0).any():
                raise RuntimeError(
                    f"student_{student_id}: negative std in {std_col}"
                )

            low, high = GLOBAL_RANGES[band]

            if not df[m_col].between(low, high).all():
                raise RuntimeError(
                    f"student_{student_id}: values outside range in {m_col}"
                )

    # Soft check on the 14-channel band means used by CognitiveStateAnalyzer.
    for band in ("alpha", "beta", "theta"):
        cols = [f"{ch} {band} m" for ch in CHANNELS]
        series = df[cols].mean(axis=1).to_numpy(dtype=float)

        relative_steps = (
            np.abs(np.diff(series))
            / np.maximum(np.abs(series[:-1]), 1e-9)
        )

        max_step = float(relative_steps.max())

        if max_step > 0.10:
            warnings.append(
                f"student_{student_id}: {band} mean has a "
                f"{max_step:.1%} minute-to-minute change"
            )

    return warnings


# ============================================================
# METADATA
# ============================================================

def profile_metadata(profile: StudentProfile) -> dict:
    return {
        "student_id": profile.student_id,
        "profile_type": profile.profile_type,
        "phases": profile.phases,
        "state_centers": profile.state_centers,
        "band_scale": profile.band_scale,
        "channel_scale": profile.channel_scale,
        "temporal_rho": profile.temporal_rho,
        "common_noise_frac": profile.common_noise_frac,
        "channel_noise_frac": profile.channel_noise_frac,
        "channel_noise_rho": profile.channel_noise_rho,
        "within_segment_sigma_frac": profile.within_segment_sigma_frac,
    }


def save_metadata(
    run_dir: Path,
    seed: int,
    assignments: dict[int, str],
    profiles: list[StudentProfile],
) -> None:

    metadata = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "reproduce_with": f"python {Path(__file__).name} --seed {seed}",
        "dataset_reference": DATASET_REFERENCE,

        "n_students": N_STUDENTS,
        "session_minutes": SESSION_MINUTES,
        "rows_per_student": SESSION_MINUTES,
        "simulation_step_seconds": SIMULATION_STEP_SECONDS,
        "interface_update_ms": INTERFACE_UPDATE_MS,

        "channels": CHANNELS,
        "bands": BANDS,
        "state_ranges": STATE_RANGES,
        "global_ranges": GLOBAL_RANGES,

        "profile_pool": PROFILE_POOL,
        "profile_assignments": assignments,

        "temporal_parameters": {
            "temporal_rho_range": TEMPORAL_RHO_RANGE,
            "common_noise_frac_range": COMMON_NOISE_FRAC_RANGE,
            "channel_noise_frac_range": CHANNEL_NOISE_FRAC_RANGE,
            "channel_noise_rho_range": CHANNEL_NOISE_RHO_RANGE,
            "max_common_step_frac": MAX_COMMON_STEP_FRAC,
            "max_channel_step_frac": MAX_CHANNEL_STEP_FRAC,
        },

        "profiles": [
            profile_metadata(profile)
            for profile in profiles
        ],

        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }

    with (run_dir / "generator_metadata.json").open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate smooth synthetic Emotiv EPOC+-shaped spectral EEG "
            "data compatible with the existing BCI prototype."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Fixed seed for exact reproducibility. "
            "If omitted, a new seed is generated."
        ),
    )

    parser.add_argument(
        "--output-root",
        default="synthetic_eeg_output",
        help="Root directory for generated runs.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    seed = (
        int(args.seed)
        if args.seed is not None
        else secrets.randbits(63)
    )

    rng = np.random.default_rng(seed)

    assignments = assign_profile_types(rng)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        Path(args.output_root)
        / f"run_{timestamp}_seed_{seed}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    profiles = []
    student_frames = []
    all_warnings = []

    for student_id in range(1, N_STUDENTS + 1):
        profile = build_student_profile(
            rng=rng,
            student_id=student_id,
            profile_type=assignments[student_id],
        )

        df = generate_student_dataframe(
            rng=rng,
            profile=profile,
        )

        warnings = validate_student(
            df,
            student_id
        )

        profiles.append(profile)
        student_frames.append(df)
        all_warnings.extend(warnings)

        student_path = (
            run_dir
            / f"student_{student_id}_session.csv"
        )

        df.to_csv(
            student_path,
            index=False,
            float_format="%.6f",
        )

        print(
            f"[OK] student_{student_id}: "
            f"profile={profile.profile_type}, "
            f"phases={profile.phases}"
        )

    all_df = pd.concat(
        student_frames,
        ignore_index=True
    )

    if len(all_df) != N_STUDENTS * SESSION_MINUTES:
        raise RuntimeError(
            f"Expected 300 total rows, got {len(all_df)}"
        )

    all_df.to_csv(
        run_dir / "all_students_sessions.csv",
        index=False,
        float_format="%.6f",
    )

    save_metadata(
        run_dir=run_dir,
        seed=seed,
        assignments=assignments,
        profiles=profiles,
    )

    with (run_dir / "requirements_generated.txt").open(
        "w",
        encoding="utf-8"
    ) as f:
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"pandas=={pd.__version__}\n")

    print()
    print(f"[OK] seed: {seed}")
    print(f"[OK] output: {run_dir}")
    print(f"[OK] 10 student CSVs + combined CSV created")
    print(f"[OK] simulation step: {SIMULATION_STEP_SECONDS} sec")
    print(f"[OK] interface playback interval: {INTERFACE_UPDATE_MS} ms")

    if all_warnings:
        print(f"[WARN] {len(all_warnings)} smoothness warning(s):")
        for warning in all_warnings:
            print(f"  - {warning}")
    else:
        print("[OK] no suspicious band-mean jumps detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
