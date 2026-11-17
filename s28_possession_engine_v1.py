
"""
S28 — MOTOR DE POSESIÓN V1

Implementa una primera máquina de estados:
- Control candidato H/A.
- Confirmación temporal de cambios.
- Conservación del último poseedor.
- Diferenciación entre balón observado e imputado.
- Trazabilidad de decisiones.

Entradas:
    local_outputs/s27_football_signals.csv

Salidas:
    local_outputs/s28_submission_v1.csv
    local_outputs/s28_decision_trace_v1.csv
    ETAPA_28_MOTOR_POSESION_V1.md

NO usa ground truth.
NO modifica run.py.
NO modifica V4 ni S27.
"""

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    OUTPUT_DIR,
    N_VIDEO_FRAMES,
    FPS,
)

INPUT_PATH = OUTPUT_DIR / "s27_football_signals.csv"
SUBMISSION_PATH = OUTPUT_DIR / "s28_submission_v1.csv"
TRACE_PATH = OUTPUT_DIR / "s28_decision_trace_v1.csv"
REPORT_PATH = ROOT_DIR / "ETAPA_28_MOTOR_POSESION_V1.md"

CONTROL_RADIUS_M = 3.0
MIN_ADVANTAGE_M = 1.0

OBSERVED_CONFIRM_S = 0.40
IMPUTED_CONFIRM_S = 0.80

OBSERVED_CONFIRM_FRAMES = max(
    1, round(OBSERVED_CONFIRM_S * FPS)
)
IMPUTED_CONFIRM_FRAMES = max(
    1, round(IMPUTED_CONFIRM_S * FPS)
)

REQUIRED = [
    "frame",
    "relevant",
    "ball_source",
    "geometry_comparable",
    "home_distance_m",
    "away_distance_m",
    "distance_margin_home_m",
    "both_teams_within_3m",
    "neither_team_within_5m",
]


def numeric(df, name):
    return pd.to_numeric(
        df[name],
        errors="coerce",
    ).to_numpy(dtype=float)


def make_candidate(
    relevant,
    comparable,
    home_distance,
    away_distance,
    margin,
):
    n = len(relevant)
    candidate = np.full(n, "NONE", dtype=object)

    h = (
        relevant
        & comparable
        & (home_distance <= CONTROL_RADIUS_M)
        & (margin >= MIN_ADVANTAGE_M)
    )

    a = (
        relevant
        & comparable
        & (away_distance <= CONTROL_RADIUS_M)
        & (margin <= -MIN_ADVANTAGE_M)
    )

    if np.any(h & a):
        raise ValueError(
            "Control simultáneo incompatible."
        )

    candidate[h] = "H"
    candidate[a] = "A"

    return candidate


def build_engine(df):
    n = len(df)

    relevant = numeric(df, "relevant") == 1
    comparable = (
        numeric(df, "geometry_comparable") == 1
    )

    home_distance = numeric(
        df, "home_distance_m"
    )
    away_distance = numeric(
        df, "away_distance_m"
    )
    margin = numeric(
        df, "distance_margin_home_m"
    )

    source = df[
        "ball_source"
    ].fillna("NONE").astype(str).to_numpy()

    candidate = make_candidate(
        relevant,
        comparable,
        home_distance,
        away_distance,
        margin,
    )

    labels = np.full(n, "X", dtype=object)
    internal_state = np.full(
        n, "EXCLUDED", dtype=object
    )
    reasons = np.full(
        n, "OUTSIDE_GAME", dtype=object
    )
    last_confirmed = np.full(
        n, "NONE", dtype=object
    )
    pending_team_trace = np.full(
        n, "NONE", dtype=object
    )
    pending_frames_trace = np.zeros(
        n, dtype=np.int32
    )

    current_team = None
    pending_team = None
    pending_frames = 0
    pending_observed_frames = 0

    for i in range(n):

        if not relevant[i]:
            current_team = None
            pending_team = None
            pending_frames = 0
            pending_observed_frames = 0
            continue

        c = candidate[i]
        is_observed = source[i] == "OBSERVED"
        is_imputed = source[i] == "IMPUTED"

        if c == "NONE":

            pending_team = None
            pending_frames = 0
            pending_observed_frames = 0

            if current_team is None:
                labels[i] = "X"
                internal_state[i] = "UNINITIALIZED"
                reasons[i] = "NO_CONFIRMED_OWNER"

            else:
                labels[i] = current_team

                if not comparable[i]:
                    internal_state[i] = (
                        "LOOSE_AFTER_" + current_team
                    )
                    reasons[i] = "NO_GEOMETRY_KEEP_OWNER"

                elif (
                    numeric(
                        df.iloc[[i]],
                        "both_teams_within_3m"
                    )[0] == 1
                ):
                    internal_state[i] = (
                        "CONTESTED_AFTER_" + current_team
                    )
                    reasons[i] = "CONTESTED_KEEP_OWNER"

                else:
                    internal_state[i] = (
                        "LOOSE_AFTER_" + current_team
                    )
                    reasons[i] = "NO_CONTROL_KEEP_OWNER"

        elif c == current_team:

            pending_team = None
            pending_frames = 0
            pending_observed_frames = 0

            labels[i] = current_team
            internal_state[i] = (
                "CONTROL_" + current_team
            )
            reasons[i] = "OWNER_CONTROL_EVIDENCE"

        else:

            if c != pending_team:
                pending_team = c
                pending_frames = 0
                pending_observed_frames = 0

            pending_frames += 1

            if is_observed:
                pending_observed_frames += 1

            # Una secuencia enteramente imputada exige
            # más evidencia temporal que una observada.
            if pending_observed_frames > 0:
                needed = OBSERVED_CONFIRM_FRAMES
            elif is_imputed:
                needed = IMPUTED_CONFIRM_FRAMES
            else:
                needed = IMPUTED_CONFIRM_FRAMES

            if pending_frames >= needed:

                previous = current_team
                current_team = c

                labels[i] = current_team
                internal_state[i] = (
                    "CONTROL_" + current_team
                )

                if previous is None:
                    reasons[i] = "INITIAL_CONTROL_CONFIRMED"
                else:
                    reasons[i] = "RIVAL_CONTROL_CONFIRMED"

                pending_team = None
                pending_frames = 0
                pending_observed_frames = 0

            else:

                if current_team is None:
                    labels[i] = "X"
                    internal_state[i] = "UNINITIALIZED"
                    reasons[i] = "INITIAL_CONTROL_PENDING"
                else:
                    labels[i] = current_team
                    internal_state[i] = (
                        "CONTESTED_AFTER_" + current_team
                    )
                    reasons[i] = "RIVAL_CONTROL_PENDING"

        last_confirmed[i] = (
            current_team
            if current_team is not None
            else "NONE"
        )

        pending_team_trace[i] = (
            pending_team
            if pending_team is not None
            else "NONE"
        )

        pending_frames_trace[i] = pending_frames

    trace = pd.DataFrame({
        "frame": df["frame"].to_numpy(),
        "relevant": relevant.astype(np.int8),
        "ball_source": source,
        "home_distance_m": home_distance,
        "away_distance_m": away_distance,
        "distance_margin_home_m": margin,
        "control_candidate": candidate,
        "pending_team": pending_team_trace,
        "pending_frames": pending_frames_trace,
        "last_confirmed_team": last_confirmed,
        "internal_state": internal_state,
        "decision_reason": reasons,
        "label": labels,
    })

    return trace


def get_spells(trace):
    """
    Tramos máximos de etiquetas H/A.
    D y X interrumpen el tramo.
    """
    labels = trace["label"].to_numpy()
    spells = []

    start = None
    previous = None

    for i, label in enumerate(labels):

        if label not in ("H", "A"):
            if start is not None:
                spells.append(
                    (previous, start, i - 1, i - start)
                )
                start = None
                previous = None
            continue

        if start is None:
            start = i
            previous = label

        elif label != previous:
            spells.append(
                (previous, start, i - 1, i - start)
            )
            start = i
            previous = label

    if start is not None:
        spells.append(
            (
                previous,
                start,
                len(labels) - 1,
                len(labels) - start,
            )
        )

    return spells


def main():

    print("=" * 65)
    print("S28 — MOTOR DE POSESIÓN V1")
    print("=" * 65)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    print("\n[1] Cargando S27...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    if len(df) != N_VIDEO_FRAMES:
        raise ValueError(
            f"Frames incorrectos: {len(df)}"
        )

    missing = [
        c for c in REQUIRED
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Columnas faltantes: {missing}"
        )

    if not np.array_equal(
        df["frame"].to_numpy(),
        np.arange(N_VIDEO_FRAMES),
    ):
        raise ValueError("Frames desalineados.")

    print("[2] Ejecutando máquina de estados...")

    trace = build_engine(df)

    labels = trace["label"].to_numpy()

    if not np.isin(
        labels, ["H", "A", "D", "X"]
    ).all():
        raise ValueError(
            "Etiquetas fuera del contrato."
        )

    submission = trace[
        ["frame", "label"]
    ].copy()

    print("[3] Analizando spells...")

    spells = get_spells(trace)

    lengths = np.asarray(
        [s[3] for s in spells],
        dtype=float,
    )

    short_spells = int(
        np.count_nonzero(lengths < FPS)
    )

    label_counts = {
        label: int(
            np.count_nonzero(labels == label)
        )
        for label in ["H", "A", "D", "X"]
    }

    reason_counts = (
        trace["decision_reason"]
        .value_counts()
        .to_dict()
    )

    transitions = int(
        reason_counts.get(
            "RIVAL_CONTROL_CONFIRMED", 0
        )
    )

    print("[4] Guardando resultados...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    trace.to_csv(
        TRACE_PATH,
        index=False,
    )

    lines = [
        "# ETAPA 28 — Motor de posesión V1",
        "",
        "## 1. Alcance",
        "",
        "Primera máquina de estados experimental "
        "derivada de S26 y S27.",
        "",
        "No usa ground truth ni modifica run.py.",
        "",
        "## 2. Parámetros",
        "",
        f"- Radio de control: {CONTROL_RADIUS_M:.2f} m",
        f"- Ventaja mínima: {MIN_ADVANTAGE_M:.2f} m",
        f"- Confirmación observada: "
        f"{OBSERVED_CONFIRM_S:.2f} s",
        f"- Confirmación imputada: "
        f"{IMPUTED_CONFIRM_S:.2f} s",
        "- Sin control nuevo: conservar último poseedor.",
        "- Sin poseedor inicial: X.",
        "- Sin detector fiable de balón muerto: "
        "no inferir D.",
        "",
        "## 3. Etiquetas",
        "",
        "| Etiqueta | Frames |",
        "|---|---:|",
    ]

    for label, total in label_counts.items():
        lines.append(
            f"| {label} | {total:,} |"
        )

    lines.extend([
        "",
        "## 4. Spells experimentales",
        "",
        f"- Spells H/A: {len(spells):,}",
        f"- Cambios de poseedor confirmados: "
        f"{transitions:,}",
        f"- Spells menores de 1 s: "
        f"{short_spells:,}",
        f"- Duración mediana: "
        f"{np.median(lengths) / FPS if len(lengths) else 0:.2f} s",
        f"- Duración P90: "
        f"{np.percentile(lengths, 90) / FPS if len(lengths) else 0:.2f} s",
        f"- Duración máxima: "
        f"{np.max(lengths) / FPS if len(lengths) else 0:.2f} s",
        "",
        "## 5. Motivos de decisión",
        "",
        "| Motivo | Frames |",
        "|---|---:|",
    ])

    for reason, total in sorted(
        reason_counts.items(),
        key=lambda item: -item[1],
    ):
        lines.append(
            f"| {reason} | {total:,} |"
        )

    lines.extend([
        "",
        "## 6. Limitaciones conocidas",
        "",
        "- No identifica todavía interrupciones D.",
        "- Los parámetros son provisionales.",
        "- La confirmación introduce retraso "
        "en los cambios H/A.",
        "- La evidencia geométrica no prueba "
        "contacto ni control futbolístico.",
        "- Los tags no intervienen aún "
        "en la decisión final.",
        "- No hay suavizado retrospectivo.",
        "- No hay validación contra ground truth.",
        "",
        "## 7. Siguiente paso",
        "",
        "S29: auditoría de spells, fronteras, "
        "duraciones anómalas y sensibilidad "
        "a parámetros; posteriormente, "
        "detección de D y suavizado.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n========== ETAPA 28 ==========")

    print(f"Frames: {len(trace):,}")

    for label, total in label_counts.items():
        print(f"  {label}: {total:,}")

    print(f"\nSpells H/A: {len(spells):,}")
    print(
        "Cambios confirmados:",
        f"{transitions:,}",
    )
    print(
        "Spells menores de 1 s:",
        f"{short_spells:,}",
    )

    if len(lengths):
        print(
            "Mediana de spell:",
            f"{np.median(lengths) / FPS:.2f} s",
        )
        print(
            "P90 de spell:",
            f"{np.percentile(lengths, 90) / FPS:.2f} s",
        )
        print(
            "Máximo de spell:",
            f"{np.max(lengths) / FPS:.2f} s",
        )

    print("\nValidación estructural: True")
    print("Sin ground truth: True")
    print("run.py intacto: True")

    print("\nSubmission:", SUBMISSION_PATH)
    print("Traza:", TRACE_PATH)
    print("Informe:", REPORT_PATH)


if __name__ == "__main__":
    main()
