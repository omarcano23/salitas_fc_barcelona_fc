
"""
S27 — AUDITORÍA DE SEÑALES FUTBOLÍSTICAS

Base:
    S26 — Diccionario futbolístico aprobado.
    S25 — Evidencia geométrica.
    V4  — Características y Smart Tagging.

Entradas:
    local_outputs/matrix_v4.csv
    local_outputs/s25_possession_evidence.csv

Salidas:
    local_outputs/s27_football_signals.csv
    ETAPA_27_SENALES_FUTBOLISTICAS.md

No usa ground truth.
No clasifica posesión.
No modifica V4 ni run.py.
"""

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    OUTPUT_DIR,
    N_VIDEO_FRAMES,
    FPS,
)

MATRIX_PATH = OUTPUT_DIR / "matrix_v4.csv"
S25_PATH = OUTPUT_DIR / "s25_possession_evidence.csv"

TRACE_PATH = OUTPUT_DIR / "s27_football_signals.csv"
REPORT_PATH = ROOT_DIR / "ETAPA_27_SENALES_FUTBOLISTICAS.md"

TAG_CODES = [
    "long_ball",
    "set_pieces",
    "attacking_transition",
    "defensive_transition",
    "build_up",
    "progression",
]

DISTANCE_THRESHOLDS = (1.5, 3.0, 5.0, 10.0)
MARGIN_THRESHOLDS = (0.5, 1.0, 2.0, 3.0)
DURATION_THRESHOLDS_S = (0.2, 0.4, 0.8, 1.2)

REQUIRED_V4 = [
    "frame",
    "ball_observed",
    "ball_imputed",
    "ball_gap_length",
    "ball_speed_mps",
    "Barcelona_nearest_ball_distance_m",
    "Atletic_Club_nearest_ball_distance_m",
    "Barcelona_players_within_1_5m",
    "Atletic_Club_players_within_1_5m",
    "Barcelona_players_within_3_0m",
    "Atletic_Club_players_within_3_0m",
    "Barcelona_players_within_5_0m",
    "Atletic_Club_players_within_5_0m",
    "smart_tagging_active_count",
]

REQUIRED_S25 = [
    "frame",
    "relevant",
    "ball_source",
    "geometry_comparable",
    "distance_advantage_home_m",
]


def num(df, name):
    return pd.to_numeric(
        df[name],
        errors="coerce",
    ).to_numpy(dtype=float)


def count(mask):
    return int(np.count_nonzero(mask))


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def validate_frame_table(df, name):
    if len(df) != N_VIDEO_FRAMES:
        raise ValueError(
            f"{name}: longitud incorrecta {len(df)}"
        )

    frames = df["frame"].to_numpy()

    if not np.array_equal(
        frames,
        np.arange(N_VIDEO_FRAMES),
    ):
        raise ValueError(
            f"{name}: frames desalineados"
        )


def run_lengths(mask, relevant):
    """
    Duración consecutiva de una condición booleana.
    Se reinicia cuando cambia la condición o
    cuando salimos de los tiempos de juego.

    Devuelve, por frame, la longitud acumulada
    hasta ese instante. Es una señal causal.
    """

    result = np.zeros(
        len(mask),
        dtype=np.int32,
    )

    current = 0

    for i in range(len(mask)):
        if relevant[i] and mask[i]:
            current += 1
        else:
            current = 0

        result[i] = current

    return result


def episode_lengths(mask, relevant):
    """
    Duraciones completas de episodios positivos.
    Solo para diagnóstico, no para decisiones.
    """

    lengths = []
    current = 0

    for i in range(len(mask)):
        if relevant[i] and mask[i]:
            current += 1
        elif current:
            lengths.append(current)
            current = 0

    if current:
        lengths.append(current)

    return np.asarray(
        lengths,
        dtype=np.int32,
    )


def summarize_episodes(mask, relevant):
    lengths = episode_lengths(mask, relevant)

    if len(lengths) == 0:
        return 0, 0.0, 0.0, 0.0

    seconds = lengths / FPS

    return (
        len(lengths),
        float(np.median(seconds)),
        float(np.percentile(seconds, 90)),
        float(np.max(seconds)),
    )


def main():
    print("=" * 65)
    print("S27 — AUDITORÍA DE SEÑALES FUTBOLÍSTICAS")
    print("=" * 65)

    for path in (MATRIX_PATH, S25_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Falta archivo: {path}"
            )

    print("\n[1] Cargando V4 y S25...")

    v4 = pd.read_csv(
        MATRIX_PATH,
        low_memory=False,
    )

    s25 = pd.read_csv(
        S25_PATH,
        low_memory=False,
    )

    validate_frame_table(v4, "V4")
    validate_frame_table(s25, "S25")

    for column in REQUIRED_V4:
        if column not in v4:
            raise ValueError(
                f"Falta columna V4: {column}"
            )

    for column in REQUIRED_S25:
        if column not in s25:
            raise ValueError(
                f"Falta columna S25: {column}"
            )

    relevant = num(s25, "relevant") == 1

    comparable = (
        num(s25, "geometry_comparable") == 1
    )

    observed = num(v4, "ball_observed") == 1
    imputed = num(v4, "ball_imputed") == 1

    if np.any(observed & imputed):
        raise ValueError(
            "Balón observado e imputado simultáneamente."
        )

    if np.any(comparable & ~relevant):
        raise ValueError(
            "Comparabilidad fuera de juego."
        )

    home_d = num(
        v4,
        "Barcelona_nearest_ball_distance_m",
    )
    away_d = num(
        v4,
        "Atletic_Club_nearest_ball_distance_m",
    )

    margin = num(
        s25,
        "distance_advantage_home_m",
    )

    expected_margin = away_d - home_d

    if not np.allclose(
        margin[comparable],
        expected_margin[comparable],
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError(
            "Margen S25 inconsistente con V4."
        )

    print("[2] Construyendo indicadores...")

    nearest = np.full(
        len(v4),
        "NONE",
        dtype=object,
    )

    nearest[comparable & (margin > 0)] = "H"
    nearest[comparable & (margin < 0)] = "A"

    home15 = num(
        v4,
        "Barcelona_players_within_1_5m",
    )
    away15 = num(
        v4,
        "Atletic_Club_players_within_1_5m",
    )

    home3 = num(
        v4,
        "Barcelona_players_within_3_0m",
    )
    away3 = num(
        v4,
        "Atletic_Club_players_within_3_0m",
    )

    home5 = num(
        v4,
        "Barcelona_players_within_5_0m",
    )
    away5 = num(
        v4,
        "Atletic_Club_players_within_5_0m",
    )

    home_close = comparable & (home_d <= 3.0)
    away_close = comparable & (away_d <= 3.0)

    both_close = home_close & away_close

    neither_close = (
        comparable
        & (home_d > 5.0)
        & (away_d > 5.0)
    )

    ambiguous_margin = (
        comparable
        & (np.abs(margin) <= 1.0)
    )

    # Hipótesis exploratorias:
    # proximidad <= 3 m y ventaja >= 1 m.
    # NO son definiciones finales de control.
    candidate_h = (
        comparable
        & (home_d <= 3.0)
        & (margin >= 1.0)
    )

    candidate_a = (
        comparable
        & (away_d <= 3.0)
        & (margin <= -1.0)
    )

    if np.any(candidate_h & candidate_a):
        raise ValueError(
            "Candidatos H/A incompatibles."
        )

    duration_h = run_lengths(
        candidate_h,
        relevant,
    )
    duration_a = run_lengths(
        candidate_a,
        relevant,
    )

    speed = num(
        v4,
        "ball_speed_mps",
    )

    speed_valid = (
        relevant & np.isfinite(speed)
    )

    tags = {}

    for code in TAG_CODES:
        column = (
            f"smart_tagging_code_{code}_count"
        )

        if column not in v4.columns:
            raise ValueError(
                f"Falta Smart Tagging: {column}"
            )

        tags[code] = (
            num(v4, column) > 0
        ) & relevant

    active_tag = (
        num(
            v4,
            "smart_tagging_active_count",
        ) > 0
    ) & relevant

    trace = pd.DataFrame({
        "frame": v4["frame"].to_numpy(),
        "relevant": relevant.astype(np.int8),
        "ball_source": np.where(
            observed,
            "OBSERVED",
            np.where(
                imputed,
                "IMPUTED",
                "NONE",
            ),
        ),
        "ball_gap_length": num(
            v4,
            "ball_gap_length",
        ),
        "ball_speed_mps": speed,
        "geometry_comparable": (
            comparable.astype(np.int8)
        ),
        "home_distance_m": home_d,
        "away_distance_m": away_d,
        "distance_margin_home_m": margin,
        "nearest_team": nearest,
        "home_within_1_5m": home15,
        "away_within_1_5m": away15,
        "home_within_3m": home3,
        "away_within_3m": away3,
        "home_within_5m": home5,
        "away_within_5m": away5,
        "both_teams_within_3m": (
            both_close.astype(np.int8)
        ),
        "neither_team_within_5m": (
            neither_close.astype(np.int8)
        ),
        "ambiguous_margin_1m": (
            ambiguous_margin.astype(np.int8)
        ),
        "control_candidate_h": (
            candidate_h.astype(np.int8)
        ),
        "control_candidate_a": (
            candidate_a.astype(np.int8)
        ),
        "control_candidate_h_run_frames": duration_h,
        "control_candidate_a_run_frames": duration_a,
        "tag_active": active_tag.astype(np.int8),
    })

    for code, mask in tags.items():
        trace[f"tag_{code}"] = (
            mask.astype(np.int8)
        )

    print("[3] Auditando episodios y contexto...")

    conditions = {
        "Candidato de control H": candidate_h,
        "Candidato de control A": candidate_a,
        "Ambos equipos a <=3 m": both_close,
        "Ningún equipo a <=5 m": neither_close,
        "Margen <=1 m": ambiguous_margin,
        "Sin evidencia comparable": (
            relevant & ~comparable
        ),
    }

    lines = [
        "# ETAPA 27 — Señales futbolísticas",
        "",
        "## 1. Alcance",
        "",
        "Traducción exploratoria del diccionario S26 "
        "a indicadores geométricos, temporales "
        "y contextuales.",
        "",
        "No se asignan etiquetas de posesión. "
        "No se utiliza ground truth.",
        "",
        "## 2. Cobertura",
        "",
        f"- Frames: {len(v4):,}",
        f"- En juego: {count(relevant):,}",
        f"- Comparables: {count(comparable):,}",
        f"- Observados: {count(relevant & observed):,}",
        f"- Imputados: {count(relevant & imputed):,}",
        f"- Sin comparación: "
        f"{count(relevant & ~comparable):,}",
        "",
        "## 3. Episodios geométricos",
        "",
        "| Condición | Frames | Episodios "
        "| Mediana (s) | P90 (s) | Máx. (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, mask in conditions.items():
        n_runs, med, p90, maximum = (
            summarize_episodes(mask, relevant)
        )

        lines.append(
            f"| {name} | {count(mask):,} "
            f"| {n_runs:,} | {med:.2f} "
            f"| {p90:.2f} | {maximum:.2f} |"
        )

    lines.extend([
        "",
        "## 4. Sensibilidad de candidatos de control",
        "",
        "Criterio exploratorio: equipo cercano "
        "y ventaja mínima frente al rival.",
        "",
        "| Radio máx. | Margen mín. "
        "| Frames H | Frames A |",
        "|---:|---:|---:|---:|",
    ])

    for radius in DISTANCE_THRESHOLDS:
        for threshold in MARGIN_THRESHOLDS:

            h = (
                comparable
                & (home_d <= radius)
                & (margin >= threshold)
            )

            a = (
                comparable
                & (away_d <= radius)
                & (margin <= -threshold)
            )

            lines.append(
                f"| {radius:.1f} m "
                f"| {threshold:.1f} m "
                f"| {count(h):,} "
                f"| {count(a):,} |"
            )

    lines.extend([
        "",
        "## 5. Duración de candidatos",
        "",
        "Criterio base: radio <=3 m, "
        "ventaja >=1 m.",
        "",
        "| Duración mínima | H confirmado "
        "| A confirmado |",
        "|---:|---:|---:|",
    ])

    for seconds in DURATION_THRESHOLDS_S:
        frames = int(round(seconds * FPS))

        h = count(duration_h >= frames)
        a = count(duration_a >= frames)

        lines.append(
            f"| {seconds:.2f} s "
            f"| {h:,} | {a:,} |"
        )

    lines.extend([
        "",
        "## 6. Smart Tagging",
        "",
        "| Tag | Frames activos | % juego "
        "| Con balón observado | Con balón imputado |",
        "|---|---:|---:|---:|---:|",
    ])

    for code, mask in tags.items():
        lines.append(
            f"| {code} | {count(mask):,} "
            f"| {percentage(count(mask), count(relevant)):.2f}% "
            f"| {count(mask & observed):,} "
            f"| {count(mask & imputed):,} |"
        )

    lines.extend([
        "",
        "## 7. Coexistencia de tags y evidencia",
        "",
        "| Tag | Candidato H | Candidato A "
        "| Ambos cerca | Ninguno cerca |",
        "|---|---:|---:|---:|---:|",
    ])

    for code, mask in tags.items():
        lines.append(
            f"| {code} "
            f"| {count(mask & candidate_h):,} "
            f"| {count(mask & candidate_a):,} "
            f"| {count(mask & both_close):,} "
            f"| {count(mask & neither_close):,} |"
        )

    lines.extend([
        "",
        "## 8. Cinemática descriptiva",
        "",
        f"- Velocidad disponible: "
        f"{count(speed_valid):,} frames.",
        "",
        "| Procedencia | N | Mediana m/s "
        "| P90 m/s | P99 m/s |",
        "|---|---:|---:|---:|",
    ])

    for name, source_mask in [
        ("OBSERVED", observed),
        ("IMPUTED", imputed),
    ]:
        values = speed[
            speed_valid & source_mask
        ]

        if len(values):
            med, p90, p99 = np.percentile(
                values,
                [50, 90, 99],
            )
        else:
            med = p90 = p99 = float("nan")

        lines.append(
            f"| {name} | {len(values):,} "
            f"| {med:.2f} | {p90:.2f} "
            f"| {p99:.2f} |"
        )

    lines.extend([
        "",
        "## 9. Interpretación y restricciones",
        "",
        "- Los candidatos de control son hipótesis "
        "geométricas, no control confirmado.",
        "- Los episodios se calculan sin atravesar "
        "frames excluidos.",
        "- La duración acumulada es causal.",
        "- Un tag SET PIECES no implica por sí solo D.",
        "- Un tag LONG BALL no demuestra recepción.",
        "- Un duelo aéreo no puede confirmarse "
        "sin información vertical fiable.",
        "- El balón libre mantiene al último "
        "poseedor hasta control rival o D.",
        "- No se han recortado velocidades extremas.",
        "- No se ha utilizado ground truth.",
        "",
        "## 10. Siguiente etapa",
        "",
        "S28: máquina de estados parametrizada, "
        "control rival persistente, continuidad "
        "del último poseedor, traza de decisiones "
        "y archivo de predicciones.",
    ])

    print("[4] Guardando traza...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace.to_csv(
        TRACE_PATH,
        index=False,
    )

    print("[5] Guardando informe...")

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n========== ETAPA 27 ==========")
    print(f"Frames: {len(v4):,}")
    print(f"En juego: {count(relevant):,}")
    print(f"Comparables: {count(comparable):,}")
    print(
        "Candidatos H:",
        f"{count(candidate_h):,}",
    )
    print(
        "Candidatos A:",
        f"{count(candidate_a):,}",
    )
    print(
        "Ambos equipos a <=3 m:",
        f"{count(both_close):,}",
    )
    print(
        "Ningún equipo a <=5 m:",
        f"{count(neither_close):,}",
    )
    print(
        "Sin evidencia comparable:",
        f"{count(relevant & ~comparable):,}",
    )

    print("\nTags:")
    for code, mask in tags.items():
        print(
            f"  {code}: {count(mask):,}"
        )

    print("\nValidación estructural: True")
    print("Sin ground truth: True")
    print("Sin modificación de V4: True")
    print("\nTraza:", TRACE_PATH)
    print("Informe:", REPORT_PATH)


if __name__ == "__main__":
    main()
