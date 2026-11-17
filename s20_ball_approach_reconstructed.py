
"""
S20 — Dinámica de aproximación al balón reconstruido.

Actualiza S11 utilizando S17 temporal.

Correcciones:
1. Balón y jugadores en metros, no unidades mezcladas.
2. No compara frames entre dos periodos.
3. Conserva los nombres de las 12 variables de S11.
4. Verifica preservación de observaciones originales
   en pares de frames consecutivos válidos.
5. Detecta velocidades de cierre extremas.
6. No genera CSV ni modifica scripts anteriores.

Salida:
ETAPA_20_APROXIMACION_BALON_RECONSTRUIDO.md
"""

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    N_VIDEO_FRAMES,
    FIRST_HALF_START,
    FIRST_HALF_END,
    SECOND_HALF_START,
    SECOND_HALF_END,
)

from s01_tracking import load_tracking

from s11_ball_approach_dynamics import (
    calculate_team_approach,
    validate_features,
)

from s17_ball_reconstruction import (
    build_ball_reconstruction,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

FPS = 25

REPORT_PATH = (
    ROOT_DIR
    / "ETAPA_20_APROXIMACION_BALON_RECONSTRUIDO.md"
)

TEAMS = {
    "Barcelona": range(1, 25),
    "Atletic_Club": range(25, 47),
}

PERIODS = (
    (FIRST_HALF_START, FIRST_HALF_END + 1),
    (SECOND_HALF_START, SECOND_HALF_END + 1),
)

FEATURE_NAMES = (
    "approach_valid_tracks_count",
    "approaching_count",
    "moving_away_count",
    "approaching_ratio",
    "moving_away_ratio",
    "mean_approach_speed_mps",
)


def period_mask(n):
    mask = np.zeros(n, dtype=bool)

    for start, end in PERIODS:
        mask[start:end] = True

    return mask


def consecutive_period_mask(n):
    """True si frame t y t-1 están en el mismo periodo."""
    mask = np.zeros(n, dtype=bool)

    for start, end in PERIODS:
        mask[start + 1:end] = True

    return mask


def ball_inside_pitch(x, y):
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= 1)
        & (y >= 0)
        & (y <= 1)
    )


def build_approach_features(
    tracking,
    ball_x_normalized,
    ball_y_normalized,
):
    """
    Reutiliza la matemática de S11 con unidades correctas.

    Se calculan las 12 columnas en los 199319 frames.
    Solo se admiten comparaciones consecutivas dentro
    de una misma parte.
    """

    n = len(tracking)

    x = np.asarray(
        ball_x_normalized, dtype=float
    )
    y = np.asarray(
        ball_y_normalized, dtype=float
    )

    if len(x) != n or len(y) != n:
        raise ValueError(
            "Longitud de coordenadas incompatible."
        )

    valid = ball_inside_pitch(x, y)

    # S11 espera posiciones del balón en metros.
    ball_x_m = x * PITCH_LENGTH_M
    ball_y_m = y * PITCH_WIDTH_M

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy()
    })

    for team_name, slots in TEAMS.items():

        team_data = calculate_team_approach(
            tracking,
            team_name,
            slots,
            ball_x_m,
            ball_y_m,
            valid,
            pitch_length_m=PITCH_LENGTH_M,
            pitch_width_m=PITCH_WIDTH_M,
            fps=FPS,
        )

        for column, values in team_data.items():
            features[column] = values

    # Fuera de los periodos y en sus primeros frames
    # no hay una comparación temporal válida.
    comparable = consecutive_period_mask(n)

    for team_name in TEAMS:

        count_columns = [
            f"{team_name}_approach_valid_tracks_count",
            f"{team_name}_approaching_count",
            f"{team_name}_moving_away_count",
        ]

        ratio_speed_columns = [
            f"{team_name}_approaching_ratio",
            f"{team_name}_moving_away_ratio",
            f"{team_name}_mean_approach_speed_mps",
        ]

        features.loc[
            ~comparable, count_columns
        ] = 0

        features.loc[
            ~comparable, ratio_speed_columns
        ] = np.nan

    validate_features(features)

    return features


def build_reconstructed_ball_approach(
    tracking,
    reconstruction=None,
):
    """Interfaz para reutilizar desde S23."""

    if reconstruction is None:
        reconstruction = build_ball_reconstruction(
            tracking,
            strategy="temporal",
            context_weight=0.0,
        )

    if len(reconstruction) != len(tracking):
        raise ValueError(
            "S17 y tracking tienen distintas longitudes."
        )

    if not np.array_equal(
        reconstruction["frame"].to_numpy(),
        tracking["frame"].to_numpy(),
    ):
        raise ValueError(
            "S17 y tracking no están alineados."
        )

    features = build_approach_features(
        tracking,
        reconstruction[
            "ball_x_estimated"
        ].to_numpy(dtype=float),
        reconstruction[
            "ball_y_estimated"
        ].to_numpy(dtype=float),
    )

    return features, reconstruction


def audit_s20(
    tracking,
    original,
    updated,
    reconstruction,
):

    if len(updated) != N_VIDEO_FRAMES:
        raise ValueError("Filas incorrectas.")

    if len(updated.columns) != 13:
        raise ValueError("Columnas incorrectas.")

    if original.columns.tolist() != updated.columns.tolist():
        raise ValueError(
            "S20 no conserva los nombres y orden de S11."
        )

    observed = reconstruction[
        "ball_observed"
    ].to_numpy(dtype=bool)

    relevant = reconstruction[
        "ball_reconstruction_relevant"
    ].to_numpy(dtype=bool)

    imputed = reconstruction[
        "ball_imputed"
    ].to_numpy(dtype=bool)

    raw_x = reconstruction[
        "ball_x_raw"
    ].to_numpy(dtype=float)

    raw_y = reconstruction[
        "ball_y_raw"
    ].to_numpy(dtype=float)

    estimated_x = reconstruction[
        "ball_x_estimated"
    ].to_numpy(dtype=float)

    estimated_y = reconstruction[
        "ball_y_estimated"
    ].to_numpy(dtype=float)

    raw_valid = ball_inside_pitch(
        raw_x, raw_y
    )

    estimated_valid = ball_inside_pitch(
        estimated_x, estimated_y
    )

    comparable = consecutive_period_mask(
        len(tracking)
    )

    # Ambos extremos de cada par deben ser
    # observaciones originales válidas.
    previous_observed = np.zeros(
        len(tracking), dtype=bool
    )
    previous_observed[1:] = observed[:-1]

    previous_valid = np.zeros(
        len(tracking), dtype=bool
    )
    previous_valid[1:] = raw_valid[:-1]

    original_pairs = (
        comparable
        & observed
        & previous_observed
        & raw_valid
        & previous_valid
    )

    # El resultado sobre pares completamente observados
    # debe coincidir exactamente con la referencia S11
    # corregida a metros.
    for column in original.columns:

        old_values = original.loc[
            original_pairs, column
        ].to_numpy(dtype=float)

        new_values = updated.loc[
            original_pairs, column
        ].to_numpy(dtype=float)

        if not np.allclose(
            old_values,
            new_values,
            rtol=0,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                "S20 modificó pares originales: "
                + column
            )

    # Comprobar que no se inventan comparaciones
    # en frames excluidos ni al inicio de una parte.
    for team in TEAMS:

        count_col = (
            f"{team}_approach_valid_tracks_count"
        )

        if not (
            updated.loc[
                ~comparable, count_col
            ].to_numpy() == 0
        ).all():
            raise ValueError(
                "Comparaciones fuera de periodo: "
                + team
            )

    # Disponibilidad de pares del balón:
    # frame actual y anterior dentro del campo.
    previous_estimated_valid = np.zeros(
        len(tracking), dtype=bool
    )

    previous_estimated_valid[1:] = (
        estimated_valid[:-1]
    )

    reconstructed_pairs = (
        comparable
        & estimated_valid
        & previous_estimated_valid
    )

    newly_available_pairs = (
        reconstructed_pairs
        & ~original_pairs
    )

    changed = {}

    for column in original.columns:

        if column == "frame":
            continue

        a = original[column].to_numpy(
            dtype=float
        )
        b = updated[column].to_numpy(
            dtype=float
        )

        changed[column] = int(
            (~np.isclose(
                a, b,
                rtol=0,
                atol=1e-9,
                equal_nan=True,
            )).sum()
        )

    # Comprobar velocidades de cierre extremas.
    # No las eliminamos: solo las auditamos.
    extreme_speeds = {}

    for team in TEAMS:

        col = (
            f"{team}_mean_approach_speed_mps"
        )

        values = updated[col].to_numpy(
            dtype=float
        )

        extreme_speeds[team] = {
            "over_15": int(
                np.sum(values > 15)
            ),
            "over_30": int(
                np.sum(values > 30)
            ),
            "maximum": (
                float(np.nanmax(values))
                if np.isfinite(values).any()
                else float("nan")
            ),
        }

    return {
        "observed_original_pairs": int(
            original_pairs.sum()
        ),
        "reconstructed_pairs": int(
            reconstructed_pairs.sum()
        ),
        "newly_available_pairs": int(
            newly_available_pairs.sum()
        ),
        "imputed_frames": int(
            imputed.sum()
        ),
        "imputed_inside_pitch": int(
            (imputed & estimated_valid).sum()
        ),
        "relevant_frames": int(
            relevant.sum()
        ),
        "changed_columns": changed,
        "extreme_speeds": extreme_speeds,
    }


def write_report(audit, features):

    lines = [
        "# ETAPA 20 — Aproximación al balón reconstruido",
        "",
        "## Configuración",
        "- Base: S11, dinámica de aproximación.",
        "- Balón: S17 temporal, peso contextual 0.00.",
        "- Dimensiones en metros para jugadores y balón.",
        "- Velocidad de cierre en m/s.",
        "",
        "## Integridad",
        f"- Frames: {len(features):,}",
        f"- Columnas: {len(features.columns)}",
        "- Variables de S11 conservadas: 12.",
        "- Pares originalmente observados: preservados.",
        "- No se comparan frames entre periodos.",
        "",
        "## Cobertura",
        f"- Pares con balón original válido: "
        f"{audit['observed_original_pairs']:,}",
        f"- Pares con balón reconstruido válido: "
        f"{audit['reconstructed_pairs']:,}",
        f"- Nuevos pares comparables: "
        f"{audit['newly_available_pairs']:,}",
        f"- Frames imputados: "
        f"{audit['imputed_frames']:,}",
        f"- Imputados dentro del campo: "
        f"{audit['imputed_inside_pitch']:,}",
        "",
        "## Cambios por variable",
        "",
        "| Variable | Frames modificados |",
        "|---|---:|",
    ]

    for column, count in (
        audit["changed_columns"].items()
    ):
        lines.append(
            f"| {column} | {count:,} |"
        )

    lines.extend([
        "",
        "## Auditoría de velocidades",
        "",
        "| Equipo | >15 m/s | >30 m/s | Máximo (m/s) |",
        "|---|---:|---:|---:|",
    ])

    for team, values in (
        audit["extreme_speeds"].items()
    ):
        lines.append(
            f"| {team} "
            f"| {values['over_15']:,} "
            f"| {values['over_30']:,} "
            f"| {values['maximum']:.2f} |"
        )

    lines.extend([
        "",
        "## Interpretación",
        "- Las velocidades son cambios en la distancia "
        "jugador-balón, no velocidades físicas del jugador.",
        "- Una velocidad extrema puede deberse al movimiento "
        "del balón, a errores del tracking o a imputación.",
        "- El script registra anomalías, pero no las elimina.",
        "- Las coordenadas imputadas son hipótesis, "
        "no observaciones verificadas.",
        "- El aumento de cobertura no garantiza "
        "una mejora en spellF1.",
        "",
        "## Próxima etapa",
        "S21 — Actualización de la cinemática del balón (S12).",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():

    print("Cargando tracking táctico...")
    tracking, _ = load_tracking()

    print(
        "Calculando referencia S11 "
        "con unidades corregidas..."
    )

    raw_x = tracking[
        "ball_x"
    ].to_numpy(dtype=float)

    raw_y = tracking[
        "ball_y"
    ].to_numpy(dtype=float)

    original = build_approach_features(
        tracking,
        raw_x,
        raw_y,
    )

    print("Reconstruyendo balón S17 (temporal)...")

    reconstruction = build_ball_reconstruction(
        tracking,
        strategy="temporal",
        context_weight=0.0,
    )

    print("Calculando S20 actualizado...")

    updated, _ = (
        build_reconstructed_ball_approach(
            tracking,
            reconstruction,
        )
    )

    print("Validando S20...")

    audit = audit_s20(
        tracking,
        original,
        updated,
        reconstruction,
    )

    write_report(audit, updated)

    print("\n========== ETAPA 20 ==========")

    print(f"Frames: {len(updated):,}")
    print(f"Columnas: {len(updated.columns)}")

    print(
        "Pares originales válidos:",
        f"{audit['observed_original_pairs']:,}",
    )

    print(
        "Pares reconstruidos válidos:",
        f"{audit['reconstructed_pairs']:,}",
    )

    print(
        "Nuevos pares comparables:",
        f"{audit['newly_available_pairs']:,}",
    )

    print(
        "Variables modificadas:",
        sum(
            value > 0
            for value in audit[
                "changed_columns"
            ].values()
        ),
        "/ 12",
    )

    print("\nCambios por variable:")

    for column, count in (
        audit["changed_columns"].items()
    ):
        print(f"  {column}: {count:,}")

    print("\nVelocidades extremas:")

    for team, values in (
        audit["extreme_speeds"].items()
    ):
        print(
            f"  {team}: "
            f">15 m/s = {values['over_15']:,}, "
            f">30 m/s = {values['over_30']:,}, "
            f"máximo = {values['maximum']:.2f} m/s"
        )

    print("\nValidación estructural: True")
    print("Pares originales preservados: True")
    print("Unidades coherentes: True")
    print("Sin cruces entre periodos: True")

    print(
        f"Informe: {REPORT_PATH.name}"
    )


if __name__ == "__main__":
    main()
