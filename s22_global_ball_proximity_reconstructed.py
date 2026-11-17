
"""
S22 — Proximidad global jugador-balón con S17.

Actualiza las seis variables de S13 utilizando
el balón reconstruido por S17 temporal.

Conserva:
- Las 199319 filas.
- Los nombres y orden de las seis variables S13.
- Las observaciones originales durante el juego.
- Los criterios geométricos originales.
- Los NaN fuera de los dos tiempos.

No modifica S13, S17 ni el tracking original.
No genera CSV intermedios.

Informe:
ETAPA_22_PROXIMIDAD_GLOBAL_RECONSTRUIDA.md
"""

import numpy as np
import pandas as pd

from config import ROOT_DIR, N_VIDEO_FRAMES
from s01_tracking import load_tracking

from s13_global_player_ball_proximity import (
    build_global_player_ball_proximity,
    validate_features,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

from s17_ball_reconstruction import (
    build_ball_reconstruction,
)


STRATEGY = "temporal"
CONTEXT_WEIGHT = 0.0

REPORT_PATH = (
    ROOT_DIR
    / "ETAPA_22_PROXIMIDAD_GLOBAL_RECONSTRUIDA.md"
)

EXPECTED_COLUMNS = [
    "frame",
    "global_players_within_1_5m",
    "global_players_within_3m",
    "global_players_within_5m",
    "global_players_within_10m",
    "global_nearest_player_slot",
    "global_nearest_player_distance_m",
]


# ============================================================
# UTILIDADES
# ============================================================

def inside_pitch(x, y):
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= 1)
        & (y >= 0)
        & (y <= 1)
    )


def same_numeric(a, b):
    return np.isclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    )


# ============================================================
# CONSTRUCCIÓN S22
# ============================================================

def build_reconstructed_global_proximity(
    tracking,
    reconstruction=None,
):
    """
    Interfaz reutilizable por S23.

    Returns:
        features: frame + 6 variables originales S13
        reconstruction: metadatos de S17
    """

    if len(tracking) != N_VIDEO_FRAMES:
        raise ValueError(
            "Número de frames incorrecto."
        )

    if reconstruction is None:
        reconstruction = build_ball_reconstruction(
            tracking,
            strategy=STRATEGY,
            context_weight=CONTEXT_WEIGHT,
        )

    if len(reconstruction) != len(tracking):
        raise ValueError(
            "S17 y tracking tienen longitudes diferentes."
        )

    if not np.array_equal(
        tracking["frame"].to_numpy(),
        reconstruction["frame"].to_numpy(),
    ):
        raise ValueError(
            "S17 y tracking no están alineados."
        )

    working_tracking = tracking.copy()

    working_tracking["ball_x"] = (
        reconstruction[
            "ball_x_estimated"
        ].to_numpy(dtype=float)
    )

    working_tracking["ball_y"] = (
        reconstruction[
            "ball_y_estimated"
        ].to_numpy(dtype=float)
    )

    features = build_global_player_ball_proximity(
        working_tracking,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
    )

    return features, reconstruction


# ============================================================
# AUDITORÍA
# ============================================================

def audit_s22(
    tracking,
    original,
    updated,
    reconstruction,
):

    n = len(tracking)

    if n != N_VIDEO_FRAMES:
        raise ValueError(
            "Tracking con longitud incorrecta."
        )

    if len(original) != n or len(updated) != n:
        raise ValueError(
            "Número incorrecto de filas."
        )

    if original.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            "S13 no tiene las columnas esperadas."
        )

    if updated.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            "S22 no conserva las columnas de S13."
        )

    if not np.array_equal(
        updated["frame"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "Frames desalineados."
        )

    observed = reconstruction[
        "ball_observed"
    ].to_numpy(dtype=bool)

    imputed = reconstruction[
        "ball_imputed"
    ].to_numpy(dtype=bool)

    relevant = reconstruction[
        "ball_reconstruction_relevant"
    ].to_numpy(dtype=bool)

    raw_x = reconstruction[
        "ball_x_raw"
    ].to_numpy(dtype=float)

    raw_y = reconstruction[
        "ball_y_raw"
    ].to_numpy(dtype=float)

    est_x = reconstruction[
        "ball_x_estimated"
    ].to_numpy(dtype=float)

    est_y = reconstruction[
        "ball_y_estimated"
    ].to_numpy(dtype=float)

    raw_valid = inside_pitch(
        raw_x, raw_y
    )

    reconstructed_valid = inside_pitch(
        est_x, est_y
    )

    # Reutilizar validaciones originales S13.
    validate_features(
        original,
        raw_valid,
    )

    validate_features(
        updated,
        reconstructed_valid,
    )

    # S17 debe preservar el balón observado
    # durante los periodos de juego.
    observed_in_game = (
        relevant & observed
    )

    if not np.array_equal(
        raw_x[observed_in_game],
        est_x[observed_in_game],
    ):
        raise ValueError(
            "S17 modificó ball_x observado."
        )

    if not np.array_equal(
        raw_y[observed_in_game],
        est_y[observed_in_game],
    ):
        raise ValueError(
            "S17 modificó ball_y observado."
        )

    # Comparamos todas las variables de S13
    # en los frames observados y válidos.
    comparison_mask = (
        relevant
        & observed
        & raw_valid
    )

    for column in EXPECTED_COLUMNS[1:]:

        old_values = original.loc[
            comparison_mask,
            column,
        ].to_numpy(dtype=float)

        new_values = updated.loc[
            comparison_mask,
            column,
        ].to_numpy(dtype=float)

        if not same_numeric(
            old_values,
            new_values,
        ).all():
            raise ValueError(
                "S22 alteró observaciones originales "
                f"en {column}"
            )

    # No se calculan proximidades fuera de
    # los dos tiempos de juego.
    if not updated.loc[
        ~relevant,
        EXPECTED_COLUMNS[1:],
    ].isna().all().all():
        raise ValueError(
            "S22 generó proximidades fuera de juego."
        )

    # Nuevos frames con balón válido.
    new_valid = (
        relevant
        & reconstructed_valid
        & ~raw_valid
    )

    # Jugador más cercano identificable.
    original_nearest = original[
        "global_nearest_player_slot"
    ].notna().to_numpy()

    updated_nearest = updated[
        "global_nearest_player_slot"
    ].notna().to_numpy()

    new_nearest = (
        relevant
        & updated_nearest
        & ~original_nearest
    )

    # Cuántos nuevos frames imputados tienen
    # al menos un jugador a 10 metros.
    near_10 = updated[
        "global_players_within_10m"
    ].to_numpy(dtype=float)

    imputed_with_near_player = (
        imputed
        & reconstructed_valid
        & np.isfinite(near_10)
        & (near_10 > 0)
    )

    # Diferencias por columna.
    changed_columns = {}

    for column in EXPECTED_COLUMNS[1:]:

        old_values = original[
            column
        ].to_numpy(dtype=float)

        new_values = updated[
            column
        ].to_numpy(dtype=float)

        changed_columns[column] = int(
            (~same_numeric(
                old_values,
                new_values,
            )).sum()
        )

    # Comprobar que los slots siguen en el rango.
    slots = updated[
        "global_nearest_player_slot"
    ].dropna().to_numpy(dtype=float)

    if not (
        np.all(slots >= 1)
        and np.all(slots <= 46)
        and np.all(slots == np.floor(slots))
    ):
        raise ValueError(
            "Slots de jugador inválidos."
        )

    return {
        "frames": n,
        "columns": len(updated.columns),

        "observed_in_game": int(
            observed_in_game.sum()
        ),

        "imputed_in_game": int(
            imputed.sum()
        ),

        "original_valid_in_game": int(
            (relevant & raw_valid).sum()
        ),

        "updated_valid_in_game": int(
            (relevant & reconstructed_valid).sum()
        ),

        "new_valid_positions": int(
            new_valid.sum()
        ),

        "imputed_outside_pitch": int(
            (
                imputed
                & ~reconstructed_valid
            ).sum()
        ),

        "original_nearest_in_game": int(
            (
                relevant
                & original_nearest
            ).sum()
        ),

        "updated_nearest_in_game": int(
            (
                relevant
                & updated_nearest
            ).sum()
        ),

        "new_nearest_frames": int(
            new_nearest.sum()
        ),

        "imputed_with_player_10m": int(
            imputed_with_near_player.sum()
        ),

        "changed_columns": changed_columns,
    }


# ============================================================
# INFORME
# ============================================================

def write_report(audit):

    lines = [
        "# ETAPA 22 — Proximidad global reconstruida",
        "",
        "## Configuración",
        "- Base: S13.",
        "- Reconstrucción: S17 temporal.",
        "- Peso contextual: 0.00.",
        f"- Campo: {PITCH_LENGTH_M} × "
        f"{PITCH_WIDTH_M} m.",
        "- Se reutilizan las fórmulas originales de S13.",
        "",
        "## Integridad",
        f"- Frames: {audit['frames']:,}.",
        f"- Columnas: {audit['columns']}.",
        "- Variables originales S13: 6.",
        "- Nombres y orden: preservados.",
        "- Observaciones originales: preservadas.",
        "- Sin proximidades fuera de los dos tiempos.",
        "",
        "## Cobertura del balón",
        f"- Observaciones durante juego: "
        f"{audit['observed_in_game']:,}.",
        f"- Frames imputados: "
        f"{audit['imputed_in_game']:,}.",
        f"- Balón original válido en juego: "
        f"{audit['original_valid_in_game']:,}.",
        f"- Balón reconstruido válido en juego: "
        f"{audit['updated_valid_in_game']:,}.",
        f"- Nuevas posiciones válidas: "
        f"{audit['new_valid_positions']:,}.",
        f"- Imputaciones fuera del campo: "
        f"{audit['imputed_outside_pitch']:,}.",
        "",
        "## Proximidad global",
        f"- Frames con jugador identificable "
        f"antes: {audit['original_nearest_in_game']:,}.",
        f"- Frames con jugador identificable "
        f"después: {audit['updated_nearest_in_game']:,}.",
        f"- Nuevos frames con jugador identificable: "
        f"{audit['new_nearest_frames']:,}.",
        f"- Frames imputados con algún jugador "
        f"a 10 m: {audit['imputed_with_player_10m']:,}.",
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
        "## Interpretación",
        "- El slot más cercano no representa "
        "una identidad persistente.",
        "- La proximidad no demuestra posesión.",
        "- La posición imputada sigue siendo "
        "una hipótesis temporal.",
        "- Los conteos solo se calculan si "
        "el balón está dentro del campo.",
        "- Los frames con balón fuera del campo "
        "mantienen NaN en las seis variables.",
        "- Los indicadores de imputación se "
        "incorporarán a la matriz S23.",
        "",
        "## Validaciones",
        "- Estructura y alineación: correctas.",
        "- Monotonía de los radios: correcta.",
        "- Slots de jugador: válidos.",
        "- Distancias no negativas: correctas.",
        "- Observaciones originales: preservadas.",
        "- Frames excluidos: sin proximidades.",
        "",
        "## Próxima etapa",
        "S23 — Matriz final integrada con "
        "S19, S20, S21 y S22.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("Cargando tracking táctico...")
    tracking, _ = load_tracking()

    print("Calculando S13 original...")

    original = build_global_player_ball_proximity(
        tracking,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
    )

    print("Reconstruyendo balón S17 (temporal)...")

    reconstruction = build_ball_reconstruction(
        tracking,
        strategy=STRATEGY,
        context_weight=CONTEXT_WEIGHT,
    )

    print("Calculando S22 actualizado...")

    updated, _ = build_reconstructed_global_proximity(
        tracking,
        reconstruction,
    )

    print("Validando S22...")

    audit = audit_s22(
        tracking,
        original,
        updated,
        reconstruction,
    )

    write_report(audit)

    print("\n========== ETAPA 22 ==========")

    print(f"Frames: {audit['frames']:,}")
    print(f"Columnas: {audit['columns']}")

    print(
        "Balón original válido en juego:",
        f"{audit['original_valid_in_game']:,}",
    )

    print(
        "Balón reconstruido válido en juego:",
        f"{audit['updated_valid_in_game']:,}",
    )

    print(
        "Nuevas posiciones válidas:",
        f"{audit['new_valid_positions']:,}",
    )

    print(
        "Imputaciones fuera del campo:",
        f"{audit['imputed_outside_pitch']:,}",
    )

    print(
        "Jugador más cercano identificable "
        "antes:",
        f"{audit['original_nearest_in_game']:,}",
    )

    print(
        "Jugador más cercano identificable "
        "después:",
        f"{audit['updated_nearest_in_game']:,}",
    )

    print(
        "Nuevos frames con jugador identificable:",
        f"{audit['new_nearest_frames']:,}",
    )

    print(
        "Imputados con jugador a 10 m:",
        f"{audit['imputed_with_player_10m']:,}",
    )

    print("\nCambios por variable:")

    for column, count in (
        audit["changed_columns"].items()
    ):
        print(f"  {column}: {count:,}")

    print("\nValidación estructural: True")
    print("Observaciones originales preservadas: True")
    print("Monotonía de proximidades: True")
    print("Sin proximidades fuera de juego: True")

    print(
        f"Informe: {REPORT_PATH.name}"
    )


if __name__ == "__main__":
    main()
