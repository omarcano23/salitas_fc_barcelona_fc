
"""
S19 v2 — Proximidad jugador-balón corregida.

Corrige el error de unidades de S10:
    jugadores: coordenadas normalizadas -> metros
    balón:     coordenadas normalizadas -> metros

Mantiene:
- 199319 frames.
- Las 16 variables y el orden de S10.
- Balón consolidado de S17 temporal.
- El contrato de importación utilizado por S23/S24.

Validación adicional:
La distancia global mínima de S22 debe coincidir
con el mínimo de las distancias de ambos equipos.

No modifica S10, S17, S22 ni el tracking.
No genera CSV intermedios.
"""

import numpy as np
import pandas as pd

from config import ROOT_DIR, N_VIDEO_FRAMES
from s01_tracking import load_tracking

from s10_ball_proximity import (
    calculate_team_proximity,
    validate_features,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    TEAMS,
)

from s17_ball_reconstruction import (
    build_ball_reconstruction,
)

from s22_global_ball_proximity_reconstructed import (
    build_reconstructed_global_proximity,
)


STRATEGY = "temporal"
CONTEXT_WEIGHT = 0.0

REPORT_PATH = (
    ROOT_DIR
    / "ETAPA_19_PROXIMIDAD_BALON_RECONSTRUIDO.md"
)


def inside_pitch(x, y):
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= 1)
        & (y >= 0)
        & (y <= 1)
    )


def same_numeric(a, b, atol=1e-7):
    return np.isclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        atol=atol,
        rtol=0,
        equal_nan=True,
    )


def prepare_reconstructed_tracking(
    tracking,
    reconstruction=None,
):
    if reconstruction is None:
        reconstruction = build_ball_reconstruction(
            tracking,
            strategy=STRATEGY,
            context_weight=CONTEXT_WEIGHT,
        )

    if len(tracking) != N_VIDEO_FRAMES:
        raise ValueError("Longitud incorrecta del tracking.")

    if len(reconstruction) != len(tracking):
        raise ValueError("Longitud incorrecta de S17.")

    if not np.array_equal(
        tracking["frame"].to_numpy(),
        reconstruction["frame"].to_numpy(),
    ):
        raise ValueError("Tracking y S17 desalineados.")

    working = tracking.copy()

    working["ball_x"] = reconstruction[
        "ball_x_estimated"
    ].to_numpy(dtype=float)

    working["ball_y"] = reconstruction[
        "ball_y_estimated"
    ].to_numpy(dtype=float)

    return working, reconstruction


def build_corrected_proximity(
    tracking,
    ball_x_normalized,
    ball_y_normalized,
):
    """
    Construye las 16 variables con unidades coherentes.

    Se reutiliza calculate_team_proximity de S10,
    pasando explícitamente el balón en metros.
    """

    n = len(tracking)

    ball_x_normalized = np.asarray(
        ball_x_normalized, dtype=float
    )
    ball_y_normalized = np.asarray(
        ball_y_normalized, dtype=float
    )

    if (
        len(ball_x_normalized) != n
        or len(ball_y_normalized) != n
    ):
        raise ValueError("Longitud incorrecta del balón.")

    ball_valid = inside_pitch(
        ball_x_normalized,
        ball_y_normalized,
    )

    # CORRECCIÓN CENTRAL:
    # S10 ya convierte los jugadores a metros.
    # Debemos pasar también el balón en metros.
    ball_x_m = ball_x_normalized * PITCH_LENGTH_M
    ball_y_m = ball_y_normalized * PITCH_WIDTH_M

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy()
    })

    for team_name, slots in TEAMS.items():

        team_features = calculate_team_proximity(
            tracking,
            team_name,
            slots,
            ball_x_m,
            ball_y_m,
            ball_valid,
            pitch_length_m=PITCH_LENGTH_M,
            pitch_width_m=PITCH_WIDTH_M,
        )

        for column, values in team_features.items():
            features[column] = values

    validate_features(features, ball_valid)

    return features


def build_reconstructed_ball_proximity(
    tracking,
    reconstruction=None,
):
    """
    Interfaz compatible con S23 y S24.

    Returns:
        features: frame + 16 variables
        reconstruction: metadatos S17
    """

    working, reconstruction = (
        prepare_reconstructed_tracking(
            tracking,
            reconstruction,
        )
    )

    features = build_corrected_proximity(
        working,
        working["ball_x"].to_numpy(dtype=float),
        working["ball_y"].to_numpy(dtype=float),
    )

    return features, reconstruction


def audit_s19_v2(
    tracking,
    corrected,
    reconstruction,
):
    n = len(tracking)

    if corrected.shape != (N_VIDEO_FRAMES, 17):
        raise ValueError(
            f"Dimensiones incorrectas: {corrected.shape}"
        )

    if not np.array_equal(
        corrected["frame"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError("Frames desalineados.")

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

    relevant = reconstruction[
        "ball_reconstruction_relevant"
    ].to_numpy(dtype=bool)

    observed = reconstruction[
        "ball_observed"
    ].to_numpy(dtype=bool)

    imputed = reconstruction[
        "ball_imputed"
    ].to_numpy(dtype=bool)

    original_valid = inside_pitch(raw_x, raw_y)
    estimated_valid = inside_pitch(est_x, est_y)

    observed_in_game = relevant & observed

    # Preservación de coordenadas, no de las
    # distancias antiguas calculadas con unidades
    # incompatibles.
    if not np.array_equal(
        raw_x[observed_in_game],
        est_x[observed_in_game],
    ):
        raise ValueError(
            "Se modificó X del balón observado."
        )

    if not np.array_equal(
        raw_y[observed_in_game],
        est_y[observed_in_game],
    ):
        raise ValueError(
            "Se modificó Y del balón observado."
        )

    # Referencia corregida que utiliza únicamente
    # las posiciones originales del balón.
    corrected_original = build_corrected_proximity(
        tracking,
        raw_x,
        raw_y,
    )

    comparison_mask = (
        relevant & observed & original_valid
    )

    for col in corrected.columns:

        if not same_numeric(
            corrected_original.loc[
                comparison_mask, col
            ],
            corrected.loc[
                comparison_mask, col
            ],
        ).all():
            raise ValueError(
                f"Observación corregida alterada: {col}"
            )

    # Validación cruzada independiente con S22.
    global_features, _ = (
        build_reconstructed_global_proximity(
            tracking,
            reconstruction,
        )
    )

    barca = corrected[
        "Barcelona_nearest_ball_distance_m"
    ].to_numpy(dtype=float)

    athletic = corrected[
        "Atletic_Club_nearest_ball_distance_m"
    ].to_numpy(dtype=float)

    global_distance = global_features[
        "global_nearest_player_distance_m"
    ].to_numpy(dtype=float)

    # Evitar warnings por filas con dos NaN.
    expected_global = np.full(n, np.nan)

    both = np.isfinite(barca) & np.isfinite(athletic)
    only_barca = np.isfinite(barca) & ~np.isfinite(athletic)
    only_athletic = ~np.isfinite(barca) & np.isfinite(athletic)

    expected_global[both] = np.minimum(
        barca[both],
        athletic[both],
    )
    expected_global[only_barca] = barca[only_barca]
    expected_global[only_athletic] = athletic[only_athletic]

    valid_comparison = (
        np.isfinite(expected_global)
        & np.isfinite(global_distance)
    )

    missing_disagreement = (
        np.isfinite(expected_global)
        != np.isfinite(global_distance)
    )

    if missing_disagreement.any():
        raise ValueError(
            "S19 y S22 discrepan en disponibilidad "
            f"de distancia: {missing_disagreement.sum()} frames."
        )

    errors = np.abs(
        expected_global[valid_comparison]
        - global_distance[valid_comparison]
    )

    max_error = (
        float(errors.max())
        if len(errors)
        else 0.0
    )

    if max_error > 1e-6:
        raise ValueError(
            "S19 y S22 discrepan en distancia mínima. "
            f"Error máximo: {max_error:.9f} m"
        )

    # Los conteos por radio de ambos equipos
    # deben sumar los conteos globales.
    threshold_mapping = {
        "1_5": "global_players_within_1_5m",
        "3_0": "global_players_within_3m",
        "5_0": "global_players_within_5m",
        "10_0": "global_players_within_10m",
    }

    threshold_checks = {}

    for suffix, global_col in threshold_mapping.items():

        barca_count = corrected[
            f"Barcelona_players_within_{suffix}m"
        ].to_numpy(dtype=float)

        athletic_count = corrected[
            f"Atletic_Club_players_within_{suffix}m"
        ].to_numpy(dtype=float)

        global_count = global_features[
            global_col
        ].to_numpy(dtype=float)

        # S13 representa la ausencia del balón
        # con NaN; S10 usa conteos cero.
        mask = estimated_valid

        mismatch = ~same_numeric(
            barca_count[mask] + athletic_count[mask],
            global_count[mask],
            atol=0,
        )

        if mismatch.any():
            raise ValueError(
                f"Conteos incompatibles a {suffix} m: "
                f"{mismatch.sum()} frames."
            )

        threshold_checks[suffix] = int(
            (
                barca_count[mask]
                + athletic_count[mask]
            ).sum()
        )

    # Verificar ausencia de distancias fuera
    # de los dos tiempos de juego.
    for team in TEAMS:

        nearest = (
            f"{team}_nearest_ball_distance_m"
        )

        if corrected.loc[
            ~relevant, nearest
        ].notna().any():
            raise ValueError(
                "Distancias fuera de los tiempos."
            )

    # Comparación con el cálculo S10 defectuoso.
    # Solo para medir el impacto del arreglo.
    from s10_ball_proximity import build_ball_proximity

    working, _ = prepare_reconstructed_tracking(
        tracking,
        reconstruction,
    )

    old_s19 = build_ball_proximity(working)

    changed = {}

    for col in corrected.columns:
        if col == "frame":
            continue

        changed[col] = int(
            (~same_numeric(
                old_s19[col],
                corrected[col],
            )).sum()
        )

    proximity_summary = {}

    for team in TEAMS:

        proximity_summary[team] = {}

        for suffix in threshold_mapping:

            col = (
                f"{team}_players_within_{suffix}m"
            )

            values = corrected[
                col
            ].to_numpy(dtype=float)

            proximity_summary[team][suffix] = {
                "frames_positive": int(
                    (values > 0).sum()
                ),
                "total_players": int(
                    values.sum()
                ),
            }

    return {
        "frames": n,
        "columns": len(corrected.columns),
        "observed_in_game": int(
            observed_in_game.sum()
        ),
        "imputed_in_game": int(
            imputed.sum()
        ),
        "original_valid_in_game": int(
            (relevant & original_valid).sum()
        ),
        "estimated_valid_in_game": int(
            (relevant & estimated_valid).sum()
        ),
        "new_valid": int(
            (
                relevant
                & estimated_valid
                & ~original_valid
            ).sum()
        ),
        "global_comparable": int(
            valid_comparison.sum()
        ),
        "global_max_error_m": max_error,
        "changed_vs_old_s19": changed,
        "proximity_summary": proximity_summary,
        "threshold_checks": threshold_checks,
    }


def write_report(audit):

    lines = [
        "# ETAPA 19 v2 — Proximidad corregida",
        "",
        "## Corrección",
        "- Error detectado: jugadores en metros "
        "y balón normalizado en S10/S19 v1.",
        "- Solución: convertir también el balón "
        "a metros antes de calcular distancias.",
        "- No se modifica S10.",
        "- Se conserva la reconstrucción temporal S17.",
        "- Las posiciones observadas no cambian.",
        "- Las distancias antiguas pueden cambiar "
        "incluso en frames observados.",
        "",
        "## Integridad",
        f"- Frames: {audit['frames']:,}.",
        f"- Columnas: {audit['columns']}.",
        f"- Observados durante juego: "
        f"{audit['observed_in_game']:,}.",
        f"- Imputados durante juego: "
        f"{audit['imputed_in_game']:,}.",
        f"- Balón original válido en juego: "
        f"{audit['original_valid_in_game']:,}.",
        f"- Balón reconstruido válido en juego: "
        f"{audit['estimated_valid_in_game']:,}.",
        f"- Nuevas posiciones válidas: "
        f"{audit['new_valid']:,}.",
        "",
        "## Validación cruzada con S22",
        f"- Frames comparables: "
        f"{audit['global_comparable']:,}.",
        f"- Error máximo distancia mínima: "
        f"{audit['global_max_error_m']:.10f} m.",
        "- Conteos globales por radio: "
        "suma exacta de ambos equipos.",
        "",
        "## Activación de proximidades",
        "",
        "| Equipo | Radio | Frames con >=1 jugador "
        "| Total jugadores contados |",
        "|---|---:|---:|---:|",
    ]

    for team, values in (
        audit["proximity_summary"].items()
    ):
        for radius, summary in values.items():
            lines.append(
                f"| {team} | {radius.replace('_', '.')} m "
                f"| {summary['frames_positive']:,} "
                f"| {summary['total_players']:,} |"
            )

    lines.extend([
        "",
        "## Cambios frente a S19 v1",
        "",
        "| Variable | Frames modificados |",
        "|---|---:|",
    ])

    for col, count in (
        audit["changed_vs_old_s19"].items()
    ):
        lines.append(
            f"| {col} | {count:,} |"
        )

    lines.extend([
        "",
        "## Conclusión",
        "- Un conteo positivo indica proximidad "
        "geométrica, no posesión confirmada.",
        "- El balón consolidado prioriza observación "
        "real sobre reconstrucción S17.",
        "- No se recortan velocidades ni distancias.",
        "- La matriz V4 debe regenerarse con S23.",
        "- La auditoría S24 debe repetirse.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():

    print("Cargando tracking...")
    tracking, _ = load_tracking()

    print("Reconstruyendo balón S17...")
    reconstruction = build_ball_reconstruction(
        tracking,
        strategy=STRATEGY,
        context_weight=CONTEXT_WEIGHT,
    )

    print("Calculando S19 v2 con unidades corregidas...")
    features, _ = build_reconstructed_ball_proximity(
        tracking,
        reconstruction,
    )

    print("Validando contra S22...")
    audit = audit_s19_v2(
        tracking,
        features,
        reconstruction,
    )

    write_report(audit)

    print("\n========== ETAPA 19 v2 ==========")
    print(f"Frames: {audit['frames']:,}")
    print(f"Columnas: {audit['columns']}")

    print(
        "Balón original válido en juego:",
        f"{audit['original_valid_in_game']:,}",
    )
    print(
        "Balón reconstruido válido en juego:",
        f"{audit['estimated_valid_in_game']:,}",
    )
    print(
        "Nuevas posiciones válidas:",
        f"{audit['new_valid']:,}",
    )
    print(
        "Comparaciones con S22:",
        f"{audit['global_comparable']:,}",
    )
    print(
        "Error máximo frente a S22:",
        f"{audit['global_max_error_m']:.10f} m",
    )

    print("\nFrames con al menos un jugador próximo:")

    for team, values in (
        audit["proximity_summary"].items()
    ):
        print(f"  {team}:")
        for radius, summary in values.items():
            print(
                f"    {radius.replace('_', '.')} m: "
                f"{summary['frames_positive']:,}"
            )

    print("\nCambios frente a S19 v1:")

    for col, count in (
        audit["changed_vs_old_s19"].items()
    ):
        print(f"  {col}: {count:,}")

    print("\nValidación estructural: True")
    print("Unidades coherentes: True")
    print("Observaciones de S17 preservadas: True")
    print("Distancia mínima coincide con S22: True")
    print("Conteos por radio coinciden con S22: True")
    print(f"\nInforme: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
