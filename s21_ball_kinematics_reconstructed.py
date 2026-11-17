
"""
S21 — Actualización de S12 con balón reconstruido.

Reutiliza las 17 variables originales de S12 y
la reconstrucción temporal de S17.

Objetivos:
- Conservar nombres y orden de las columnas S12.
- No modificar tracking, S12 ni S17.
- Mantener NaN fuera de los dos tiempos.
- Preservar cinemática entre observaciones reales.
- Auditar velocidades observadas, imputadas y mixtas.
- Conservar anomalías sin recortarlas.
- No crear CSV intermedios.

Informe:
ETAPA_21_CINEMATICA_BALON_RECONSTRUIDO.md
"""

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    N_VIDEO_FRAMES,
    FPS,
    FIRST_HALF_START,
    FIRST_HALF_END,
    SECOND_HALF_START,
    SECOND_HALF_END,
)

from s01_tracking import load_tracking

from s12_ball_kinematics import (
    build_ball_kinematics,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)

from s17_ball_reconstruction import (
    build_ball_reconstruction,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

STRATEGY = "temporal"
CONTEXT_WEIGHT = 0.0

REPORT_PATH = (
    ROOT_DIR
    / "ETAPA_21_CINEMATICA_BALON_RECONSTRUIDO.md"
)

PERIODS = (
    (FIRST_HALF_START, FIRST_HALF_END + 1),
    (SECOND_HALF_START, SECOND_HALF_END + 1),
)


# ============================================================
# UTILIDADES
# ============================================================

def get_period_masks(n):
    relevant = np.zeros(n, dtype=bool)
    consecutive = np.zeros(n, dtype=bool)

    for start, end in PERIODS:
        relevant[start:end] = True
        consecutive[start + 1:end] = True

    return relevant, consecutive


def same_numeric(a, b, atol=1e-9):
    return np.isclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        rtol=0,
        atol=atol,
        equal_nan=True,
    )


def count_changed(old, new, mask=None):
    a = np.asarray(old, dtype=float)
    b = np.asarray(new, dtype=float)

    if mask is not None:
        a = a[mask]
        b = b[mask]

    return int((~same_numeric(a, b)).sum())


def velocity_statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "over40": 0,
            "over50": 0,
        }

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "over40": int(np.sum(values > 40)),
        "over50": int(np.sum(values > 50)),
    }


# ============================================================
# CONSTRUCCIÓN S21
# ============================================================

def build_reconstructed_ball_kinematics(
    tracking,
    reconstruction=None,
):
    """
    Interfaz reutilizable por S23.

    Returns:
        features: frame + 17 variables S12
        reconstruction: metadatos S17
    """

    if reconstruction is None:
        reconstruction = build_ball_reconstruction(
            tracking,
            strategy=STRATEGY,
            context_weight=CONTEXT_WEIGHT,
        )

    if len(tracking) != N_VIDEO_FRAMES:
        raise ValueError(
            "Número de frames incorrecto."
        )

    if len(reconstruction) != len(tracking):
        raise ValueError(
            "S17 y tracking tienen longitudes distintas."
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

    # Reutilizamos las fórmulas y validaciones S12.
    features, ball_valid = build_ball_kinematics(
        working_tracking,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        fps=FPS,
    )

    return features, reconstruction


# ============================================================
# AUDITORÍA
# ============================================================

def audit_s21(
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

    if len(updated) != n:
        raise ValueError(
            "S21 tiene longitud incorrecta."
        )

    if len(updated.columns) != 18:
        raise ValueError(
            "S21 debe tener frame + 17 variables."
        )

    if original.columns.tolist() != updated.columns.tolist():
        raise ValueError(
            "S21 no conserva columnas y orden de S12."
        )

    if not np.array_equal(
        updated["frame"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "S21 tiene frames desalineados."
        )

    relevant, consecutive = get_period_masks(n)

    observed = reconstruction[
        "ball_observed"
    ].to_numpy(dtype=bool)

    imputed = reconstruction[
        "ball_imputed"
    ].to_numpy(dtype=bool)

    raw = reconstruction[
        ["ball_x_raw", "ball_y_raw"]
    ].to_numpy(dtype=float)

    estimated = reconstruction[
        ["ball_x_estimated", "ball_y_estimated"]
    ].to_numpy(dtype=float)

    raw_finite = np.isfinite(raw).all(axis=1)

    raw_inside = (
        raw_finite
        & (raw[:, 0] >= 0)
        & (raw[:, 0] <= 1)
        & (raw[:, 1] >= 0)
        & (raw[:, 1] <= 1)
    )

    estimated_finite = (
        np.isfinite(estimated).all(axis=1)
    )

    estimated_inside = (
        estimated_finite
        & (estimated[:, 0] >= 0)
        & (estimated[:, 0] <= 1)
        & (estimated[:, 1] >= 0)
        & (estimated[:, 1] <= 1)
    )

    # Preservación exacta de las observaciones
    # durante los periodos de juego.
    observed_in_game = relevant & observed

    if not np.array_equal(
        raw[observed_in_game],
        estimated[observed_in_game],
    ):
        raise ValueError(
            "S17 modificó posiciones originales."
        )

    # Las posiciones de S12 se conservan en
    # observaciones originalmente válidas.
    position_columns = [
        "ball_position_available",
        "ball_x_normalized",
        "ball_y_normalized",
        "ball_x_m",
        "ball_y_m",
        "ball_outside_normalized",
    ]

    for col in position_columns:
        old = original[col].to_numpy(dtype=float)
        new = updated[col].to_numpy(dtype=float)

        if not same_numeric(
            old[observed_in_game],
            new[observed_in_game],
        ).all():
            raise ValueError(
                f"Posición observada alterada: {col}"
            )

    # Pares consecutivos dentro de un mismo tiempo.
    previous_observed = np.zeros(n, dtype=bool)
    previous_observed[1:] = observed[:-1]

    previous_raw_inside = np.zeros(n, dtype=bool)
    previous_raw_inside[1:] = raw_inside[:-1]

    previous_estimated_inside = np.zeros(
        n, dtype=bool
    )
    previous_estimated_inside[1:] = (
        estimated_inside[:-1]
    )

    previous_imputed = np.zeros(n, dtype=bool)
    previous_imputed[1:] = imputed[:-1]

    observed_pairs = (
        consecutive
        & observed
        & previous_observed
        & raw_inside
        & previous_raw_inside
    )

    reconstructed_pairs = (
        consecutive
        & estimated_inside
        & previous_estimated_inside
    )

    imputed_pairs = (
        reconstructed_pairs
        & imputed
        & previous_imputed
    )

    mixed_pairs = (
        reconstructed_pairs
        & (
            (imputed & previous_observed)
            | (observed & previous_imputed)
        )
    )

    # Verificamos que ninguna cinemática observada
    # haya cambiado respecto a S12.
    for col in original.columns:

        old = original[col].to_numpy(dtype=float)
        new = updated[col].to_numpy(dtype=float)

        if not same_numeric(
            old[observed_pairs],
            new[observed_pairs],
        ).all():
            raise ValueError(
                "Cinemática observada alterada: "
                + col
            )

    # No permitir velocidad entre periodos.
    velocity_available = updated[
        "ball_velocity_available"
    ].to_numpy(dtype=bool)

    if np.any(
        velocity_available & ~consecutive
    ):
        raise ValueError(
            "Velocidades calculadas entre periodos."
        )

    if not np.array_equal(
        velocity_available,
        reconstructed_pairs,
    ):
        raise ValueError(
            "Disponibilidad de velocidad inconsistente."
        )

    # No se generan posiciones fuera del juego.
    if updated.loc[
        ~relevant, "ball_position_available"
    ].ne(0).any():
        raise ValueError(
            "Posiciones disponibles fuera de los tiempos."
        )

    if updated.loc[
        ~relevant, "ball_velocity_available"
    ].ne(0).any():
        raise ValueError(
            "Velocidades fuera de los tiempos."
        )

    # Velocidades separadas por procedencia.
    speed = updated[
        "ball_speed_mps"
    ].to_numpy(dtype=float)

    velocity_groups = {
        "OBSERVED_OBSERVED": observed_pairs,
        "IMPUTED_IMPUTED": imputed_pairs,
        "MIXED": mixed_pairs,
    }

    stats = {
        name: velocity_statistics(speed[mask])
        for name, mask in velocity_groups.items()
    }

    changed_columns = {}

    for col in original.columns:
        if col == "frame":
            continue

        changed_columns[col] = count_changed(
            original[col].to_numpy(dtype=float),
            updated[col].to_numpy(dtype=float),
        )

    # Fuera del campo S12 conserva su criterio
    # original: registra la posición como no válida.
    outside_estimated = (
        relevant
        & estimated_finite
        & ~estimated_inside
    )

    # Diferencias de cobertura calculadas solo
    # dentro de los periodos de juego.
    original_available = (
        relevant & raw_inside
    )

    updated_available = (
        relevant & estimated_inside
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
        "original_available": int(
            original_available.sum()
        ),
        "updated_available": int(
            updated_available.sum()
        ),
        "new_positions": int(
            (
                updated_available
                & ~original_available
            ).sum()
        ),
        "imputed_outside": int(
            outside_estimated.sum()
        ),
        "observed_pairs": int(
            observed_pairs.sum()
        ),
        "reconstructed_pairs": int(
            reconstructed_pairs.sum()
        ),
        "new_velocity_pairs": int(
            (
                reconstructed_pairs
                & ~observed_pairs
            ).sum()
        ),
        "imputed_pairs": int(
            imputed_pairs.sum()
        ),
        "mixed_pairs": int(
            mixed_pairs.sum()
        ),
        "changed_columns": changed_columns,
        "velocity_stats": stats,
    }


# ============================================================
# INFORME
# ============================================================

def write_report(audit):

    lines = [
        "# ETAPA 21 — Cinemática del balón reconstruido",
        "",
        "## Configuración",
        "- Base: S12.",
        "- Reconstrucción: S17 temporal.",
        "- Peso contextual: 0.00.",
        f"- FPS: {FPS}.",
        f"- Campo: {PITCH_LENGTH_M} × "
        f"{PITCH_WIDTH_M} m.",
        "- Las velocidades extremas se conservan.",
        "",
        "## Integridad",
        f"- Frames: {audit['frames']:,}.",
        f"- Columnas: {audit['columns']}.",
        "- Variables originales S12: 17.",
        "- Nombres y orden: preservados.",
        "- Observaciones originales: preservadas.",
        "- Sin velocidades entre periodos.",
        "",
        "## Cobertura de posición",
        f"- Observados durante juego: "
        f"{audit['observed_in_game']:,}.",
        f"- Imputados durante juego: "
        f"{audit['imputed_in_game']:,}.",
        f"- Balón original válido en juego: "
        f"{audit['original_available']:,}.",
        f"- Balón reconstruido válido en juego: "
        f"{audit['updated_available']:,}.",
        f"- Nuevas posiciones válidas: "
        f"{audit['new_positions']:,}.",
        f"- Estimaciones fuera del campo: "
        f"{audit['imputed_outside']:,}.",
        "",
        "## Cobertura de velocidad",
        f"- Pares observados: "
        f"{audit['observed_pairs']:,}.",
        f"- Pares reconstruidos: "
        f"{audit['reconstructed_pairs']:,}.",
        f"- Nuevos pares con velocidad: "
        f"{audit['new_velocity_pairs']:,}.",
        f"- Pares imputado-imputado: "
        f"{audit['imputed_pairs']:,}.",
        f"- Pares mixtos: "
        f"{audit['mixed_pairs']:,}.",
        "",
        "## Cambios por variable",
        "",
        "| Variable | Frames modificados |",
        "|---|---:|",
    ]

    for col, count in (
        audit["changed_columns"].items()
    ):
        lines.append(
            f"| {col} | {count:,} |"
        )

    lines.extend([
        "",
        "## Velocidades por procedencia",
        "",
        "| Tipo de par | N | Media | P95 | P99 | "
        "Máximo | >40 | >50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for name, values in (
        audit["velocity_stats"].items()
    ):
        lines.append(
            f"| {name} "
            f"| {values['count']:,} "
            f"| {values['mean']:.2f} "
            f"| {values['p95']:.2f} "
            f"| {values['p99']:.2f} "
            f"| {values['max']:.2f} "
            f"| {values['over40']:,} "
            f"| {values['over50']:,} |"
        )

    lines.extend([
        "",
        "## Interpretación",
        "- Las velocidades derivadas son geométricas.",
        "- Los valores extremos no se recortan.",
        "- La posición estimada no equivale a una "
        "observación real.",
        "- Los pares mixtos son especialmente "
        "relevantes para auditar discontinuidades.",
        "- Los frames con balón fuera del campo "
        "se mantienen como posición no disponible "
        "para S12, respetando su contrato original.",
        "- La reconstrucción no garantiza mejoras "
        "en clasificación de posesión.",
        "",
        "## Próxima etapa",
        "S22 — Actualización de S13, proximidad "
        "global al balón.",
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

    print("Calculando S12 original...")

    original, _ = build_ball_kinematics(
        tracking,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        fps=FPS,
    )

    print("Reconstruyendo balón S17 (temporal)...")

    reconstruction = build_ball_reconstruction(
        tracking,
        strategy=STRATEGY,
        context_weight=CONTEXT_WEIGHT,
    )

    print("Calculando S21 actualizado...")

    updated, _ = build_reconstructed_ball_kinematics(
        tracking,
        reconstruction,
    )

    print("Validando S21...")

    audit = audit_s21(
        tracking,
        original,
        updated,
        reconstruction,
    )

    write_report(audit)

    print("\n========== ETAPA 21 ==========")

    print(f"Frames: {audit['frames']:,}")
    print(f"Columnas: {audit['columns']}")

    print(
        "Balón original válido en juego:",
        f"{audit['original_available']:,}",
    )

    print(
        "Balón reconstruido válido en juego:",
        f"{audit['updated_available']:,}",
    )

    print(
        "Nuevas posiciones válidas:",
        f"{audit['new_positions']:,}",
    )

    print(
        "Estimaciones fuera del campo:",
        f"{audit['imputed_outside']:,}",
    )

    print(
        "Pares observados:",
        f"{audit['observed_pairs']:,}",
    )

    print(
        "Pares reconstruidos:",
        f"{audit['reconstructed_pairs']:,}",
    )

    print(
        "Nuevos pares con velocidad:",
        f"{audit['new_velocity_pairs']:,}",
    )

    print("\nVelocidades por procedencia:")

    for name, values in (
        audit["velocity_stats"].items()
    ):
        print(
            f"  {name}: "
            f"N={values['count']:,}, "
            f"P99={values['p99']:.2f} m/s, "
            f"max={values['max']:.2f} m/s, "
            f">50={values['over50']:,}"
        )

    print("\nCambios por variable:")

    for col, count in (
        audit["changed_columns"].items()
    ):
        print(f"  {col}: {count:,}")

    print("\nValidación estructural: True")
    print("Observaciones originales preservadas: True")
    print("Sin velocidades entre periodos: True")

    print(
        f"Informe: {REPORT_PATH.name}"
    )


if __name__ == "__main__":
    main()
