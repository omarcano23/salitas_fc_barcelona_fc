
import warnings

import numpy as np
import pandas as pd

from config import ROOT_DIR, N_VIDEO_FRAMES
from s01_tracking import load_tracking


REPORT_PATH = ROOT_DIR / "ETAPA_03_GEOMETRIA_GLOBAL.md"

EXPECTED_SLOTS = 46


def row_quantile(sorted_values, counts, q):
    """
    Calcula un cuantil por fila sobre valores ya ordenados.
    Los valores no válidos deben estar representados por +inf.
    """
    n_rows = sorted_values.shape[0]
    result = np.full(n_rows, np.nan, dtype=np.float64)

    valid_rows = counts > 0
    rows = np.flatnonzero(valid_rows)

    if len(rows) == 0:
        return result

    positions = (counts[rows] - 1) * q
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weight = positions - lower

    lower_values = sorted_values[rows, lower]
    upper_values = sorted_values[rows, upper]

    result[rows] = (
        lower_values * (1.0 - weight)
        + upper_values * weight
    )

    return result


def build_global_geometry(tracking, pitch_length_m, pitch_width_m):
    print("Calculando geometría global por frame...")

    x_columns = [f"{slot}_x" for slot in range(1, EXPECTED_SLOTS + 1)]
    y_columns = [f"{slot}_y" for slot in range(1, EXPECTED_SLOTS + 1)]

    missing_columns = [
        column
        for column in x_columns + y_columns
        if column not in tracking.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas de tracking: {missing_columns}"
        )

    # Matrices de coordenadas normalizadas: frames x slots.
    x_norm = tracking[x_columns].to_numpy(dtype=np.float64)
    y_norm = tracking[y_columns].to_numpy(dtype=np.float64)

    # Una posición solo existe si tiene ambas coordenadas.
    position_available = np.isfinite(x_norm) & np.isfinite(y_norm)

    # Conversión de coordenadas normalizadas a metros.
    x_m = x_norm * pitch_length_m
    y_m = y_norm * pitch_width_m

    inside = (
        position_available
        & (x_m >= 0)
        & (x_m <= pitch_length_m)
        & (y_m >= 0)
        & (y_m <= pitch_width_m)
    )

    outside = position_available & ~inside

    inside_count = inside.sum(axis=1).astype(np.int8)
    outside_count = outside.sum(axis=1).astype(np.int8)

    # Para estadísticas espaciales, ignoramos posiciones exteriores.
    x_inside = np.where(inside, x_m, np.inf)
    y_inside = np.where(inside, y_m, np.inf)

    x_sorted = np.sort(x_inside, axis=1)
    y_sorted = np.sort(y_inside, axis=1)

    row_indices = np.arange(len(tracking))
    has_inside = inside_count > 0

    # Centroide.
    x_sum = np.where(inside, x_m, 0.0).sum(axis=1)
    y_sum = np.where(inside, y_m, 0.0).sum(axis=1)

    centroid_x = np.full(len(tracking), np.nan)
    centroid_y = np.full(len(tracking), np.nan)

    centroid_x[has_inside] = (
        x_sum[has_inside] / inside_count[has_inside]
    )
    centroid_y[has_inside] = (
        y_sum[has_inside] / inside_count[has_inside]
    )

    # Dispersión: desviación estándar poblacional (ddof=0).
    spread_x = np.full(len(tracking), np.nan)
    spread_y = np.full(len(tracking), np.nan)

    x_centered = np.where(inside, x_m - centroid_x[:, None], 0.0)
    y_centered = np.where(inside, y_m - centroid_y[:, None], 0.0)

    spread_x[has_inside] = np.sqrt(
        (x_centered[has_inside] ** 2).sum(axis=1)
        / inside_count[has_inside]
    )
    spread_y[has_inside] = np.sqrt(
        (y_centered[has_inside] ** 2).sum(axis=1)
        / inside_count[has_inside]
    )

    # Extremos.
    min_x = np.where(has_inside, x_sorted[:, 0], np.nan)
    min_y = np.where(has_inside, y_sorted[:, 0], np.nan)

    max_indices = np.maximum(inside_count.astype(np.int64) - 1, 0)

    max_x = x_sorted[row_indices, max_indices]
    max_y = y_sorted[row_indices, max_indices]

    max_x[~has_inside] = np.nan
    max_y[~has_inside] = np.nan

    # Cuartiles.
    q25_x = row_quantile(x_sorted, inside_count, 0.25)
    q75_x = row_quantile(x_sorted, inside_count, 0.75)
    q25_y = row_quantile(y_sorted, inside_count, 0.25)
    q75_y = row_quantile(y_sorted, inside_count, 0.75)

    # Inicializar índices seguros. Las posiciones ausentes
    # o exteriores no necesitan asignación de zona.
    zone_x = np.zeros(x_m.shape, dtype=np.int64)
    zone_y = np.zeros(y_m.shape, dtype=np.int64)

    # Calcular zonas exclusivamente para posiciones interiores.
    zone_x[inside] = np.floor(
        x_m[inside] / (pitch_length_m / 3)
    ).astype(np.int64)

    zone_y[inside] = np.floor(
        y_m[inside] / (pitch_width_m / 3)
    ).astype(np.int64)

    # Incluir los límites máximos en la tercera celda.
    zone_x = np.clip(zone_x, 0, 2)
    zone_y = np.clip(zone_y, 0, 2)

    zone_id = zone_x * 3 + zone_y

    frame_zone_id = (
        row_indices[:, None] * 9 + zone_id
    )

    zone_counts = np.bincount(
        frame_zone_id[inside],
        minlength=len(tracking) * 9,
    ).reshape(len(tracking), 9)

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy(dtype=np.int64),
        "inside_slots_count": inside_count,
        "centroid_x_m": centroid_x,
        "centroid_y_m": centroid_y,
        "spread_x_m": spread_x,
        "spread_y_m": spread_y,
        "min_x_m": min_x,
        "max_x_m": max_x,
        "min_y_m": min_y,
        "max_y_m": max_y,
        "q25_x_m": q25_x,
        "q75_x_m": q75_x,
        "q25_y_m": q25_y,
        "q75_y_m": q75_y,
        "outside_slots_count": outside_count,
    })

    zone_names = [
        f"zone_x{x}_y{y}"
        for x in range(1, 4)
        for y in range(1, 4)
    ]

    for index, name in enumerate(zone_names):
        features[name] = zone_counts[:, index].astype(np.int64)

    return features


def validate_geometry(features):
    zone_columns = [
        f"zone_x{x}_y{y}"
        for x in range(1, 4)
        for y in range(1, 4)
    ]

    zone_sum = features[zone_columns].sum(axis=1)
    zone_consistency = (
        zone_sum == features["inside_slots_count"]
    )

    return {
        "rows": len(features),
        "columns": len(features.columns),
        "frame_min": int(features["frame"].min()),
        "frame_max": int(features["frame"].max()),
        "unique_frames": bool(features["frame"].is_unique),
        "mean_inside": float(
            features["inside_slots_count"].mean()
        ),
        "max_inside": int(
            features["inside_slots_count"].max()
        ),
        "mean_outside": float(
            features["outside_slots_count"].mean()
        ),
        "max_outside": int(
            features["outside_slots_count"].max()
        ),
        "frames_without_inside_positions": int(
            (features["inside_slots_count"] == 0).sum()
        ),
        "frames_with_single_inside_position": int(
            (features["inside_slots_count"] == 1).sum()
        ),
        "missing_spread_x": int(
            features["spread_x_m"].isna().sum()
        ),
        "missing_spread_y": int(
            features["spread_y_m"].isna().sum()
        ),
        "zone_counts_consistent": bool(zone_consistency.all()),
        "zone_inconsistency_frames": int((~zone_consistency).sum()),
        "alignment_ok": (
            len(features) == N_VIDEO_FRAMES
            and features["frame"].is_unique
            and features["frame"].iloc[0] == 0
            and features["frame"].iloc[-1] == N_VIDEO_FRAMES - 1
            and len(features.columns) == 24
            and zone_consistency.all()
        ),
    }


def generate_report(validation, pitch_length_m, pitch_width_m):
    lines = [
        "# ETAPA 03 — GEOMETRÍA GLOBAL",
        "",
        "## Fuente y transformación",
        "",
        "- Fuente: tracking ATD Tactical.",
        "- Seguidores: slots 1 a 46, sin separación por equipo.",
        f"- Largo del campo: {pitch_length_m} m.",
        f"- Ancho del campo: {pitch_width_m} m.",
        "- Conversión: x_normalizada × largo; y_normalizada × ancho.",
        "- Posición válida: ambas coordenadas disponibles.",
        "- Posición interior: coordenadas dentro de los límites inclusivos.",
        "- Dispersión: desviación estándar poblacional (`ddof=0`).",
        "- Cuadrícula: 3 × 3; límites máximos incluidos en la tercera celda.",
        "",
        "## Validación estructural",
        "",
        f"- Filas: {validation['rows']:,}",
        f"- Columnas: {validation['columns']}",
        f"- Frames: {validation['frame_min']} a {validation['frame_max']}",
        f"- Frames únicos: {validation['unique_frames']}",
        f"- Media de seguidores dentro: {validation['mean_inside']:.3f}",
        f"- Máximo de seguidores dentro: {validation['max_inside']}",
        f"- Media de seguidores fuera: {validation['mean_outside']:.3f}",
        f"- Máximo de seguidores fuera: {validation['max_outside']}",
        f"- Frames sin posiciones interiores: {validation['frames_without_inside_positions']}",
        f"- Frames con una posición interior: {validation['frames_with_single_inside_position']}",
        f"- Nulos en spread_x_m: {validation['missing_spread_x']}",
        f"- Nulos en spread_y_m: {validation['missing_spread_y']}",
        f"- Suma de zonas coincide con inside_slots_count: {validation['zone_counts_consistent']}",
        f"- Frames con inconsistencia de zonas: {validation['zone_inconsistency_frames']}",
        "",
        "## Columnas generadas",
        "",
    ]

    for column in [
        "inside_slots_count",
        "centroid_x_m", "centroid_y_m",
        "spread_x_m", "spread_y_m",
        "min_x_m", "max_x_m", "min_y_m", "max_y_m",
        "q25_x_m", "q75_x_m", "q25_y_m", "q75_y_m",
        "outside_slots_count",
        *[
            f"zone_x{x}_y{y}"
            for x in range(1, 4)
            for y in range(1, 4)
        ],
    ]:
        lines.append(f"- `{column}`")

    lines.extend([
        "",
        "## Resultado",
        "",
        f"- Alineación estructural correcta: {validation['alignment_ok']}",
        "- Las 23 variables de la familia C se mantienen en memoria.",
        "- No se guardan matrices intermedias en CSV.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    tracking, dataset = load_tracking()

    pitch = dataset.metadata.pitch_dimensions
    pitch_length_m = float(pitch.pitch_length)
    pitch_width_m = float(pitch.pitch_width)

    features = build_global_geometry(
        tracking,
        pitch_length_m,
        pitch_width_m,
    )

    validation = validate_geometry(features)

    print("\nResultado de geometría global:")
    print(f"Filas: {validation['rows']:,}")
    print(f"Columnas: {validation['columns']}")
    print(
        f"Frames: {validation['frame_min']} "
        f"a {validation['frame_max']}"
    )
    print(f"Media seguidores dentro: {validation['mean_inside']:.3f}")
    print(f"Máximo seguidores dentro: {validation['max_inside']}")
    print(f"Media seguidores fuera: {validation['mean_outside']:.3f}")
    print(
        "Frames sin posiciones interiores: "
        f"{validation['frames_without_inside_positions']}"
    )
    print(
        "Consistencia de las 9 zonas: "
        f"{validation['zone_counts_consistent']}"
    )
    print(f"Alineación correcta: {validation['alignment_ok']}")

    generate_report(
        validation,
        pitch_length_m,
        pitch_width_m,
    )

    if not validation["alignment_ok"]:
        raise RuntimeError(
            "ETAPA 03: falló la validación estructural."
        )

    print(f"\nDocumento generado: {REPORT_PATH.name}")
    print("ETAPA 03 COMPLETADA.")


if __name__ == "__main__":
    main()
