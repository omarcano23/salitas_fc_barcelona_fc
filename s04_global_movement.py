
from pathlib import Path

import numpy as np
import pandas as pd

from config import GAME, ROOT_DIR, N_VIDEO_FRAMES
from starter.load_atd import load


N_SLOTS = 46
MOVING_SPEED_THRESHOLD = 0.5
SUSPICIOUS_SPEED_THRESHOLD = 12.0


def row_mean(values, valid):
    counts = valid.sum(axis=1)
    totals = np.where(valid, values, 0.0).sum(axis=1)

    result = np.full(len(values), np.nan)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def row_median(values, valid):
    result = np.full(len(values), np.nan)

    for i in range(len(values)):
        row = values[i, valid[i]]
        if len(row):
            result[i] = np.median(row)

    return result


def row_max(values, valid):
    result = np.full(len(values), np.nan)

    for i in range(len(values)):
        row = values[i, valid[i]]
        if len(row):
            result[i] = np.max(row)

    return result


def build_global_movement():
    print("Cargando tracking ATD Tactical...")

    dataset = load(game=GAME, source="tactical")
    df = dataset.to_df().copy()

    if "frame_id" not in df.columns:
        df["frame_id"] = df.index + 1

    df["frame"] = df["frame_id"].astype(int) - 1
    df = df.sort_values("frame").reset_index(drop=True)

    expected_frames = np.arange(N_VIDEO_FRAMES)

    if len(df) != N_VIDEO_FRAMES:
        raise ValueError(
            f"Filas inesperadas: {len(df)}. "
            f"Se esperaban {N_VIDEO_FRAMES}."
        )

    if not np.array_equal(df["frame"].to_numpy(), expected_frames):
        raise ValueError("El tracking no está alineado con los frames de vídeo.")

    print("Calculando movimiento global por frame...")

    x_cols = [f"{slot}_x" for slot in range(1, N_SLOTS + 1)]
    y_cols = [f"{slot}_y" for slot in range(1, N_SLOTS + 1)]
    speed_cols = [f"{slot}_s" for slot in range(1, N_SLOTS + 1)]

    missing = [
        col for col in x_cols + y_cols + speed_cols
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Faltan columnas ATD: {missing}")

    pitch_length_m = float(
        dataset.metadata.pitch_dimensions.pitch_length
    )
    pitch_width_m = float(
        dataset.metadata.pitch_dimensions.pitch_width
    )

    # Posiciones normalizadas convertidas a metros.
    x_m = df[x_cols].to_numpy(dtype=float) * pitch_length_m
    y_m = df[y_cols].to_numpy(dtype=float) * pitch_width_m
    speeds = df[speed_cols].to_numpy(dtype=float)

    position_valid = np.isfinite(x_m) & np.isfinite(y_m)
    speed_valid = np.isfinite(speeds)

    # A. Disponibilidad y cobertura.
    tracked_count = position_valid.sum(axis=1)

    position_coverage = tracked_count / N_SLOTS

    inside = (
        position_valid
        & (x_m >= 0)
        & (x_m <= pitch_length_m)
        & (y_m >= 0)
        & (y_m <= pitch_width_m)
    )

    outside_count = (position_valid & ~inside).sum(axis=1)

    outside_ratio = np.zeros(len(df), dtype=float)
    np.divide(
        outside_count,
        tracked_count,
        out=outside_ratio,
        where=tracked_count > 0
    )

    # B. Velocidad ATD, expresada en m/s.
    moving_count = (
        speed_valid & (speeds > MOVING_SPEED_THRESHOLD)
    ).sum(axis=1)

    suspicious_count = (
        speed_valid & (speeds > SUSPICIOUS_SPEED_THRESHOLD)
    ).sum(axis=1)

    speed_mean = row_mean(speeds, speed_valid)
    speed_median = row_median(speeds, speed_valid)
    speed_max = row_max(speeds, speed_valid)

    # C. Desplazamiento entre frames consecutivos del mismo slot.
    # No se interpolan huecos ni se conectan observaciones discontinuas.
    displacement = np.full_like(x_m, np.nan, dtype=float)

    consecutive_valid = (
        position_valid[1:] & position_valid[:-1]
    )

    dx = x_m[1:] - x_m[:-1]
    dy = y_m[1:] - y_m[:-1]

    distances = np.sqrt(dx ** 2 + dy ** 2)
    displacement[1:] = np.where(
        consecutive_valid,
        distances,
        np.nan
    )

    displacement_valid = np.isfinite(displacement)

    displacement_mean = row_mean(
        displacement, displacement_valid
    )
    displacement_median = row_median(
        displacement, displacement_valid
    )

    # D. Tabla final: 10 variables + frame.
    features = pd.DataFrame({
        "frame": expected_frames,
        "global_tracked_count": tracked_count.astype(np.int8),
        "global_moving_count": moving_count.astype(np.int8),
        "global_suspicious_speed_count": suspicious_count.astype(np.int8),
        "global_speed_mean": speed_mean,
        "global_speed_median": speed_median,
        "global_speed_max": speed_max,
        "global_displacement_mean_m": displacement_mean,
        "global_displacement_median_m": displacement_median,
        "global_position_coverage": position_coverage,
        "global_outside_position_ratio": outside_ratio,
    })

    return features, pitch_length_m, pitch_width_m


def validate(features):
    expected_frames = np.arange(N_VIDEO_FRAMES)

    assert len(features) == N_VIDEO_FRAMES
    assert features["frame"].is_unique
    assert np.array_equal(features["frame"].to_numpy(), expected_frames)
    assert features.shape[1] == 11

    assert features["global_tracked_count"].between(
        0, N_SLOTS
    ).all()

    assert features["global_moving_count"].between(
        0, N_SLOTS
    ).all()

    assert features["global_suspicious_speed_count"].between(
        0, N_SLOTS
    ).all()

    assert features["global_position_coverage"].between(
        0, 1
    ).all()

    assert features["global_outside_position_ratio"].between(
        0, 1
    ).all()

    return True


def write_report(features, pitch_length_m, pitch_width_m):
    report_path = ROOT_DIR / "ETAPA_04_MOVIMIENTO_GLOBAL.md"

    report = f"""# Etapa 04 — Movimiento global

## Resultado
- Filas: {len(features):,}
- Columnas: {len(features.columns)}
- Frames: {features['frame'].min()} a {features['frame'].max()}
- Alineación: correcta

## Configuración
- Dimensiones utilizadas: {pitch_length_m:.2f} x {pitch_width_m:.2f} m
- Slots ATD: {N_SLOTS}
- Umbral de movimiento: > {MOVING_SPEED_THRESHOLD} m/s
- Umbral de velocidad sospechosa: > {SUSPICIOUS_SPEED_THRESHOLD} m/s

## Resumen
- Media de slots con posición: {features['global_tracked_count'].mean():.3f}
- Media de slots en movimiento: {features['global_moving_count'].mean():.3f}
- Media de velocidades: {features['global_speed_mean'].mean():.3f} m/s
- Velocidad máxima observada: {features['global_speed_max'].max():.3f} m/s
- Media de cobertura de posiciones: {features['global_position_coverage'].mean():.3f}
- Media de proporción fuera del campo: {features['global_outside_position_ratio'].mean():.5f}

## Criterios
- La velocidad procede del campo ATD `_s`.
- El desplazamiento se calcula entre frames consecutivos del mismo slot.
- No se interpolan posiciones ausentes.
- Los valores ausentes se conservan como NaN.

## Columnas generadas
{chr(10).join(f'- `{col}`' for col in features.columns)}

## Validación
- 199.319 filas y 11 columnas.
- Frames consecutivos desde 0 hasta 199.318.
- Conteos dentro del rango 0–46.
- Cobertura y proporciones dentro del rango 0–1.

**ETAPA 04 COMPLETADA.**
"""

    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    features, pitch_length_m, pitch_width_m = build_global_movement()

    print("\nValidando estructura...")
    validate(features)

    report_path = write_report(
        features,
        pitch_length_m,
        pitch_width_m
    )

    print("\nResultado de movimiento global:")
    print(f"Filas: {len(features):,}")
    print(f"Columnas: {len(features.columns)}")
    print(
        f"Frames: {features['frame'].min()} "
        f"a {features['frame'].max()}"
    )
    print(
        "Media de slots con posición: "
        f"{features['global_tracked_count'].mean():.3f}"
    )
    print(
        "Media de slots en movimiento: "
        f"{features['global_moving_count'].mean():.3f}"
    )
    print(
        "Velocidad máxima: "
        f"{features['global_speed_max'].max():.3f} m/s"
    )
    print("Alineación correcta: True")
    print(f"\nDocumento generado: {report_path.name}")
    print("ETAPA 04 COMPLETADA.")


if __name__ == "__main__":
    main()
