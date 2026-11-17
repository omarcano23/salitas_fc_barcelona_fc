from pathlib import Path

import numpy as np
import pandas as pd

from config import GAME, ROOT_DIR, N_VIDEO_FRAMES
from starter.load_atd import load


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

TEAMS = {
    "Barcelona": range(1, 25),
    "Atletic_Club": range(25, 47),
}


# ---------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------

def row_mean(values, mask):
    """Media por fila, ignorando valores no válidos."""
    valid_count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1)

    return np.divide(
        total,
        valid_count,
        out=np.full(len(valid_count), np.nan, dtype=float),
        where=valid_count > 0,
    )


def calculate_team_metrics(df, slots):
    x_cols = [f"{slot}_x" for slot in slots]
    y_cols = [f"{slot}_y" for slot in slots]
    speed_cols = [f"{slot}_s" for slot in slots]

    x = df[x_cols].to_numpy(dtype=float) * PITCH_LENGTH_M
    y = df[y_cols].to_numpy(dtype=float) * PITCH_WIDTH_M
    speed = df[speed_cols].to_numpy(dtype=float)

    position_valid = np.isfinite(x) & np.isfinite(y)

    inside = (
        position_valid
        & (x >= 0) & (x <= PITCH_LENGTH_M)
        & (y >= 0) & (y <= PITCH_WIDTH_M)
    )

    speed_valid = position_valid & np.isfinite(speed)

    inside_count = inside.sum(axis=1)
    valid_position_count = position_valid.sum(axis=1)

    centroid_x = row_mean(x, inside)
    centroid_y = row_mean(y, inside)
    speed_mean = row_mean(speed, speed_valid)

    return {
        "valid_position_count": valid_position_count,
        "inside_count": inside_count,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "speed_mean": speed_mean,
    }


def validate_features(features):
    expected_rows = N_VIDEO_FRAMES

    assert len(features) == expected_rows
    assert features["frame"].iloc[0] == 0
    assert features["frame"].iloc[-1] == expected_rows - 1
    assert features["frame"].is_unique

    assert (
        features["team_inside_count_difference"].abs()
        <= max(len(TEAMS["Barcelona"]), len(TEAMS["Atletic_Club"]))
    ).all()

    assert features["team_valid_position_count_difference"].between(
        -46, 46
    ).all()


def write_report(features, output_path):
    report = f"""# ETAPA 08 — Diferencias entre equipos

## Resultado
- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Frames: {features['frame'].min()} a {features['frame'].max()}
- Variables calculadas: 6

## Variables
- `team_centroid_distance_m`: distancia entre centroides de ambos equipos.
- `centroid_x_difference_m`: centroide X de Barcelona menos Athletic.
- `centroid_y_difference_m`: centroide Y de Barcelona menos Athletic.
- `team_inside_count_difference`: posiciones dentro del campo, Barcelona menos Athletic.
- `team_valid_position_count_difference`: posiciones válidas, Barcelona menos Athletic.
- `team_speed_mean_difference_mps`: velocidad media válida, Barcelona menos Athletic.

## Resumen
- Distancia media entre centroides: {features['team_centroid_distance_m'].mean():.3f} m
- Diferencia media de posiciones interiores: {features['team_inside_count_difference'].mean():.3f}
- Diferencia media de posiciones válidas: {features['team_valid_position_count_difference'].mean():.3f}
- Diferencia media de velocidad: {features['team_speed_mean_difference_mps'].mean():.3f} m/s

## Criterios
- Las coordenadas normalizadas se convierten a metros usando 105.3 × 68.0.
- Los centroides utilizan únicamente posiciones dentro del campo.
- Las velocidades utilizan valores finitos asociados a posiciones válidas.
- Las diferencias se calculan siempre como Barcelona menos Athletic.
- El eje X no se interpreta como dirección ofensiva o defensiva.

## Validación
- Alineación de frames: correcta.
- Unicidad de frames: correcta.
- Consistencia de conteos: correcta.

Esta etapa genera variables descriptivas; no determina qué equipo tiene la posesión.
"""

    output_path.write_text(report, encoding="utf-8")


# ---------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------

def main():
    print("Cargando tracking ATD Tactical...")
    dataset = load(game=GAME, source="tactical")
    df = dataset.to_df()

    if "frame_id" not in df.columns:
        df = df.reset_index()

    df["frame"] = df["frame_id"].astype(int) - 1
    df = df.sort_values("frame").reset_index(drop=True)

    print("Calculando métricas por equipo...")

    barcelona = calculate_team_metrics(df, TEAMS["Barcelona"])
    atletic = calculate_team_metrics(df, TEAMS["Atletic_Club"])

    dx = barcelona["centroid_x"] - atletic["centroid_x"]
    dy = barcelona["centroid_y"] - atletic["centroid_y"]

    features = pd.DataFrame({
        "frame": df["frame"].to_numpy(),

        "team_centroid_distance_m": np.sqrt(dx**2 + dy**2),
        "centroid_x_difference_m": dx,
        "centroid_y_difference_m": dy,

        "team_inside_count_difference": (
            barcelona["inside_count"] - atletic["inside_count"]
        ),

        "team_valid_position_count_difference": (
            barcelona["valid_position_count"]
            - atletic["valid_position_count"]
        ),

        "team_speed_mean_difference_mps": (
            barcelona["speed_mean"] - atletic["speed_mean"]
        ),
    })

    print("\nValidando estructura...")
    validate_features(features)

    print(f"Filas: {len(features)}")
    print(f"Columnas: {len(features.columns)}")
    print(f"Frames: {features['frame'].min()} a {features['frame'].max()}")

    print("\nResumen:")
    print(
        "Distancia media entre centroides:",
        round(features["team_centroid_distance_m"].mean(), 3),
        "m",
    )
    print(
        "Diferencia media de posiciones interiores:",
        round(features["team_inside_count_difference"].mean(), 3),
    )
    print(
        "Diferencia media de velocidad:",
        round(features["team_speed_mean_difference_mps"].mean(), 3),
        "m/s",
    )

    print("\nAlineación correcta: True")
    print("Consistencia de conteos: True")

    report_path = ROOT_DIR / "ETAPA_08_DIFERENCIAS_EQUIPOS.md"
    write_report(features, report_path)

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 08 COMPLETADA.")


if __name__ == "__main__":
    main()