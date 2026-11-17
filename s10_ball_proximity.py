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

DISTANCE_THRESHOLDS = (1.5, 3.0, 5.0, 10.0)


# ---------------------------------------------------------
# PROXIMIDAD DEL BALÓN POR EQUIPO
# ---------------------------------------------------------

def calculate_team_proximity(df,
    team_name,
    slots,
    ball_x,
    ball_y,
    ball_valid,
    pitch_length_m=105.3,
    pitch_width_m=68.0,):

    x_cols = [f"{slot}_x" for slot in slots]
    y_cols = [f"{slot}_y" for slot in slots]

    x = df[x_cols].to_numpy(dtype=float) * pitch_length_m
    y = df[y_cols].to_numpy(dtype=float) * pitch_width_m

    position_valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= pitch_length_m)
        & (y >= 0)
        & (y <= pitch_width_m)
    )

    valid_position_count = position_valid.sum(axis=1)

    # Una distancia solo es válida si el balón está observado
    # dentro del campo y el jugador tiene posición válida.
    distance_valid = position_valid & ball_valid[:, None]

    dx = x - ball_x[:, None]
    dy = y - ball_y[:, None]

    distances = np.sqrt(dx**2 + dy**2)
    distances[~distance_valid] = np.nan

    valid_distance_count = distance_valid.sum(axis=1)

    nearest_distance = np.full(len(df), np.nan, dtype=float)
    mean_distance = np.full(len(df), np.nan, dtype=float)

    has_distances = valid_distance_count > 0

    nearest_distance[has_distances] = np.nanmin(
        distances[has_distances],
        axis=1,
    )

    mean_distance[has_distances] = np.nanmean(
        distances[has_distances],
        axis=1,
    )

    result = {
        f"{team_name}_valid_position_count": valid_position_count,
        f"{team_name}_valid_distance_count": valid_distance_count,
        f"{team_name}_nearest_ball_distance_m": nearest_distance,
        f"{team_name}_mean_ball_distance_m": mean_distance,
    }

    for threshold in DISTANCE_THRESHOLDS:
        count = (
            distance_valid
            & (distances <= threshold)
        ).sum(axis=1)

        threshold_name = str(threshold).replace(".", "_")

        result[
            f"{team_name}_players_within_{threshold_name}m"
        ] = count

    return result


def build_ball_proximity(
    tracking,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
):
    """Construye las 16 variables de proximidad del balón por equipo."""

    required = ["frame", "ball_x", "ball_y"]

    for slot in list(range(1, 25)) + list(range(25, 47)):
        required.extend([f"{slot}_x", f"{slot}_y"])

    missing = [col for col in required if col not in tracking.columns]
    assert not missing, f"Faltan columnas: {missing}"

    ball_x = tracking["ball_x"].to_numpy(dtype=float)
    ball_y = tracking["ball_y"].to_numpy(dtype=float)

    ball_valid = (
        np.isfinite(ball_x)
        & np.isfinite(ball_y)
        & (ball_x >= 0)
        & (ball_x <= 1)
        & (ball_y >= 0)
        & (ball_y <= 1)
    )

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy()
    })

    teams = {
        "Barcelona": list(range(1, 25)),
        "Atletic_Club": list(range(25, 47)),
    }

    for team_name, slots in teams.items():
        team_features = calculate_team_proximity(
            tracking,
            team_name,
            slots,
            ball_x,
            ball_y,
            ball_valid,
            pitch_length_m,
            pitch_width_m,
        )

        for column, values in team_features.items():
            features[column] = values

    validate_features(features, ball_valid)

    return features


# ---------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------

def validate_features(features, ball_valid):

    assert len(features) == N_VIDEO_FRAMES
    assert len(features.columns) == 17  # 16 variables + frame

    assert features["frame"].iloc[0] == 0
    assert features["frame"].iloc[-1] == N_VIDEO_FRAMES - 1
    assert features["frame"].is_unique

    for team_name in TEAMS:

        position_col = f"{team_name}_valid_position_count"
        distance_col = f"{team_name}_valid_distance_count"
        nearest_col = f"{team_name}_nearest_ball_distance_m"
        mean_col = f"{team_name}_mean_ball_distance_m"

        assert (
            features[distance_col] <= features[position_col]
        ).all()

        assert (
            features[distance_col].to_numpy()[~ball_valid] == 0
        ).all()

        assert features.loc[~ball_valid, nearest_col].isna().all()
        assert features.loc[~ball_valid, mean_col].isna().all()

        for threshold in DISTANCE_THRESHOLDS:
            threshold_name = str(threshold).replace(".", "_")
            count_col = (
                f"{team_name}_players_within_{threshold_name}m"
            )

            assert (
                features[count_col] <= features[distance_col]
            ).all()

        # Los conteos deben ser monotónicos:
        # dentro de 1.5 m <= 3 m <= 5 m <= 10 m.
        counts = [
            features[
                f"{team_name}_players_within_"
                f"{str(t).replace('.', '_')}m"
            ].to_numpy()
            for t in DISTANCE_THRESHOLDS
        ]

        for smaller, larger in zip(counts, counts[1:]):
            assert (smaller <= larger).all()

        has_distances = features[distance_col] > 0

        assert features.loc[has_distances, nearest_col].notna().all()
        assert features.loc[has_distances, mean_col].notna().all()

        assert (
            features.loc[has_distances, nearest_col]
            <= features.loc[has_distances, mean_col] + 1e-9
        ).all()


# ---------------------------------------------------------
# DOCUMENTACIÓN
# ---------------------------------------------------------

def write_report(features, ball_valid, output_path):

    report = f"""# ETAPA 10 — Proximidad del balón

## Resultado
- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Variables calculadas: 16
- Frames: {features['frame'].min()} a {features['frame'].max()}

## Disponibilidad del balón
- Frames con balón observado dentro del campo: {int(ball_valid.sum())}
- Proporción: {ball_valid.mean():.4%}

## Variables por equipo
- Número de posiciones válidas dentro del campo.
- Número de distancias válidas al balón.
- Distancia mínima al balón (m).
- Distancia media al balón (m).
- Jugadores a menos de 1.5 m.
- Jugadores a menos de 3 m.
- Jugadores a menos de 5 m.
- Jugadores a menos de 10 m.

## Resumen

### Barcelona
- Distancia mínima media: {features['Barcelona_nearest_ball_distance_m'].mean():.3f} m
- Distancia media al balón: {features['Barcelona_mean_ball_distance_m'].mean():.3f} m
- Jugadores medios a menos de 3 m: {features['Barcelona_players_within_3_0m'].mean():.3f}

### Athletic Club
- Distancia mínima media: {features['Atletic_Club_nearest_ball_distance_m'].mean():.3f} m
- Distancia media al balón: {features['Atletic_Club_mean_ball_distance_m'].mean():.3f} m
- Jugadores medios a menos de 3 m: {features['Atletic_Club_players_within_3_0m'].mean():.3f}

## Criterios
- Coordenadas convertidas a metros usando 105.3 × 68.0.
- Solo se consideran posiciones de jugadores dentro del campo.
- El balón debe estar observado y dentro del campo.
- No se imputan posiciones ni distancias.
- Si no hay distancias válidas, las distancias mínima y media son NaN.
- Los conteos de proximidad usan límites inclusivos (<=).
- Los umbrales son medidas geométricas, no reglas de posesión.

## Validación
- Alineación y unicidad de frames: correctas.
- Distancias solo cuando el balón y el jugador son válidos: correcto.
- Conteos dentro de los límites de distancias válidas: correcto.
- Monotonía de los umbrales: correcta.
- Distancia mínima menor o igual que la media: correcta.

Esta etapa describe proximidad espacial; no determina posesión.
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

    print("Calculando disponibilidad y posición del balón...")

    ball_x_raw = df["ball_x"].to_numpy(dtype=float)
    ball_y_raw = df["ball_y"].to_numpy(dtype=float)

    ball_valid = (
        np.isfinite(ball_x_raw)
        & np.isfinite(ball_y_raw)
        & (ball_x_raw >= 0)
        & (ball_x_raw <= 1)
        & (ball_y_raw >= 0)
        & (ball_y_raw <= 1)
    )

    ball_x = ball_x_raw * PITCH_LENGTH_M
    ball_y = ball_y_raw * PITCH_WIDTH_M

    print("Calculando proximidad por equipo...")

    barcelona = calculate_team_proximity(
        df,
        "Barcelona",
        TEAMS["Barcelona"],
        ball_x,
        ball_y,
        ball_valid,
    )

    atletic = calculate_team_proximity(
        df,
        "Atletic_Club",
        TEAMS["Atletic_Club"],
        ball_x,
        ball_y,
        ball_valid,
    )

    features = pd.DataFrame({
        "frame": df["frame"].to_numpy(),
    })

    for team_data in (barcelona, atletic):
        for column, values in team_data.items():
            features[column] = values

    print("\nValidando estructura...")
    validate_features(features, ball_valid)

    print(f"Filas: {len(features)}")
    print(f"Columnas: {len(features.columns)}")
    print(
        f"Frames: {features['frame'].min()} "
        f"a {features['frame'].max()}"
    )

    print(
        "\nFrames con balón observado dentro del campo:",
        int(ball_valid.sum()),
    )

    for team_name in TEAMS:
        print(f"\n{team_name}:")
        print(
            "Media de distancias válidas por frame:",
            round(
                features[
                    f"{team_name}_valid_distance_count"
                ].mean(),
                3,
            ),
        )
        print(
            "Distancia mínima media:",
            round(
                features[
                    f"{team_name}_nearest_ball_distance_m"
                ].mean(),
                3,
            ),
        )
        print(
            "Jugadores medios a menos de 3 m:",
            round(
                features[
                    f"{team_name}_players_within_3_0m"
                ].mean(),
                3,
            ),
        )

    print("\nConteos consistentes: True")
    print("Umbrales monotónicos: True")
    print("Alineación correcta: True")

    report_path = ROOT_DIR / "ETAPA_10_PROXIMIDAD_BALON.md"
    write_report(features, ball_valid, report_path)

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 10 COMPLETADA.")


if __name__ == "__main__":
    main()