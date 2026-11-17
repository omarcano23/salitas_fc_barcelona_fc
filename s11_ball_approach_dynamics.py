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
FPS = 25

TEAMS = {
    "Barcelona": range(1, 25),
    "Atletic_Club": range(25, 47),
}


# ---------------------------------------------------------
# DINÁMICA DE APROXIMACIÓN POR EQUIPO
# ---------------------------------------------------------

def calculate_team_approach(df,
    team_name,
    slots,
    ball_x,
    ball_y,
    ball_valid,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
    fps=25):

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

    # Distancias del frame actual.
    dx = x - ball_x[:, None]
    dy = y - ball_y[:, None]

    current_distance = np.sqrt(dx**2 + dy**2)

    # Desplazamos las posiciones un frame para comparar
    # únicamente frames consecutivos.
    previous_x = np.full_like(x, np.nan)
    previous_y = np.full_like(y, np.nan)

    previous_x[1:] = x[:-1]
    previous_y[1:] = y[:-1]

    previous_position_valid = np.zeros_like(position_valid)
    previous_position_valid[1:] = position_valid[:-1]

    previous_ball_x = np.full(len(df), np.nan)
    previous_ball_y = np.full(len(df), np.nan)

    previous_ball_x[1:] = ball_x[:-1]
    previous_ball_y[1:] = ball_y[:-1]

    previous_ball_valid = np.zeros(len(df), dtype=bool)
    previous_ball_valid[1:] = ball_valid[:-1]

    previous_dx = previous_x - previous_ball_x[:, None]
    previous_dy = previous_y - previous_ball_y[:, None]

    previous_distance = np.sqrt(previous_dx**2 + previous_dy**2)

    # Se requiere posición válida del jugador y del balón
    # en los dos frames consecutivos.
    valid_pair = (
        position_valid
        & previous_position_valid
        & ball_valid[:, None]
        & previous_ball_valid[:, None]
        & np.isfinite(current_distance)
        & np.isfinite(previous_distance)
    )

    # Positivo: se aproxima.
    # Negativo: se aleja.
    # Cero: distancia sin cambio.
    closing_speed = (
        previous_distance - current_distance
    ) * fps

    closing_speed[~valid_pair] = np.nan

    approaching = valid_pair & (closing_speed > 0)
    moving_away = valid_pair & (closing_speed < 0)

    valid_count = valid_pair.sum(axis=1)
    approaching_count = approaching.sum(axis=1)
    moving_away_count = moving_away.sum(axis=1)

    approaching_ratio = np.divide(
        approaching_count,
        valid_count,
        out=np.full(len(df), np.nan, dtype=float),
        where=valid_count > 0,
    )

    moving_away_ratio = np.divide(
        moving_away_count,
        valid_count,
        out=np.full(len(df), np.nan, dtype=float),
        where=valid_count > 0,
    )

    # Velocidad media de cierre solo entre los tracks
    # que efectivamente se aproximan.
    approach_speed_sum = np.where(
        approaching,
        closing_speed,
        0.0,
    ).sum(axis=1)

    mean_approach_speed = np.divide(
        approach_speed_sum,
        approaching_count,
        out=np.full(len(df), np.nan, dtype=float),
        where=approaching_count > 0,
    )

    return {
        f"{team_name}_approach_valid_tracks_count": valid_count,
        f"{team_name}_approaching_count": approaching_count,
        f"{team_name}_moving_away_count": moving_away_count,
        f"{team_name}_approaching_ratio": approaching_ratio,
        f"{team_name}_moving_away_ratio": moving_away_ratio,
        f"{team_name}_mean_approach_speed_mps": mean_approach_speed,
    }


def build_team_approach(
    tracking,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
    fps=25,
):
    """Construye las 12 variables de dinámica de aproximación."""

    required = ["frame", "ball_x", "ball_y"]

    teams = {
        "Barcelona": list(range(1, 25)),
        "Atletic_Club": list(range(25, 47)),
    }

    for slots in teams.values():
        for slot in slots:
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

    for team_name, slots in teams.items():
        team_features = calculate_team_approach(
            tracking,
            team_name,
            slots,
            ball_x,
            ball_y,
            ball_valid,
            pitch_length_m,
            pitch_width_m,
            fps,
        )

        for column, values in team_features.items():
            features[column] = values

    validate_features(features)

    return features


# ---------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------

def validate_features(features):

    assert len(features) == N_VIDEO_FRAMES
    assert len(features.columns) == 13  # 12 variables + frame

    assert features["frame"].iloc[0] == 0
    assert features["frame"].iloc[-1] == N_VIDEO_FRAMES - 1
    assert features["frame"].is_unique

    for team_name in TEAMS:

        valid_col = f"{team_name}_approach_valid_tracks_count"
        approaching_col = f"{team_name}_approaching_count"
        away_col = f"{team_name}_moving_away_count"
        approaching_ratio_col = f"{team_name}_approaching_ratio"
        away_ratio_col = f"{team_name}_moving_away_ratio"
        speed_col = f"{team_name}_mean_approach_speed_mps"

        valid_count = features[valid_col]
        approaching_count = features[approaching_col]
        away_count = features[away_col]

        assert (approaching_count <= valid_count).all()
        assert (away_count <= valid_count).all()
        assert ((approaching_count + away_count) <= valid_count).all()

        has_valid = valid_count > 0
        no_valid = ~has_valid

        # Ratios definidos solo cuando hay comparaciones válidas.
        assert features.loc[has_valid, approaching_ratio_col].notna().all()
        assert features.loc[has_valid, away_ratio_col].notna().all()

        assert features.loc[no_valid, approaching_ratio_col].isna().all()
        assert features.loc[no_valid, away_ratio_col].isna().all()

        for ratio_col in (approaching_ratio_col, away_ratio_col):
            values = features.loc[has_valid, ratio_col]
            assert values.between(0, 1).all()

        # Las proporciones no pueden sumar más de uno.
        ratio_sum = (
            features.loc[has_valid, approaching_ratio_col]
            + features.loc[has_valid, away_ratio_col]
        )
        assert (ratio_sum <= 1.0 + 1e-9).all()

        # La velocidad media de aproximación solo existe
        # cuando hay al menos un jugador aproximándose.
        has_approaching = approaching_count > 0
        no_approaching = ~has_approaching

        assert features.loc[has_approaching, speed_col].notna().all()
        assert features.loc[no_approaching, speed_col].isna().all()

        assert (features.loc[has_approaching, speed_col] > 0).all()

    # El primer frame no tiene un frame anterior comparable.
    for team_name in TEAMS:
        assert features.loc[0, f"{team_name}_approach_valid_tracks_count"] == 0
        assert features.loc[0, f"{team_name}_approaching_count"] == 0
        assert features.loc[0, f"{team_name}_moving_away_count"] == 0


# ---------------------------------------------------------
# DOCUMENTACIÓN
# ---------------------------------------------------------

def write_report(features, output_path):

    report = f"""# ETAPA 11 — Dinámica de aproximación al balón

## Resultado
- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Variables calculadas: 12
- Frames: {features['frame'].min()} a {features['frame'].max()}

## Variables por equipo
- Tracks con distancias comparables entre frames consecutivos.
- Jugadores aproximándose al balón.
- Jugadores alejándose del balón.
- Proporción de jugadores aproximándose.
- Proporción de jugadores alejándose.
- Velocidad media de aproximación (m/s).

## Criterio de cálculo
- Se compara la distancia jugador-balón entre dos frames consecutivos.
- Velocidad de cierre = (distancia anterior - distancia actual) × FPS.
- Velocidad positiva: aproximación.
- Velocidad negativa: alejamiento.
- Velocidad cero: distancia sin cambio.
- Solo se usan pares de frames con posiciones válidas del jugador y del balón.
- No se rellenan huecos ni se conectan observaciones no consecutivas.
- Las proporciones utilizan como denominador los tracks comparables.
- La velocidad media de aproximación considera solo jugadores aproximándose.

## Resumen

### Barcelona
- Media de tracks comparables: {features['Barcelona_approach_valid_tracks_count'].mean():.3f}
- Media de jugadores aproximándose: {features['Barcelona_approaching_count'].mean():.3f}
- Media de jugadores alejándose: {features['Barcelona_moving_away_count'].mean():.3f}
- Proporción media de aproximación: {features['Barcelona_approaching_ratio'].mean():.4f}
- Velocidad media de aproximación: {features['Barcelona_mean_approach_speed_mps'].mean():.3f} m/s

### Athletic Club
- Media de tracks comparables: {features['Atletic_Club_approach_valid_tracks_count'].mean():.3f}
- Media de jugadores aproximándose: {features['Atletic_Club_approaching_count'].mean():.3f}
- Media de jugadores alejándose: {features['Atletic_Club_moving_away_count'].mean():.3f}
- Proporción media de aproximación: {features['Atletic_Club_approaching_ratio'].mean():.4f}
- Velocidad media de aproximación: {features['Atletic_Club_mean_approach_speed_mps'].mean():.3f} m/s

## Validación
- Alineación y unicidad de frames: correctas.
- Conteos dentro de los tracks comparables: correctos.
- Proporciones entre 0 y 1: correctas.
- Proporciones de aproximación y alejamiento no superan 1 en conjunto.
- Velocidad media positiva cuando existe aproximación.
- Primer frame sin comparación anterior: correcto.

Esta etapa describe cambios geométricos de distancia al balón; no determina posesión ni intención táctica.
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

    print("Calculando dinámica de aproximación por equipo...")

    barcelona = calculate_team_approach(
        df,
        "Barcelona",
        TEAMS["Barcelona"],
        ball_x,
        ball_y,
        ball_valid,
    )

    atletic = calculate_team_approach(
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
    validate_features(features)

    print(f"Filas: {len(features)}")
    print(f"Columnas: {len(features.columns)}")
    print(
        f"Frames: {features['frame'].min()} "
        f"a {features['frame'].max()}"
    )

    for team_name in TEAMS:
        print(f"\n{team_name}:")
        print(
            "Media de tracks comparables:",
            round(
                features[
                    f"{team_name}_approach_valid_tracks_count"
                ].mean(),
                3,
            ),
        )
        print(
            "Media de jugadores aproximándose:",
            round(
                features[
                    f"{team_name}_approaching_count"
                ].mean(),
                3,
            ),
        )
        print(
            "Media de jugadores alejándose:",
            round(
                features[
                    f"{team_name}_moving_away_count"
                ].mean(),
                3,
            ),
        )
        print(
            "Velocidad media de aproximación:",
            round(
                features[
                    f"{team_name}_mean_approach_speed_mps"
                ].mean(),
                3,
            ),
            "m/s",
        )

    print("\nConteos consistentes: True")
    print("Proporciones consistentes: True")
    print("Velocidades de aproximación positivas: True")
    print("Alineación correcta: True")

    report_path = ROOT_DIR / "ETAPA_11_DINAMICA_APROXIMACION.md"
    write_report(features, report_path)

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 11 COMPLETADA.")


if __name__ == "__main__":
    main()