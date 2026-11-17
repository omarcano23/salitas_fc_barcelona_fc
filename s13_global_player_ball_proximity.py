
import numpy as np
import pandas as pd

from s01_tracking import load_tracking
from config import GAME, ROOT_DIR, N_VIDEO_FRAMES
from starter.load_atd import load


PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

PLAYER_SLOTS = range(1, 47)

DISTANCE_THRESHOLDS_M = {
    "global_players_within_1_5m": 1.5,
    "global_players_within_3m": 3.0,
    "global_players_within_5m": 5.0,
    "global_players_within_10m": 10.0,
}

EXPECTED_COLUMNS = 7  # frame + 6 variables


def validate_features(features, ball_valid):

    expected_columns = [
        "frame",
        "global_players_within_1_5m",
        "global_players_within_3m",
        "global_players_within_5m",
        "global_players_within_10m",
        "global_nearest_player_slot",
        "global_nearest_player_distance_m",
    ]

    # Estructura
    assert len(features) == N_VIDEO_FRAMES
    assert len(features.columns) == EXPECTED_COLUMNS
    assert features.columns.tolist() == expected_columns

    # Alineación
    expected_frames = np.arange(N_VIDEO_FRAMES)

    assert np.array_equal(
        features["frame"].to_numpy(),
        expected_frames
    )

    assert features["frame"].is_unique

    # Cuando el balón no es válido, no se puede calcular proximidad.
    unavailable = ~ball_valid

    proximity_columns = expected_columns[1:]

    for col in proximity_columns:
        assert features.loc[unavailable, col].isna().all()

    # Cuando el balón es válido, los contadores deben estar definidos.
    count_columns = list(DISTANCE_THRESHOLDS_M.keys())

    assert features.loc[ball_valid, count_columns].notna().all().all()

    # Los contadores deben ser enteros entre 0 y 46.
    for col in count_columns:
        values = features.loc[ball_valid, col]

        assert (values >= 0).all()
        assert (values <= len(PLAYER_SLOTS)).all()
        assert np.equal(values, np.floor(values)).all()

    # La cantidad de jugadores debe ser monotónica
    # al aumentar el radio de búsqueda.
    assert (
        features.loc[ball_valid, "global_players_within_1_5m"]
        <= features.loc[ball_valid, "global_players_within_3m"]
    ).all()

    assert (
        features.loc[ball_valid, "global_players_within_3m"]
        <= features.loc[ball_valid, "global_players_within_5m"]
    ).all()

    assert (
        features.loc[ball_valid, "global_players_within_5m"]
        <= features.loc[ball_valid, "global_players_within_10m"]
    ).all()

    # Consistencia entre slot más cercano y distancia.
    nearest_slot = features["global_nearest_player_slot"]
    nearest_distance = features["global_nearest_player_distance_m"]

    assert nearest_slot.notna().equals(nearest_distance.notna())

    available_nearest = nearest_distance.notna()

    assert (
        nearest_distance.loc[available_nearest] >= 0
    ).all()

    assert (
        nearest_slot.loc[available_nearest].between(1, 46)
    ).all()

    # Si hay un jugador más cercano, debe estar dentro de 10 m
    # cuando el contador correspondiente sea mayor que cero.
    has_players_within_10m = (
        features["global_players_within_10m"] > 0
    )

    assert (
        features.loc[
            has_players_within_10m,
            "global_nearest_player_distance_m"
        ] <= 10.0
    ).all()

    return True


def write_report(features, ball_valid, output_path):

    valid_nearest = features[
        "global_nearest_player_distance_m"
    ].dropna()

    valid_counts = features.loc[ball_valid]

    report = f"""# ETAPA 13 — Proximidad global jugador-balón

## Resultado

- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Variables calculadas: 6
- Frames: {features['frame'].min()} a {features['frame'].max()}
- Alineación: correcta

## Disponibilidad

- Frames con balón válido: {int(ball_valid.sum())}
- Frames con al menos un jugador cercano a 10 m: {int((features['global_players_within_10m'] > 0).sum())}
- Frames con jugador más cercano identificable: {len(valid_nearest)}

## Variables generadas

- global_players_within_1_5m
- global_players_within_3m
- global_players_within_5m
- global_players_within_10m
- global_nearest_player_slot
- global_nearest_player_distance_m

## Resumen de proximidad

- Media de jugadores a 1.5 m: {valid_counts['global_players_within_1_5m'].mean():.3f}
- Media de jugadores a 3 m: {valid_counts['global_players_within_3m'].mean():.3f}
- Media de jugadores a 5 m: {valid_counts['global_players_within_5m'].mean():.3f}
- Media de jugadores a 10 m: {valid_counts['global_players_within_10m'].mean():.3f}

## Jugador más cercano

- Distancia media: {valid_nearest.mean():.3f} m
- Distancia mediana: {valid_nearest.median():.3f} m
- Distancia máxima: {valid_nearest.max():.3f} m

## Criterios de cálculo

- Se utilizan los 46 slots de tracking de ambos equipos.
- Las coordenadas normalizadas se convierten a metros con dimensiones de 105.3 × 68.0.
- Solo se consideran posiciones de jugadores finitas y dentro del campo.
- El balón debe tener coordenadas válidas dentro del rango normalizado [0, 1].
- Las distancias se calculan mediante distancia euclidiana en el plano de juego.
- Los radios de proximidad son 1.5, 3, 5 y 10 metros.
- El slot más cercano corresponde al índice del tracking, no a una identidad estable.
- No se interpolan posiciones ni se imputan distancias.
- Si no hay balón válido, las variables de proximidad quedan como NaN.
- Si hay balón válido pero ningún jugador válido, los contadores son cero y el jugador más cercano queda sin determinar.

## Validaciones

- Número de filas y columnas: correcto.
- Nombres y orden de columnas: correctos.
- Alineación y unicidad de frames: correctas.
- Contadores dentro del rango permitido: correctos.
- Monotonía de radios de proximidad: correcta.
- Consistencia de slot y distancia mínima: correcta.
- Ausencia de imputación: correcta.

## Consideraciones

Los slots de tracking no representan identidades persistentes de jugadores. El jugador más cercano debe interpretarse como el slot más próximo en ese frame.

La proximidad no determina por sí sola la posesión. Es una característica espacial que deberá combinarse con otras señales en los indicadores y el motor de decisión.
"""

    output_path.write_text(report, encoding="utf-8")

def build_global_player_ball_proximity(
    tracking,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
):
    """
    Construye las 6 características globales de proximidad
    jugador-balón a partir del tracking ya cargado.
    """

    n = len(tracking)
    frames = tracking["frame"].to_numpy()

    # Coordenadas del balón normalizadas.
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

    # Convertir el balón a metros.
    ball_x_m = ball_x * pitch_length_m
    ball_y_m = ball_y * pitch_width_m

    # Matriz de distancias: una columna por cada slot.
    distances = np.full((n, 46), np.nan, dtype=float)

    for slot in range(1, 47):

        x_col = f"{slot}_x"
        y_col = f"{slot}_y"

        player_x = tracking[x_col].to_numpy(dtype=float)
        player_y = tracking[y_col].to_numpy(dtype=float)

        player_valid = (
            np.isfinite(player_x)
            & np.isfinite(player_y)
            & (player_x >= 0)
            & (player_x <= 1)
            & (player_y >= 0)
            & (player_y <= 1)
        )

        comparable = ball_valid & player_valid

        player_x_m = player_x * pitch_length_m
        player_y_m = player_y * pitch_width_m

        dx = player_x_m - ball_x_m
        dy = player_y_m - ball_y_m

        distances[comparable, slot - 1] = np.sqrt(
            dx[comparable] ** 2 + dy[comparable] ** 2
        )

    # Contar jugadores dentro de cada radio.
    counts_1_5 = np.full(n, np.nan)
    counts_3 = np.full(n, np.nan)
    counts_5 = np.full(n, np.nan)
    counts_10 = np.full(n, np.nan)

    # Identificar al jugador más cercano.
    nearest_slot = np.full(n, np.nan)
    nearest_distance = np.full(n, np.nan)

    for i in range(n):

        if not ball_valid[i]:
            continue

        valid_distances = np.isfinite(distances[i])

        if not valid_distances.any():
            # Balón válido, pero ningún jugador comparable.
            counts_1_5[i] = 0
            counts_3[i] = 0
            counts_5[i] = 0
            counts_10[i] = 0
            continue

        row_distances = distances[i, valid_distances]

        counts_1_5[i] = np.sum(row_distances <= 1.5)
        counts_3[i] = np.sum(row_distances <= 3.0)
        counts_5[i] = np.sum(row_distances <= 5.0)
        counts_10[i] = np.sum(row_distances <= 10.0)

        valid_slots = np.where(valid_distances)[0]
        nearest_local_index = np.argmin(row_distances)
        nearest_global_index = valid_slots[nearest_local_index]

        nearest_slot[i] = nearest_global_index + 1
        nearest_distance[i] = distances[i, nearest_global_index]

    features = pd.DataFrame({
        "frame": frames,

        "global_players_within_1_5m": counts_1_5,
        "global_players_within_3m": counts_3,
        "global_players_within_5m": counts_5,
        "global_players_within_10m": counts_10,

        "global_nearest_player_slot": nearest_slot,
        "global_nearest_player_distance_m": nearest_distance,
    })

    validate_features(features, ball_valid)

    return features


def main():

    print("Cargando tracking ATD Tactical...")

    tracking, dataset = load_tracking()
    df = tracking.copy()

    print("Calculando proximidad global jugador-balón...")

    features = build_global_player_ball_proximity(
        tracking,
        pitch_length_m=105.3,
        pitch_width_m=68.0,
    )

    # Coordenadas originales del balón.
    ball_x = df["ball_x"].to_numpy(dtype=float)
    ball_y = df["ball_y"].to_numpy(dtype=float)

    ball_valid = (
        np.isfinite(ball_x)
        & np.isfinite(ball_y)
        & (ball_x >= 0)
        & (ball_x <= 1)
        & (ball_y >= 0)
        & (ball_y <= 1)
    )

    # Coordenadas métricas del balón.
    ball_x_m = ball_x * PITCH_LENGTH_M
    ball_y_m = ball_y * PITCH_WIDTH_M

    # Los resultados solo se calculan cuando el balón es válido.
    counts = {
        name: np.full(len(df), np.nan)
        for name in DISTANCE_THRESHOLDS_M
    }

    for values in counts.values():
        values[ball_valid] = 0

    nearest_distance = np.full(len(df), np.nan)
    nearest_slot = np.full(len(df), np.nan)

    # Recorrido de los 46 slots.
    for slot in PLAYER_SLOTS:

        player_x = df[f"{slot}_x"].to_numpy(dtype=float)
        player_y = df[f"{slot}_y"].to_numpy(dtype=float)

        player_valid = (
            np.isfinite(player_x)
            & np.isfinite(player_y)
            & (player_x >= 0)
            & (player_x <= 1)
            & (player_y >= 0)
            & (player_y <= 1)
        )

        comparable = ball_valid & player_valid

        if not comparable.any():
            continue

        player_x_m = player_x * PITCH_LENGTH_M
        player_y_m = player_y * PITCH_WIDTH_M

        dx = player_x_m[comparable] - ball_x_m[comparable]
        dy = player_y_m[comparable] - ball_y_m[comparable]

        distances = np.sqrt(dx**2 + dy**2)

        comparable_indices = np.where(comparable)[0]

        # Contadores por radio.
        for name, radius in DISTANCE_THRESHOLDS_M.items():
            counts[name][comparable_indices] += (
                distances <= radius
            ).astype(int)

        # Actualizar jugador más cercano.
        previous_nearest = nearest_distance[comparable_indices]

        update_nearest = (
            ~np.isfinite(previous_nearest)
            | (distances < previous_nearest)
        )

        update_indices = comparable_indices[update_nearest]

        nearest_distance[update_indices] = distances[update_nearest]
        nearest_slot[update_indices] = slot

    print("Construyendo matriz de características...")

    features = pd.DataFrame({

        "frame": df["frame"].to_numpy(),

        "global_players_within_1_5m": (
            counts["global_players_within_1_5m"]
        ),

        "global_players_within_3m": (
            counts["global_players_within_3m"]
        ),

        "global_players_within_5m": (
            counts["global_players_within_5m"]
        ),

        "global_players_within_10m": (
            counts["global_players_within_10m"]
        ),

        "global_nearest_player_slot": nearest_slot,

        "global_nearest_player_distance_m": nearest_distance,
    })

    print("\nValidando estructura...")

    validate_features(features, ball_valid)

    print("Validación estructural: correcta")

    print(f"\nFilas: {len(features)}")
    print(f"Columnas: {len(features.columns)}")
    print(
        f"Frames: {features['frame'].min()} "
        f"a {features['frame'].max()}"
    )

    print("\nProximidad global:")

    for col in DISTANCE_THRESHOLDS_M:
        print(
            f"Media {col}: "
            f"{features[col].mean():.3f}"
        )

    valid_nearest = features[
        "global_nearest_player_distance_m"
    ].dropna()

    print(
        "Distancia media al jugador más cercano:",
        round(valid_nearest.mean(), 3)
    )

    print(
        "Distancia mediana al jugador más cercano:",
        round(valid_nearest.median(), 3)
    )

    print(
        "Frames con jugador más cercano identificable:",
        len(valid_nearest)
    )

    print("\nAlineación correcta: True")
    print("Contadores consistentes: True")
    print("Proximidad sin imputación: True")

    report_path = ROOT_DIR / "ETAPA_13_PROXIMIDAD_GLOBAL.md"

    write_report(features, ball_valid, report_path)

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 13 COMPLETADA.")


if __name__ == "__main__":
    main()
