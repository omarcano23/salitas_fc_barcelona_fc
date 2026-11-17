
import numpy as np
import pandas as pd

from config import GAME, ROOT_DIR, N_VIDEO_FRAMES
from starter.load_atd import load


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

TEAMS = {
    "Barcelona": range(1, 25),
    "Atletic Club": range(25, 47),
}


# --------------------------------------------------
# CÁLCULOS
# --------------------------------------------------

def calculate_team_geometry(x, y):

    n_frames = len(x)

    valid = np.isfinite(x) & np.isfinite(y)

    inside = (
        valid
        & (x >= 0)
        & (x <= PITCH_LENGTH_M)
        & (y >= 0)
        & (y <= PITCH_WIDTH_M)
    )

    outside = valid & ~inside

    valid_count = valid.sum(axis=1)
    inside_count = inside.sum(axis=1)
    outside_count = outside.sum(axis=1)

    # Las estadísticas espaciales se calculan solo
    # con posiciones dentro del terreno de juego.
    x_inside = np.where(inside, x, np.nan)
    y_inside = np.where(inside, y, np.nan)

    stats = {
        "centroid_x_m": np.full(n_frames, np.nan),
        "centroid_y_m": np.full(n_frames, np.nan),
        "spread_x_m": np.full(n_frames, np.nan),
        "spread_y_m": np.full(n_frames, np.nan),
        "min_x_m": np.full(n_frames, np.nan),
        "max_x_m": np.full(n_frames, np.nan),
        "min_y_m": np.full(n_frames, np.nan),
        "max_y_m": np.full(n_frames, np.nan),
    }

    for i in range(n_frames):

        mask = inside[i]

        if not mask.any():
            continue

        xi = x[i, mask]
        yi = y[i, mask]

        stats["centroid_x_m"][i] = np.mean(xi)
        stats["centroid_y_m"][i] = np.mean(yi)

        stats["spread_x_m"][i] = np.std(xi)
        stats["spread_y_m"][i] = np.std(yi)

        stats["min_x_m"][i] = np.min(xi)
        stats["max_x_m"][i] = np.max(xi)

        stats["min_y_m"][i] = np.min(yi)
        stats["max_y_m"][i] = np.max(yi)

    return {
        "valid_position_count": valid_count,
        "inside_slots_count": inside_count,
        "outside_slots_count": outside_count,
        **stats,
    }


# --------------------------------------------------
# CONSTRUCCIÓN DE VARIABLES
# --------------------------------------------------

def build_team_geometry():

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
            f"Se esperaban {N_VIDEO_FRAMES} filas y hay {len(df)}."
        )

    if not np.array_equal(df["frame"].to_numpy(), expected_frames):
        raise ValueError("El tracking no está alineado por frame.")

    print("Calculando geometría por equipo...")

    features = pd.DataFrame({
        "frame": expected_frames
    })

    for team_name, slots in TEAMS.items():

        slot_list = list(slots)

        x_cols = [f"{slot}_x" for slot in slot_list]
        y_cols = [f"{slot}_y" for slot in slot_list]

        missing = [
            col for col in x_cols + y_cols
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Faltan columnas de tracking: {missing}"
            )

        x_norm = df[x_cols].to_numpy(dtype=float)
        y_norm = df[y_cols].to_numpy(dtype=float)

        x_m = x_norm * PITCH_LENGTH_M
        y_m = y_norm * PITCH_WIDTH_M

        team_features = calculate_team_geometry(x_m, y_m)

        prefix = team_name.replace(" ", "_")

        for name, values in team_features.items():
            features[f"{prefix}_{name}"] = values

    return features


# --------------------------------------------------
# VALIDACIÓN
# --------------------------------------------------

def validate_team_geometry(features):

    print("\nValidando estructura...")

    assert len(features) == N_VIDEO_FRAMES
    assert features.shape[1] == 23

    assert features["frame"].is_unique
    assert np.array_equal(
        features["frame"].to_numpy(),
        np.arange(N_VIDEO_FRAMES)
    )

    for team in ["Barcelona", "Atletic_Club"]:

        valid_col = f"{team}_valid_position_count"
        inside_col = f"{team}_inside_slots_count"
        outside_col = f"{team}_outside_slots_count"

        assert (
            features[inside_col] + features[outside_col]
            == features[valid_col]
        ).all()

        assert (features[valid_col] >= 0).all()
        assert (features[valid_col] <= 24 if team == "Barcelona"
                else features[valid_col] <= 22).all()

        assert (features[inside_col] >= 0).all()
        assert (features[outside_col] >= 0).all()

    print("Filas:", len(features))
    print("Columnas:", len(features.columns))
    print(
        "Frames:",
        features["frame"].min(),
        "a",
        features["frame"].max()
    )

    for team in ["Barcelona", "Atletic_Club"]:
        print(
            f"Media de posiciones válidas {team}:",
            round(
                features[f"{team}_valid_position_count"].mean(),
                3
            )
        )
        print(
            f"Media de posiciones dentro del campo {team}:",
            round(
                features[f"{team}_inside_slots_count"].mean(),
                3
            )
        )

    print("Consistencia de posiciones: True")
    print("Alineación correcta: True")

    return True


# --------------------------------------------------
# DOCUMENTACIÓN
# --------------------------------------------------

def write_report(features):

    report_path = ROOT_DIR / "ETAPA_06_GEOMETRIA_EQUIPOS.md"

    report = f"""# Etapa 06 — Geometría por equipo

## Objetivo

Calcular variables espaciales independientes para Barcelona
y Athletic Club en cada frame del vídeo táctico.

## Configuración

- Partido: {GAME}
- Frames: {N_VIDEO_FRAMES}
- Dimensiones utilizadas: {PITCH_LENGTH_M} × {PITCH_WIDTH_M} m
- Slots Barcelona: 1–24
- Slots Athletic Club: 25–46

## Variables por equipo

1. valid_position_count
2. inside_slots_count
3. outside_slots_count
4. centroid_x_m
5. centroid_y_m
6. spread_x_m
7. spread_y_m
8. min_x_m
9. max_x_m
10. min_y_m
11. max_y_m

Total: 22 variables, 11 por equipo.

## Criterios

- Las posiciones normalizadas se convierten a metros.
- Una posición válida requiere coordenadas X e Y finitas.
- Una posición interior está dentro de los límites del campo.
- Las estadísticas espaciales usan solo posiciones interiores.
- No se imputan posiciones ausentes.
- Los slots representan tracks, no identidades estables de jugadores.

## Validación estructural

- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Alineación: correcta
- Consistencia de posiciones: correcta

## Resultado

Etapa 06 completada.
"""

    report_path.write_text(report, encoding="utf-8")
    print("\nDocumento generado:", report_path.name)


# --------------------------------------------------
# EJECUCIÓN
# --------------------------------------------------

def main():

    features = build_team_geometry()

    validate_team_geometry(features)

    write_report(features)

    print("\nETAPA 06 COMPLETADA.")


if __name__ == "__main__":
    main()
