
import numpy as np
import pandas as pd

from config import GAME, ROOT_DIR, N_VIDEO_FRAMES
from starter.load_atd import load


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

MOVING_THRESHOLD = 0.5
SUSPICIOUS_THRESHOLD = 12.0

TEAMS = {
    "Barcelona": range(1, 25),
    "Atletic_Club": range(25, 47),
}


# --------------------------------------------------
# ESTADÍSTICAS POR FILA
# --------------------------------------------------

def row_stat(values, valid, operation):

    result = np.full(values.shape[0], np.nan)

    for i in range(values.shape[0]):

        row = values[i, valid[i]]

        if row.size == 0:
            continue

        if operation == "mean":
            result[i] = np.mean(row)

        elif operation == "median":
            result[i] = np.median(row)

        elif operation == "max":
            result[i] = np.max(row)

    return result


# --------------------------------------------------
# CONSTRUCCIÓN
# --------------------------------------------------

def build_team_speed_coverage():

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
            f"Se esperaban {N_VIDEO_FRAMES} filas; hay {len(df)}."
        )

    if not np.array_equal(df["frame"].to_numpy(), expected_frames):
        raise ValueError("El tracking no está alineado por frame.")

    print("Calculando velocidad y cobertura por equipo...")

    features = pd.DataFrame({
        "frame": expected_frames
    })

    for team_name, slots in TEAMS.items():

        slot_list = list(slots)
        expected_count = len(slot_list)

        x_cols = [f"{slot}_x" for slot in slot_list]
        y_cols = [f"{slot}_y" for slot in slot_list]
        speed_cols = [f"{slot}_s" for slot in slot_list]

        required = x_cols + y_cols + speed_cols
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Faltan columnas: {missing}")

        x = df[x_cols].to_numpy(dtype=float)
        y = df[y_cols].to_numpy(dtype=float)
        speed = df[speed_cols].to_numpy(dtype=float)

        # Posición válida: X e Y finitas.
        position_valid = np.isfinite(x) & np.isfinite(y)

        # Velocidad válida: velocidad finita y posición válida.
        speed_valid = np.isfinite(speed) & position_valid

        valid_position_count = position_valid.sum(axis=1)
        valid_speed_count = speed_valid.sum(axis=1)

        moving = speed_valid & (speed > MOVING_THRESHOLD)
        suspicious = speed_valid & (speed > SUSPICIOUS_THRESHOLD)

        moving_count = moving.sum(axis=1)
        suspicious_count = suspicious.sum(axis=1)

        speed_mean = row_stat(speed, speed_valid, "mean")
        speed_median = row_stat(speed, speed_valid, "median")
        speed_max = row_stat(speed, speed_valid, "max")

        # Proporción de velocidades no sospechosas.
        speed_quality_ratio = np.full(N_VIDEO_FRAMES, np.nan)
        np.divide(
            valid_speed_count - suspicious_count,
            valid_speed_count,
            out=speed_quality_ratio,
            where=valid_speed_count > 0
        )

        position_coverage = valid_position_count / expected_count
        speed_coverage = valid_speed_count / expected_count

        moving_ratio = np.full(N_VIDEO_FRAMES, np.nan)
        np.divide(
            moving_count,
            valid_speed_count,
            out=moving_ratio,
            where=valid_speed_count > 0
        )

        prefix = team_name

        team_features = {
            "valid_position_count": valid_position_count,
            "valid_speed_count": valid_speed_count,
            "moving_count": moving_count,
            "speed_mean": speed_mean,
            "speed_median": speed_median,
            "speed_max": speed_max,
            "suspicious_speed_count": suspicious_count,
            "speed_quality_ratio": speed_quality_ratio,
            "expected_player_count": np.full(
                N_VIDEO_FRAMES, expected_count, dtype=np.int8
            ),
            "position_coverage": position_coverage,
            "speed_coverage": speed_coverage,
            "moving_ratio": moving_ratio,
        }

        for name, values in team_features.items():
            features[f"{prefix}_{name}"] = values

    return features


# --------------------------------------------------
# VALIDACIÓN
# --------------------------------------------------

def validate_team_speed_coverage(features):

    print("\nValidando estructura...")

    assert len(features) == N_VIDEO_FRAMES
    assert features.shape[1] == 25

    assert features["frame"].is_unique
    assert np.array_equal(
        features["frame"].to_numpy(),
        np.arange(N_VIDEO_FRAMES)
    )

    for team, expected in [
        ("Barcelona", 24),
        ("Atletic_Club", 22),
    ]:

        position_count = features[f"{team}_valid_position_count"]
        speed_count = features[f"{team}_valid_speed_count"]
        moving_count = features[f"{team}_moving_count"]
        suspicious_count = features[f"{team}_suspicious_speed_count"]

        assert (position_count >= 0).all()
        assert (position_count <= expected).all()

        assert (speed_count >= 0).all()
        assert (speed_count <= position_count).all()

        assert (moving_count >= 0).all()
        assert (moving_count <= speed_count).all()

        assert (suspicious_count >= 0).all()
        assert (suspicious_count <= speed_count).all()

        for variable in ["position_coverage", "speed_coverage"]:
            values = features[f"{team}_{variable}"]
            assert values.between(0, 1).all()

        assert (
            features[f"{team}_expected_player_count"] == expected
        ).all()

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
            f"\n{team}:"
        )
        print(
            "Media de posiciones válidas:",
            round(features[f"{team}_valid_position_count"].mean(), 3)
        )
        print(
            "Media de velocidades válidas:",
            round(features[f"{team}_valid_speed_count"].mean(), 3)
        )
        print(
            "Media de slots en movimiento:",
            round(features[f"{team}_moving_count"].mean(), 3)
        )
        print(
            "Cobertura media de velocidad:",
            round(features[f"{team}_speed_coverage"].mean(), 4)
        )
        print(
            "Máxima velocidad:",
            round(features[f"{team}_speed_max"].max(), 3)
        )

    print("\nConsistencia de conteos: True")
    print("Coberturas dentro de rango: True")
    print("Alineación correcta: True")

    return True


# --------------------------------------------------
# DOCUMENTACIÓN
# --------------------------------------------------

def write_report(features):

    report_path = ROOT_DIR / "ETAPA_07_VELOCIDAD_COBERTURA_EQUIPOS.md"

    report = f"""# Etapa 07 — Velocidad y cobertura por equipo

## Objetivo

Construir variables de velocidad y calidad de tracking
para Barcelona y Athletic Club por frame.

## Configuración

- Partido: {GAME}
- Frames: {N_VIDEO_FRAMES}
- Barcelona: slots 1–24 (24 esperados)
- Athletic Club: slots 25–46 (22 esperados)
- Umbral de movimiento: > {MOVING_THRESHOLD} m/s
- Umbral de velocidad sospechosa: > {SUSPICIOUS_THRESHOLD} m/s

## Variables

12 variables por equipo:
- valid_position_count
- valid_speed_count
- moving_count
- speed_mean
- speed_median
- speed_max
- suspicious_speed_count
- speed_quality_ratio
- expected_player_count
- position_coverage
- speed_coverage
- moving_ratio

Total: 24 variables, más `frame`.

## Criterios

- La posición válida requiere X e Y finitas.
- La velocidad válida requiere velocidad finita y posición válida.
- No se imputan valores ausentes.
- Las coberturas usan el número de slots esperados.
- Los slots no representan identidades estables de jugadores.
- Los umbrales son criterios operativos iniciales.

## Validación estructural

- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Alineación: correcta
- Consistencia de conteos: correcta
- Coberturas dentro de rango: correcta

## Resultado

Etapa 07 completada.
"""

    report_path.write_text(report, encoding="utf-8")
    print("\nDocumento generado:", report_path.name)


# --------------------------------------------------
# EJECUCIÓN
# --------------------------------------------------

def main():

    features = build_team_speed_coverage()

    validate_team_speed_coverage(features)

    write_report(features)

    print("\nETAPA 07 COMPLETADA.")


if __name__ == "__main__":
    main()
