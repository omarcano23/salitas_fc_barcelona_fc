
from pathlib import Path

import numpy as np
import pandas as pd

from s01_tracking import load_tracking
from config import ROOT_DIR, N_VIDEO_FRAMES, FPS


# ============================================================
# CONFIGURACIÓN
# ============================================================

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

MAX_PLAUSIBLE_SPEED_MPS = 40.0
EXTREME_SPEED_MPS = 50.0

EXPECTED_FEATURE_COLUMNS = 18


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_features(
    features,
    ball_valid,
    pitch_length_m=PITCH_LENGTH_M,
    pitch_width_m=PITCH_WIDTH_M,
    fps=FPS,
):
    expected_columns = [
        "frame",
        "ball_position_available",
        "ball_x_normalized",
        "ball_y_normalized",
        "ball_x_m",
        "ball_y_m",
        "ball_delta_frame",
        "ball_delta_time_s",
        "ball_delta_x_m",
        "ball_delta_y_m",
        "ball_vx_mps",
        "ball_vy_mps",
        "ball_speed_mps",
        "ball_displacement_m",
        "ball_velocity_available",
        "ball_outside_normalized",
        "ball_speed_suspicious_40",
        "ball_speed_extreme_50",
    ]

    assert len(features) == N_VIDEO_FRAMES
    assert len(features.columns) == EXPECTED_FEATURE_COLUMNS
    assert features.columns.tolist() == expected_columns

    # Alineación temporal.
    expected_frames = np.arange(N_VIDEO_FRAMES)

    assert np.array_equal(
        features["frame"].to_numpy(),
        expected_frames,
    )

    assert features["frame"].is_unique

    assert len(ball_valid) == N_VIDEO_FRAMES

    # Disponibilidad del balón.
    assert (
        features["ball_position_available"].to_numpy()
        == ball_valid.astype(int)
    ).all()

    unavailable = ~ball_valid

    unavailable_columns = [
        "ball_x_normalized",
        "ball_y_normalized",
        "ball_x_m",
        "ball_y_m",
        "ball_delta_x_m",
        "ball_delta_y_m",
        "ball_vx_mps",
        "ball_vy_mps",
        "ball_speed_mps",
        "ball_displacement_m",
    ]

    for col in unavailable_columns:
        assert features.loc[unavailable, col].isna().all(), (
            f"{col}: hay valores en frames sin balón válido."
        )

    # Coordenadas normalizadas.
    for col in [
        "ball_x_normalized",
        "ball_y_normalized",
    ]:
        values = features.loc[ball_valid, col]

        assert values.notna().all()
        assert values.between(0, 1).all()

    # Coordenadas métricas.
    assert (
        features.loc[ball_valid, "ball_x_m"].between(
            0, pitch_length_m
        )
    ).all()

    assert (
        features.loc[ball_valid, "ball_y_m"].between(
            0, pitch_width_m
        )
    ).all()

    # Las posiciones válidas no pueden estar fuera del campo.
    assert (
        features.loc[ball_valid, "ball_outside_normalized"] == 0
    ).all()

    # Disponibilidad de velocidad.
    expected_velocity_available = (
        features["ball_speed_mps"].notna()
    ).astype(int)

    assert (
        features["ball_velocity_available"]
        == expected_velocity_available
    ).all()

    # Velocidades no negativas.
    speed = features["ball_speed_mps"].to_numpy(dtype=float)
    finite_speed = np.isfinite(speed)

    assert (speed[finite_speed] >= 0).all()

    # Indicadores de velocidad.
    expected_suspicious = (
        features["ball_speed_mps"] > MAX_PLAUSIBLE_SPEED_MPS
    ).fillna(False).astype(int)

    expected_extreme = (
        features["ball_speed_mps"] > EXTREME_SPEED_MPS
    ).fillna(False).astype(int)

    assert (
        features["ball_speed_suspicious_40"]
        == expected_suspicious
    ).all()

    assert (
        features["ball_speed_extreme_50"]
        == expected_extreme
    ).all()

    assert (
        features["ball_speed_extreme_50"]
        <= features["ball_speed_suspicious_40"]
    ).all()

    # Velocidad únicamente entre frames consecutivos.
    velocity_available = (
        features["ball_velocity_available"] == 1
    )

    assert (
        features.loc[
            velocity_available, "ball_delta_frame"
        ] == 1
    ).all()

    assert np.allclose(
        features.loc[
            velocity_available, "ball_delta_time_s"
        ].to_numpy(dtype=float),
        1.0 / fps,
    )

    # Comprobar que no existen infinitos.
    numeric_values = features.select_dtypes(
        include=[np.number]
    ).to_numpy(dtype=float)

    assert not np.isinf(numeric_values).any()

    return True


# ============================================================
# CONSTRUCCIÓN DE CARACTERÍSTICAS
# ============================================================

def build_ball_kinematics(
    tracking,
    pitch_length_m=PITCH_LENGTH_M,
    pitch_width_m=PITCH_WIDTH_M,
    fps=FPS,
):
    """
    Construye las 17 características de posición y cinemática
    del balón.

    Recibe el DataFrame de tracking ya preparado por S01.

    Returns:
        features: DataFrame con frame + 17 características.
        ball_valid: máscara booleana de disponibilidad del balón.
    """

    print("Calculando posición y cinemática del balón...")

    # --------------------------------------------------------
    # 1. Validación del tracking recibido
    # --------------------------------------------------------

    required_columns = [
        "frame",
        "ball_x",
        "ball_y",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in tracking.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en tracking: {missing_columns}"
        )

    if len(tracking) != N_VIDEO_FRAMES:
        raise ValueError(
            f"Tracking con {len(tracking)} filas; "
            f"se esperaban {N_VIDEO_FRAMES}."
        )

    expected_frames = np.arange(N_VIDEO_FRAMES)

    if not np.array_equal(
        tracking["frame"].to_numpy(),
        expected_frames,
    ):
        raise ValueError(
            "El tracking no está alineado de 0 a "
            f"{N_VIDEO_FRAMES - 1}."
        )

    if pitch_length_m <= 0 or pitch_width_m <= 0 or fps <= 0:
        raise ValueError(
            "Las dimensiones del campo y FPS deben ser positivos."
        )

    # --------------------------------------------------------
    # 2. Coordenadas originales
    # --------------------------------------------------------

    x_raw = tracking["ball_x"].to_numpy(dtype=float)
    y_raw = tracking["ball_y"].to_numpy(dtype=float)

    # Una posición es válida si ambas coordenadas son finitas
    # y están dentro del campo normalizado.
    ball_valid = (
        np.isfinite(x_raw)
        & np.isfinite(y_raw)
        & (x_raw >= 0)
        & (x_raw <= 1)
        & (y_raw >= 0)
        & (y_raw <= 1)
    )

    # --------------------------------------------------------
    # 3. Conversión a metros
    # --------------------------------------------------------

    x_m = x_raw * pitch_length_m
    y_m = y_raw * pitch_width_m

    # Las posiciones inválidas no participan en la cinemática.
    x_m[~ball_valid] = np.nan
    y_m[~ball_valid] = np.nan

    # --------------------------------------------------------
    # 4. Arrays de cinemática
    # --------------------------------------------------------

    n = len(tracking)

    delta_frame = np.full(n, np.nan)
    delta_time = np.full(n, np.nan)
    delta_x = np.full(n, np.nan)
    delta_y = np.full(n, np.nan)

    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)

    speed = np.full(n, np.nan)
    displacement = np.full(n, np.nan)

    # Solo se calculan pares de frames consecutivos válidos.
    consecutive_valid = (
        ball_valid[1:] & ball_valid[:-1]
    )

    current_indices = (
        np.where(consecutive_valid)[0] + 1
    )

    previous_indices = current_indices - 1

    dx = (
        x_m[current_indices]
        - x_m[previous_indices]
    )

    dy = (
        y_m[current_indices]
        - y_m[previous_indices]
    )

    dt = 1.0 / fps

    delta_frame[current_indices] = 1
    delta_time[current_indices] = dt

    delta_x[current_indices] = dx
    delta_y[current_indices] = dy

    vx[current_indices] = dx / dt
    vy[current_indices] = dy / dt

    displacement[current_indices] = np.sqrt(
        dx ** 2 + dy ** 2
    )

    speed[current_indices] = (
        displacement[current_indices] / dt
    )

    # --------------------------------------------------------
    # 5. Coordenadas fuera del rango normalizado
    # --------------------------------------------------------

    outside_normalized = (
        (
            np.isfinite(x_raw)
            & ((x_raw < 0) | (x_raw > 1))
        )
        |
        (
            np.isfinite(y_raw)
            & ((y_raw < 0) | (y_raw > 1))
        )
    ).astype(int)

    # --------------------------------------------------------
    # 6. Construcción del DataFrame
    # --------------------------------------------------------

    print("Construyendo matriz de características...")

    features = pd.DataFrame({

        "frame": tracking["frame"].to_numpy(),

        # Disponibilidad.
        "ball_position_available": ball_valid.astype(int),

        # Coordenadas normalizadas originales.
        "ball_x_normalized": np.where(
            ball_valid, x_raw, np.nan
        ),

        "ball_y_normalized": np.where(
            ball_valid, y_raw, np.nan
        ),

        # Coordenadas métricas.
        "ball_x_m": x_m,
        "ball_y_m": y_m,

        # Desplazamiento y tiempo.
        "ball_delta_frame": delta_frame,
        "ball_delta_time_s": delta_time,

        "ball_delta_x_m": delta_x,
        "ball_delta_y_m": delta_y,

        # Velocidad vectorial.
        "ball_vx_mps": vx,
        "ball_vy_mps": vy,

        # Velocidad y desplazamiento.
        "ball_speed_mps": speed,
        "ball_displacement_m": displacement,

        # Disponibilidad de velocidad.
        "ball_velocity_available": (
            np.isfinite(speed).astype(int)
        ),

        # Calidad de posición.
        "ball_outside_normalized": outside_normalized,

        # Indicadores de anomalías.
        "ball_speed_suspicious_40": (
            speed > MAX_PLAUSIBLE_SPEED_MPS
        ).astype(int),

        "ball_speed_extreme_50": (
            speed > EXTREME_SPEED_MPS
        ).astype(int),
    })

    # --------------------------------------------------------
    # 7. Validación
    # --------------------------------------------------------

    validate_features(
        features,
        ball_valid,
        pitch_length_m=pitch_length_m,
        pitch_width_m=pitch_width_m,
        fps=fps,
    )

    return features, ball_valid


# ============================================================
# DOCUMENTACIÓN
# ============================================================

def write_report(
    features,
    ball_valid,
    output_path,
    pitch_length_m=PITCH_LENGTH_M,
    pitch_width_m=PITCH_WIDTH_M,
    fps=FPS,
):

    available_speeds = features["ball_speed_mps"].dropna()

    report = f"""# ETAPA 12 — Posición y cinemática del balón

## Resultado

- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Variables calculadas: 17
- Frames: {features['frame'].min()} a {features['frame'].max()}
- Alineación: correcta

## Disponibilidad

- Frames con balón válido: {int(ball_valid.sum())}
- Proporción de disponibilidad: {ball_valid.mean():.4%}
- Frames con velocidad derivada: {len(available_speeds)}
- Proporción con velocidad derivada: {features['ball_velocity_available'].mean():.4%}

## Variables generadas

### Posición

- ball_position_available
- ball_x_normalized
- ball_y_normalized
- ball_x_m
- ball_y_m

### Desplazamiento y tiempo

- ball_delta_frame
- ball_delta_time_s
- ball_delta_x_m
- ball_delta_y_m
- ball_displacement_m

### Velocidad

- ball_vx_mps
- ball_vy_mps
- ball_speed_mps
- ball_velocity_available

### Calidad y anomalías

- ball_outside_normalized
- ball_speed_suspicious_40
- ball_speed_extreme_50

## Resumen de velocidad

- Media: {available_speeds.mean():.3f} m/s
- Mediana: {available_speeds.median():.3f} m/s
- Percentil 99: {available_speeds.quantile(0.99):.3f} m/s
- Máxima: {available_speeds.max():.3f} m/s
- Velocidades superiores a 40 m/s: {int(features['ball_speed_suspicious_40'].sum())}
- Velocidades superiores a 50 m/s: {int(features['ball_speed_extreme_50'].sum())}

## Criterios de cálculo

- Coordenadas originales normalizadas entre 0 y 1.
- Conversión a metros con dimensiones de campo de {pitch_length_m} × {pitch_width_m}.
- Frecuencia de vídeo: {fps} FPS.
- Se considera válido el balón cuando ambas coordenadas son finitas y están entre 0 y 1.
- Las coordenadas normalizadas originales se conservan.
- Las coordenadas métricas se calculan únicamente para posiciones válidas.
- Velocidad calculada solo entre frames consecutivos observados.
- No se interpolan huecos ni se conectan observaciones separadas.
- Velocidades superiores a 40 m/s se marcan como sospechosas.
- Velocidades superiores a 50 m/s se marcan como extremas.
- Los indicadores no eliminan ni modifican las observaciones originales.

## Validaciones

- Número de filas y columnas: correcto.
- Nombres y orden de columnas: correctos.
- Alineación y unicidad de frames: correctas.
- Disponibilidad coherente con coordenadas: correcta.
- Coordenadas normalizadas conservadas: correcto.
- Coordenadas métricas dentro del campo: correcto.
- Ausencia de interpolación: correcta.
- Velocidad solo entre frames consecutivos: correcta.
- Indicadores de velocidad: correctos.

## Consideraciones

La velocidad derivada es una estimación geométrica basada en posiciones consecutivas del tracking. No representa una medición física corregida.

Los valores extremos se conservan para que puedan ser evaluados posteriormente por los indicadores de calidad y el motor de decisión.
"""

    output_path.write_text(
        report,
        encoding="utf-8",
    )


# ============================================================
# EJECUCIÓN INDEPENDIENTE
# ============================================================

def main():

    # load_tracking ya muestra su propio mensaje de carga.
    tracking, dataset = load_tracking()

    print("Calculando posición y cinemática del balón...")

    features, ball_valid = build_ball_kinematics(
        tracking,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        fps=FPS,
    )

    print("\nValidación estructural: correcta")

    print(f"\nFilas: {len(features)}")
    print(f"Columnas: {len(features.columns)}")

    print(
        f"Frames: {features['frame'].min()} "
        f"a {features['frame'].max()}"
    )

    print("\nBalón:")

    print("Frames válidos:", int(ball_valid.sum()))

    print(
        "Frames con velocidad:",
        int(features["ball_velocity_available"].sum())
    )

    print(
        "Velocidad media:",
        round(features["ball_speed_mps"].mean(), 3)
    )

    print(
        "Velocidad mediana:",
        round(features["ball_speed_mps"].median(), 3)
    )

    print(
        "Velocidad máxima:",
        round(features["ball_speed_mps"].max(), 3)
    )

    print(
        "Velocidades >40 m/s:",
        int(features["ball_speed_suspicious_40"].sum())
    )

    print(
        "Velocidades >50 m/s:",
        int(features["ball_speed_extreme_50"].sum())
    )

    print(
        "Posiciones fuera de rango:",
        int(features["ball_outside_normalized"].sum())
    )

    print("\nAlineación correcta: True")
    print("Cinemática sin interpolación: True")
    print("Indicadores consistentes: True")

    report_path = ROOT_DIR / "ETAPA_12_CINEMATICA_BALON.md"

    write_report(
        features,
        ball_valid,
        report_path,
        pitch_length_m=PITCH_LENGTH_M,
        pitch_width_m=PITCH_WIDTH_M,
        fps=FPS,
    )

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 12 COMPLETADA.")

    return features


if __name__ == "__main__":
    main()
