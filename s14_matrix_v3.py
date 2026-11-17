
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT_DIR, OUTPUT_DIR, N_VIDEO_FRAMES, FPS

import s01_tracking as s01
import s02_homography as s02
import s03_global_geometry as s03
import s04_global_movement as s04
import s05_smart_tagging as s05
import s06_team_geometry as s06
import s07_team_speed_coverage as s07
import s08_team_differences as s08
import s09_team_zone_occupancy as s09
import s10_ball_proximity as s10
import s11_ball_approach_dynamics as s11
import s12_ball_kinematics as s12
import s13_global_player_ball_proximity as s13


# ============================================================
# CONFIGURACIÓN
# ============================================================

PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0

EXPECTED_ROWS = N_VIDEO_FRAMES
EXPECTED_COLUMNS = 251

MATRIX_PATH = OUTPUT_DIR / "matrix_v3.csv"
REPORT_PATH = ROOT_DIR / "ETAPA_14_MATRIZ_V3.md"


# ============================================================
# VALIDACIÓN COMÚN DE BLOQUES
# ============================================================

def validate_block(name, features, expected_columns):

    if not isinstance(features, pd.DataFrame):
        raise TypeError(
            f"{name}: el resultado no es un DataFrame."
        )

    if features.columns.duplicated().any():
        raise ValueError(
            f"{name}: contiene nombres de columnas duplicados."
        )

    if len(features) != EXPECTED_ROWS:
        raise ValueError(
            f"{name}: {len(features)} filas; "
            f"se esperaban {EXPECTED_ROWS}."
        )

    if "frame" not in features.columns:
        raise ValueError(
            f"{name}: falta la columna frame."
        )

    expected_frames = np.arange(EXPECTED_ROWS)

    if not np.array_equal(
        features["frame"].to_numpy(),
        expected_frames,
    ):
        raise ValueError(
            f"{name}: frames desalineados."
        )

    if len(features.columns) != expected_columns:
        raise ValueError(
            f"{name}: {len(features.columns)} columnas; "
            f"se esperaban {expected_columns}."
        )

    print(
        f"  {name}: OK "
        f"({len(features.columns) - 1} características)"
    )


# ============================================================
# FAMILIA A — IDENTIFICACIÓN Y TIEMPO
# ============================================================

def build_identification(tracking):

    required = [
        "frame",
        "frame_id",
        "period_id",
    ]

    missing = [
        col for col in required
        if col not in tracking.columns
    ]

    if missing:
        raise ValueError(
            f"Faltan columnas para la familia A: {missing}"
        )

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy(),
        "frame_id": tracking["frame_id"].to_numpy(),
        "period_id": tracking["period_id"].to_numpy(),
        "timestamp_seconds": (
            tracking["frame"].to_numpy(dtype=float) / FPS
        ),
    })

    return features


# ============================================================
# FAMILIA H — DIFERENCIAS ENTRE EQUIPOS
# ============================================================

def build_team_differences(tracking):

    barcelona = s08.calculate_team_metrics(
        tracking,
        s08.TEAMS["Barcelona"],
    )

    atletic = s08.calculate_team_metrics(
        tracking,
        s08.TEAMS["Atletic_Club"],
    )

    dx = (
        barcelona["centroid_x"]
        - atletic["centroid_x"]
    )

    dy = (
        barcelona["centroid_y"]
        - atletic["centroid_y"]
    )

    features = pd.DataFrame({

        "frame": tracking["frame"].to_numpy(),

        "team_centroid_distance_m": np.sqrt(
            dx ** 2 + dy ** 2
        ),

        "centroid_x_difference_m": dx,

        "centroid_y_difference_m": dy,

        "team_inside_count_difference": (
            barcelona["inside_count"]
            - atletic["inside_count"]
        ),

        "team_valid_position_count_difference": (
            barcelona["valid_position_count"]
            - atletic["valid_position_count"]
        ),

        "team_speed_mean_difference_mps": (
            barcelona["speed_mean"]
            - atletic["speed_mean"]
        ),
    })

    s08.validate_features(features)

    return features


# ============================================================
# CONSTRUCCIÓN DE LAS 13 FAMILIAS
# ============================================================

def build_matrix():

    print("\n" + "=" * 65)
    print("C14 — CONSTRUCCIÓN DE MATRIZ V3")
    print("=" * 65)

    blocks = {}

    # --------------------------------------------------------
    # S01 — Tracking
    # --------------------------------------------------------

    print("\n[S01] Cargando tracking...")

    tracking, dataset = s01.load_tracking()

    tracking_validation = s01.validate_tracking(tracking)

    if not tracking_validation["alignment_ok"]:
        raise ValueError(
            "S01: el tracking no supera la validación."
        )

    pitch_length_m = float(
        dataset.metadata.pitch_dimensions.pitch_length
    )

    pitch_width_m = float(
        dataset.metadata.pitch_dimensions.pitch_width
    )

    print(
        f"Dimensiones del campo: "
        f"{pitch_length_m} × {pitch_width_m} m"
    )

    # --------------------------------------------------------
    # A — Identificación y tiempo
    # --------------------------------------------------------

    print("\n[A] Identificación y tiempo")

    blocks["A"] = build_identification(tracking)

    validate_block("A", blocks["A"], 4)

    # --------------------------------------------------------
    # B — Homografía
    # --------------------------------------------------------

    print("\n[B] Homografía")

    homography, matrices = s02.load_homography_features()

    homography_validation = s02.validate_homography(
        homography,
        matrices,
    )

    if not homography_validation["alignment_ok"]:
        raise ValueError(
            "S02: la homografía no supera la validación."
        )

    blocks["B"] = homography

    validate_block("B", blocks["B"], 3)

    del matrices

    # --------------------------------------------------------
    # C — Geometría global
    # --------------------------------------------------------

    print("\n[C] Geometría global")

    blocks["C"] = s03.build_global_geometry(
        tracking,
        pitch_length_m,
        pitch_width_m,
    )

    s03.validate_geometry(blocks["C"])

    validate_block("C", blocks["C"], 24)

    # --------------------------------------------------------
    # D — Movimiento global
    # --------------------------------------------------------

    print("\n[D] Movimiento global")

    movement, movement_pitch_length, movement_pitch_width = (
        s04.build_global_movement()
    )

    s04.validate(movement)

    if not np.isclose(
        movement_pitch_length,
        pitch_length_m,
    ) or not np.isclose(
        movement_pitch_width,
        pitch_width_m,
    ):
        raise ValueError(
            "Las dimensiones de campo de S04 y S01 no coinciden."
        )

    blocks["D"] = movement

    validate_block("D", blocks["D"], 11)

    # --------------------------------------------------------
    # E — Smart Tagging
    # --------------------------------------------------------

    print("\n[E] Smart Tagging")

    smart_tagging, events = s05.build_smart_tagging()

    s05.validate_smart_tagging(
        smart_tagging,
        events,
    )

    blocks["E"] = smart_tagging

    validate_block("E", blocks["E"], 61)

    # --------------------------------------------------------
    # F — Geometría por equipo
    # --------------------------------------------------------

    print("\n[F] Geometría por equipo")

    team_geometry = s06.build_team_geometry()

    s06.validate_team_geometry(team_geometry)

    blocks["F"] = team_geometry

    validate_block("F", blocks["F"], 23)

    # --------------------------------------------------------
    # G — Velocidad y cobertura por equipo
    # --------------------------------------------------------

    print("\n[G] Velocidad y cobertura por equipo")

    team_speed = s07.build_team_speed_coverage()

    s07.validate_team_speed_coverage(team_speed)

    blocks["G"] = team_speed
    blocks["G"] = blocks["G"].rename(columns={
        "Barcelona_valid_position_count":
            "Barcelona_speed_valid_position_count",
        "Atletic_Club_valid_position_count":
            "Atletic_Club_speed_valid_position_count",
    })

    validate_block("G", blocks["G"], 25)

    # --------------------------------------------------------
    # H — Diferencias entre equipos
    # --------------------------------------------------------

    print("\n[H] Diferencias entre equipos")

    blocks["H"] = build_team_differences(tracking)

    validate_block("H", blocks["H"], 7)

    # --------------------------------------------------------
    # I — Ocupación de zonas
    # --------------------------------------------------------

    print("\n[I] Ocupación de zonas")

    blocks["I"] = s09.build_team_zone_occupancy(
        tracking,
        pitch_length_m,
        pitch_width_m
    )

    # Renombrar columnas para evitar duplicados con F.
    blocks["I"] = blocks["I"].rename(columns={
        "Barcelona_valid_position_count":
            "Barcelona_zone_valid_position_count",
        "Atletic_Club_valid_position_count":
            "Atletic_Club_zone_valid_position_count",
        "Barcelona_inside_position_count":
            "Barcelona_zone_inside_position_count",
        "Atletic_Club_inside_position_count":
            "Atletic_Club_zone_inside_position_count",
    })

    print("  I: OK (49 características)")
    validate_block("I", blocks["I"], 50)

    # --------------------------------------------------------
    # J — Proximidad del balón por equipo
    # --------------------------------------------------------

    print("\n[J] Proximidad del balón por equipo")

    blocks["J"] = s10.build_ball_proximity(
        tracking,
        pitch_length_m,
        pitch_width_m,
    )

    blocks["J"] = blocks["J"].rename(columns={
        "Barcelona_valid_position_count":
            "Barcelona_proximity_valid_position_count",
        "Atletic_Club_valid_position_count":
            "Atletic_Club_proximity_valid_position_count",
    })

    validate_block("J", blocks["J"], 17)

    # --------------------------------------------------------
    # K — Dinámica de aproximación
    # --------------------------------------------------------

    print("\n[K] Dinámica de aproximación")

    blocks["K"] = s11.build_team_approach(
        tracking,
        pitch_length_m,
        pitch_width_m,
        FPS,
    )

    validate_block("K", blocks["K"], 13)

    # --------------------------------------------------------
    # L — Posición y cinemática del balón
    # --------------------------------------------------------

    print("\n[L] Posición y cinemática del balón")

    ball_kinematics, ball_valid = (
        s12.build_ball_kinematics(
            tracking,
            pitch_length_m,
            pitch_width_m,
            FPS,
        )
    )

    blocks["L"] = ball_kinematics

    validate_block("L", blocks["L"], 18)

    # --------------------------------------------------------
    # M — Proximidad global jugador-balón
    # --------------------------------------------------------

    print("\n[M] Proximidad global jugador-balón")

    blocks["M"] = (
        s13.build_global_player_ball_proximity(
            tracking,
            pitch_length_m,
            pitch_width_m,
        )
    )

    validate_block("M", blocks["M"], 7)

    # ========================================================
    # INTEGRACIÓN
    # ========================================================

    print("\nIntegrando las 13 familias...")

    # Detectar nombres repetidos entre familias.
    all_feature_names = []

    for block in blocks.values():
        all_feature_names.extend([
            col for col in block.columns
            if col != "frame"
        ])

    print("\n[DIAGNÓSTICO] Origen de columnas duplicadas")

    for column in [
        "Barcelona_valid_position_count",
        "Atletic_Club_valid_position_count",
    ]:
        owners = [
            block_name
            for block_name, block in blocks.items()
            if column in block.columns
        ]

        print(f"{column}: presente en bloques {owners}")
    
    duplicated_names = pd.Index(
        all_feature_names
    )[pd.Index(all_feature_names).duplicated()].unique()

    if len(duplicated_names):
        raise ValueError(
            "Nombres de características duplicados: "
            f"{duplicated_names.tolist()}"
        )

    # Unir por frame, no por posición implícita.
    indexed_blocks = [
        block.set_index("frame")
        for block in blocks.values()
    ]

    matrix = pd.concat(
        indexed_blocks,
        axis=1,
        join="outer",
        verify_integrity=True,
    )

    matrix = matrix.reset_index()

    # ========================================================
    # ORDEN FINAL
    # ========================================================

    ordered_columns = []

    # Familia A completa: incluye frame.
    ordered_columns.extend(blocks["A"].columns.tolist())

    # Familias B a J.
    for family in ["B", "C", "D", "E", "F", "G", "H", "I", "J"]:
        ordered_columns.extend([
            col for col in blocks[family].columns
            if col != "frame"
        ])

    # Según el esquema original, L comienza con esta variable.
    ordered_columns.append("ball_position_available")

    # K.
    ordered_columns.extend([
        col for col in blocks["K"].columns
        if col != "frame"
    ])

    # Resto de L.
    ordered_columns.extend([
        col for col in blocks["L"].columns
        if col not in ("frame", "ball_position_available")
    ])

    # M.
    ordered_columns.extend([
        col for col in blocks["M"].columns
        if col != "frame"
    ])

    if len(ordered_columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"El esquema produce {len(ordered_columns)} columnas; "
            f"se esperaban {EXPECTED_COLUMNS}."
        )

    if len(set(ordered_columns)) != EXPECTED_COLUMNS:
        raise ValueError(
            "El orden final contiene nombres duplicados."
        )

    missing_columns = [
        col for col in ordered_columns
        if col not in matrix.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en la integración: {missing_columns}"
        )

    matrix = matrix[ordered_columns]

    # ========================================================
    # VALIDACIÓN FINAL
    # ========================================================

    print("\nValidando matriz completa...")

    if matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_COLUMNS,
    ):
        raise ValueError(
            f"Dimensión incorrecta: {matrix.shape}; "
            f"esperada: {(EXPECTED_ROWS, EXPECTED_COLUMNS)}."
        )

    expected_frames = np.arange(EXPECTED_ROWS)

    if not np.array_equal(
        matrix["frame"].to_numpy(),
        expected_frames,
    ):
        raise ValueError(
            "La matriz final tiene frames ausentes o desordenados."
        )

    if matrix.columns.duplicated().any():
        raise ValueError(
            "La matriz final tiene columnas duplicadas."
        )

    numeric = matrix.select_dtypes(
        include=[np.number]
    )

    infinite_count = int(
        np.isinf(numeric.to_numpy(dtype=float)).sum()
    )

    if infinite_count:
        raise ValueError(
            f"Se encontraron {infinite_count} valores infinitos."
        )

    missing_values = int(
        matrix.isna().sum().sum()
    )

    print("\nVALIDACIÓN FINAL")
    print("-" * 45)
    print("Filas:", len(matrix))
    print("Columnas:", len(matrix.columns))
    print("Primer frame:", int(matrix["frame"].iloc[0]))
    print("Último frame:", int(matrix["frame"].iloc[-1]))
    print("Frames únicos:", matrix["frame"].is_unique)
    print("Columnas únicas:", matrix.columns.is_unique)
    print("Valores infinitos:", infinite_count)
    print("Valores NaN:", missing_values)

    # ========================================================
    # EXPORTACIÓN
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nGuardando matriz...")

    matrix.to_csv(
        MATRIX_PATH,
        index=False,
    )

    # ========================================================
    # INFORME
    # ========================================================

    family_report = []

    for family, block in blocks.items():
        family_report.append(
            f"| {family} | {len(block.columns) - 1} |"
        )

    report = f"""# ETAPA 14 — MATRIZ V3

## Resultado

- Filas: {len(matrix)}
- Columnas: {len(matrix.columns)}
- Frames: 0 a {EXPECTED_ROWS - 1}
- Familias: 13
- Características: 250
- Columna de identificación: frame

## Distribución de columnas por familia

| Familia | Características |
|---|---:|
{chr(10).join(family_report)}

## Validaciones

- Dimensión esperada: {EXPECTED_ROWS} × {EXPECTED_COLUMNS}
- Dimensión obtenida: {matrix.shape[0]} × {matrix.shape[1]}
- Frames completos y ordenados: correcto.
- Frames duplicados: ninguno.
- Columnas duplicadas: ninguna.
- Valores infinitos: {infinite_count}
- Valores NaN: {missing_values}

## Criterios

- Integración mediante la columna `frame`.
- No se eliminan filas con valores ausentes.
- No se interpolan posiciones.
- Se conserva el orden definido para V3.
- Se conserva la distinción entre balón observado y no observado.
- Los valores NaN se mantienen para su tratamiento posterior.

## Archivo generado

`{MATRIX_PATH}`

## Estado

ETAPA 14 COMPLETADA.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("ETAPA 14 COMPLETADA")
    print("=" * 65)

    print(f"Matriz: {MATRIX_PATH}")
    print(f"Informe: {REPORT_PATH}")

    return matrix


# ============================================================
# EJECUCIÓN
# ============================================================

def main():
    return build_matrix()


if __name__ == "__main__":
    main()
