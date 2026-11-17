
from pathlib import Path

import numpy as np
import pandas as pd

from config import GAME, ROOT_DIR, N_VIDEO_FRAMES, FPS
from starter.load_smart_tagging import load


# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

TEAMS = {
    "Barcelona": "barcelona",
    "Atletic Club": "atletic",
    "N/A": "na",
}

CODES = [
    "ATTACKING TRANSITION",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "BUILD UP",
    "CREATING CHANCES",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENSIVE TRANSITION",
    "GOALS",
    "LONG BALL",
    "PLAYERS IN THE BOX",
    "PROGRESSION",
    "SET PIECES",
]

N_GENERAL = 4
N_CODES = len(CODES)
N_TEAM_CODES = len(TEAMS) * N_CODES
N_FEATURES = N_GENERAL + N_CODES + N_TEAM_CODES


# --------------------------------------------------
# UTILIDADES
# --------------------------------------------------

def normalize_code(value):
    return (value or "").strip().upper()


def normalize_team(value):
    value = (value or "N/A").strip().lower()

    if value == "barcelona":
        return "barcelona"

    if value == "atletic club":
        return "atletic"

    return "na"


# --------------------------------------------------
# CONSTRUCCIÓN DE VARIABLES
# --------------------------------------------------

def build_smart_tagging():

    print("Cargando Smart Tagging...")

    events = load(game=GAME)

    if len(events) != 638:
        raise ValueError(
            f"Se esperaban 638 eventos y se encontraron {len(events)}."
        )

    print(f"Eventos cargados: {len(events)}")
    print("Calculando variables por frame...")

    code_index = {
        code: i for i, code in enumerate(CODES)
    }

    team_names = list(TEAMS.values())
    team_index = {
        team: i for i, team in enumerate(team_names)
    }

    # Índices de las 60 variables
    idx_total = 0
    idx_barcelona = 1
    idx_atletic = 2
    idx_na = 3

    code_start = N_GENERAL
    team_code_start = N_GENERAL + N_CODES

    # Diferencias acumuladas:
    # cada intervalo suma 1 desde start hasta end - 1.
    diff = np.zeros(
        (N_VIDEO_FRAMES + 1, N_FEATURES),
        dtype=np.int32
    )

    for event in events:

        code = normalize_code(event["code"])
        team = normalize_team(event["tags"].get("Team"))

        if code not in code_index:
            raise ValueError(f"Código Smart Tagging desconocido: {code}")

        start = int(event["start_frame"])
        end = int(event["end_frame"])

        if not (0 <= start <= N_VIDEO_FRAMES):
            raise ValueError(f"Inicio fuera de rango: {start}")

        if not (0 <= end <= N_VIDEO_FRAMES):
            raise ValueError(f"Fin fuera de rango: {end}")

        if end < start:
            raise ValueError(
                f"Intervalo inválido: start={start}, end={end}"
            )

        # Los intervalos vacíos no aportan actividad.
        if start == end:
            continue

        c_idx = code_index[code]
        t_idx = team_index[team]

        # Contador general
        diff[start, idx_total] += 1
        diff[end, idx_total] -= 1

        # Contador por equipo
        team_feature = {
            "barcelona": idx_barcelona,
            "atletic": idx_atletic,
            "na": idx_na,
        }[team]

        diff[start, team_feature] += 1
        diff[end, team_feature] -= 1

        # Contador por código
        feature_code = code_start + c_idx

        diff[start, feature_code] += 1
        diff[end, feature_code] -= 1

        # Contador por equipo y código
        feature_team_code = (
            team_code_start + t_idx * N_CODES + c_idx
        )

        diff[start, feature_team_code] += 1
        diff[end, feature_team_code] -= 1

    # Acumulación temporal
    values = np.cumsum(diff[:-1], axis=0)

    # Nombres de las columnas
    columns = [
        "smart_tagging_active_count",
        "smart_tagging_barcelona_count",
        "smart_tagging_atletic_count",
        "smart_tagging_na_count",
    ]

    for code in CODES:
        code_name = code.lower().replace(" ", "_")
        columns.append(f"smart_tagging_code_{code_name}_count")

    for team_name in team_names:
        for code in CODES:
            code_name = code.lower().replace(" ", "_")
            columns.append(
                f"smart_tagging_{team_name}_{code_name}_count"
            )

    features = pd.DataFrame(values, columns=columns)
    features.insert(
        0,
        "frame",
        np.arange(N_VIDEO_FRAMES, dtype=np.int32)
    )

    return features, events


# --------------------------------------------------
# VALIDACIÓN
# --------------------------------------------------

def validate_smart_tagging(features, events):

    print("\nValidando estructura...")

    expected_frames = np.arange(N_VIDEO_FRAMES)

    assert len(features) == N_VIDEO_FRAMES
    assert features.shape[1] == 61
    assert features["frame"].is_unique
    assert np.array_equal(
        features["frame"].to_numpy(),
        expected_frames
    )

    count_columns = features.columns.drop("frame")

    assert (features[count_columns] >= 0).all().all()

    # El total debe coincidir con la suma de los códigos.
    code_columns = [
        c for c in count_columns
        if c.startswith("smart_tagging_code_")
    ]

    assert np.array_equal(
        features["smart_tagging_active_count"].to_numpy(),
        features[code_columns].sum(axis=1).to_numpy()
    )

    # Los equipos deben sumar el total general.
    team_columns = [
        "smart_tagging_barcelona_count",
        "smart_tagging_atletic_count",
        "smart_tagging_na_count",
    ]

    assert np.array_equal(
        features["smart_tagging_active_count"].to_numpy(),
        features[team_columns].sum(axis=1).to_numpy()
    )

    print("Filas:", len(features))
    print("Columnas:", len(features.columns))
    print(
        "Frames:",
        features["frame"].min(),
        "a",
        features["frame"].max()
    )
    print("Eventos originales:", len(events))
    print(
        "Frames con Smart Tagging activo:",
        int((features["smart_tagging_active_count"] > 0).sum())
    )
    print(
        "Máximo de eventos simultáneos:",
        int(features["smart_tagging_active_count"].max())
    )
    print("Consistencia de códigos: True")
    print("Consistencia de equipos: True")
    print("Alineación correcta: True")

    return True


# --------------------------------------------------
# DOCUMENTACIÓN
# --------------------------------------------------

def write_report(features, events):

    report_path = ROOT_DIR / "ETAPA_05_SMART_TAGGING.md"

    active_frames = int(
        (features["smart_tagging_active_count"] > 0).sum()
    )

    report = f"""# Etapa 05 — Smart Tagging por frame

## Objetivo

Transformar los intervalos de Smart Tagging en variables
por frame de vídeo táctico.

## Datos de entrada

- Partido: {GAME}
- FPS: {FPS}
- Eventos originales: {len(events)}
- Frames: {N_VIDEO_FRAMES}

## Variables generadas

- 4 contadores generales y por equipo.
- 14 contadores por código táctico.
- 42 contadores por equipo y código.
- Total: 60 variables, más la columna de alineación `frame`.

## Criterios

- Conversión temporal realizada por el loader oficial.
- Intervalos semiabiertos: [start_frame, end_frame).
- Los eventos simultáneos se contabilizan por separado.
- Los equipos se agrupan en Barcelona, Atletic Club y N/A.
- Los códigos se normalizan a mayúsculas.
- No se interpretan los eventos como posesiones.
- No se imputan intervalos ni se crean etiquetas.

## Validación estructural

- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Frames activos: {active_frames}
- Máximo de eventos simultáneos: {int(features["smart_tagging_active_count"].max())}
- Alineación: correcta
- Consistencia de códigos: correcta
- Consistencia de equipos: correcta

## Resultado

Etapa 05 completada. Las variables quedan disponibles
en memoria para su integración posterior en la matriz V3.
"""

    report_path.write_text(report, encoding="utf-8")

    print("\nDocumento generado:", report_path.name)


# --------------------------------------------------
# EJECUCIÓN
# --------------------------------------------------

def main():

    features, events = build_smart_tagging()

    validate_smart_tagging(features, events)

    write_report(features, events)

    print("\nETAPA 05 COMPLETADA.")


if __name__ == "__main__":
    main()
