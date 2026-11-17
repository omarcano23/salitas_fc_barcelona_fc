
from pathlib import Path

import pandas as pd

from config import ROOT_DIR, N_VIDEO_FRAMES
from games import GAME, GAMES
from starter.load_atd import load


# ============================================================
# CONFIGURACIÓN
# ============================================================

SOURCE = "tactical"
REPORT_PATH = ROOT_DIR / "ETAPA_01_TRACKING.md"


# ============================================================
# CARGA
# ============================================================

def load_tracking():
    """
    Carga el tracking ATD Tactical y devuelve:

    tracking: DataFrame con una fila por frame.
    dataset: objeto original de Kloppy.
    """

    print("Cargando tracking ATD Tactical...")

    dataset = load(game=GAME, source=SOURCE)

    tracking = dataset.to_df()

    # Asegurar que frame_id sea una columna.
    if "frame_id" not in tracking.columns:
        if tracking.index.name == "frame_id":
            tracking = tracking.reset_index()
        else:
            raise ValueError(
                "No se encontró frame_id en el DataFrame de Kloppy."
            )

    # Orden temporal.
    tracking = tracking.sort_values("frame_id").reset_index(drop=True)

    # Conservar el identificador original y crear el frame de vídeo.
    tracking["frame_id"] = tracking["frame_id"].astype("int64")
    tracking["frame"] = tracking["frame_id"] - 1

    return tracking, dataset


# ============================================================
# VALIDACIÓN GENERAL
# ============================================================

def validate_tracking(tracking):

    expected_frames = N_VIDEO_FRAMES

    frame_count = len(tracking)

    frame_min = int(tracking["frame"].min())
    frame_max = int(tracking["frame"].max())

    frame_unique = tracking["frame"].is_unique

    alignment_ok = (
        frame_count == expected_frames
        and frame_min == 0
        and frame_max == expected_frames - 1
        and frame_unique
    )

    return {
        "rows": frame_count,
        "columns": len(tracking.columns),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "unique_frames": frame_unique,
        "alignment_ok": alignment_ok,
    }


# ============================================================
# DOCUMENTACIÓN DE ETAPA
# ============================================================

def generate_report(tracking, dataset, validation):

    teams = [
        str(team)
        for team in dataset.metadata.teams
    ]

    pitch_length = dataset.metadata.pitch_dimensions.pitch_length
    pitch_width = dataset.metadata.pitch_dimensions.pitch_width

    lines = [
        "# ETAPA 01 — TRACKING TACTICAL",
        "",
        "## Objetivo",
        "",
        "Cargar el tracking ATD Tactical mediante el loader oficial.",
        "Conservar los campos originales y añadir el índice de vídeo.",
        "",
        "## Fuente",
        "",
        "- Partido: BAR-ATH",
        "- Fuente: Tactical",
        "- Loader: `starter/load_atd.py`",
        "- Formato de lectura: Kloppy DataFrame",
        "",
        "## Dimensiones",
        "",
        f"- Filas: {validation['rows']:,}",
        f"- Columnas: {validation['columns']}",
        f"- Primer frame: {validation['frame_min']}",
        f"- Último frame: {validation['frame_max']}",
        f"- FPS: 25",
        f"- Equipos detectados: {', '.join(teams)}",
        f"- Campo según metadata ATD: {pitch_length} × {pitch_width} m",
        "",
        "## Variables disponibles",
        "",
        "| Campo | Tipo |",
        "|---|---|",
    ]

    for column in tracking.columns:
        lines.append(
            f"| `{column}` | `{tracking[column].dtype}` |"
        )

    lines.extend([
        "",
        "## Convención temporal",
        "",
        "- `frame_id`: identificador original ATD, comienza en 1.",
        "- `frame`: índice de vídeo, comienza en 0.",
        "- Conversión: `frame = frame_id - 1`.",
        "",
        "## Validación general",
        "",
        f"- Número de filas esperado: {N_VIDEO_FRAMES:,}",
        f"- Frames únicos: {validation['unique_frames']}",
        f"- Alineación estructural correcta: {validation['alignment_ok']}",
        "",
        "## Resultado",
        "",
        (
            "ETAPA COMPLETADA."
            if validation["alignment_ok"]
            else "ETAPA PENDIENTE: revisar dimensiones o alineación."
        ),
        "",
        "Los datos permanecen en memoria. No se generó una tabla intermedia.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# EJECUCIÓN DIRECTA
# ============================================================

def main():

    tracking, dataset = load_tracking()

    validation = validate_tracking(tracking)

    print("\nResultado de carga:")
    print(f"Filas: {validation['rows']:,}")
    print(f"Columnas: {validation['columns']}")
    print(
        f"Frames: {validation['frame_min']} "
        f"a {validation['frame_max']}"
    )
    print(f"Alineación correcta: {validation['alignment_ok']}")

    print("\nEquipos:")
    for team in dataset.metadata.teams:
        print(f"- {team}")

    print("\nPrimeras columnas:")
    print(list(tracking.columns[:15]))

    generate_report(tracking, dataset, validation)

    print(f"\nDocumento generado: {REPORT_PATH.name}")

    if not validation["alignment_ok"]:
        raise RuntimeError(
            "La validación general del tracking no fue superada."
        )

    print("\nETAPA 01 COMPLETADA.")


if __name__ == "__main__":
    main()
