
"""
S23 — Matriz V4 con balón reconstruido.

Entrada:
    output/matrix_v3.csv, generada por S14.

Actualiza:
    J: S19, proximidad por equipo.
    K: S20, dinámica de aproximación.
    L: S21, cinemática del balón.
    M: S22, proximidad global.

Añade cinco indicadores S17:
    ball_observed
    ball_imputed
    ball_estimation_confidence
    ball_gap_length
    ball_context_support

Salida:
    output/matrix_v4.csv
    ETAPA_23_MATRIZ_V4.md

No sobrescribe matrix_v3.csv.
No cambia el orden de las 251 columnas originales.
"""

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    OUTPUT_DIR,
    N_VIDEO_FRAMES,
)

from s01_tracking import load_tracking

from s17_ball_reconstruction import (
    build_ball_reconstruction,
)

from s19_ball_proximity_reconstructed import (
    build_reconstructed_ball_proximity,
)

from s20_ball_approach_reconstructed import (
    build_reconstructed_ball_approach,
)

from s21_ball_kinematics_reconstructed import (
    build_reconstructed_ball_kinematics,
)

from s22_global_ball_proximity_reconstructed import (
    build_reconstructed_global_proximity,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

V3_PATH = OUTPUT_DIR / "matrix_v3.csv"
V4_PATH = OUTPUT_DIR / "matrix_v4.csv"

REPORT_PATH = ROOT_DIR / "ETAPA_23_MATRIZ_V4.md"

EXPECTED_ROWS = N_VIDEO_FRAMES
EXPECTED_V3_COLUMNS = 251
EXPECTED_V4_COLUMNS = 256

METADATA_COLUMNS = [
    "ball_observed",
    "ball_imputed",
    "ball_estimation_confidence",
    "ball_gap_length",
    "ball_context_support",
]

# S14 renombra dos columnas de S10 para
# evitar duplicados con otras familias.
S19_RENAMES = {
    "Barcelona_valid_position_count":
        "Barcelona_proximity_valid_position_count",

    "Atletic_Club_valid_position_count":
        "Atletic_Club_proximity_valid_position_count",
}

FAMILY_EXPECTED_FEATURES = {
    "J_S19": 16,
    "K_S20": 12,
    "L_S21": 17,
    "M_S22": 6,
}


# ============================================================
# VALIDACIONES
# ============================================================

def check_frame_alignment(df, name):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} no es un DataFrame."
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"{name}: filas incorrectas: {len(df)}"
        )

    if "frame" not in df.columns:
        raise ValueError(
            f"{name}: falta frame."
        )

    frames = df["frame"].to_numpy()

    if not np.array_equal(
        frames,
        np.arange(EXPECTED_ROWS),
    ):
        raise ValueError(
            f"{name}: frames desalineados."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            f"{name}: columnas duplicadas."
        )


def check_no_infinity(df, name):

    numeric = df.select_dtypes(
        include=[np.number]
    )

    values = numeric.to_numpy(dtype=float)

    count = int(np.isinf(values).sum())

    if count:
        raise ValueError(
            f"{name}: {count} infinitos."
        )

    return count


def count_differences(a, b):

    old = pd.to_numeric(
        a, errors="coerce"
    ).to_numpy(dtype=float)

    new = pd.to_numeric(
        b, errors="coerce"
    ).to_numpy(dtype=float)

    equal = np.isclose(
        old,
        new,
        rtol=0,
        atol=1e-9,
        equal_nan=True,
    )

    return int((~equal).sum())


def check_block(block, name, expected_features):

    check_frame_alignment(block, name)

    actual = len(block.columns) - 1

    if actual != expected_features:
        raise ValueError(
            f"{name}: {actual} variables, "
            f"esperadas {expected_features}."
        )

    check_no_infinity(block, name)


# ============================================================
# CARGA DE V3
# ============================================================

def load_v3():

    if not V3_PATH.exists():
        raise FileNotFoundError(
            f"No existe {V3_PATH}. "
            "Ejecuta S14 antes de S23."
        )

    print("Cargando matriz V3...")

    matrix = pd.read_csv(
        V3_PATH,
        low_memory=False,
    )

    check_frame_alignment(matrix, "V3")

    if len(matrix.columns) != EXPECTED_V3_COLUMNS:
        raise ValueError(
            "V3 no tiene 251 columnas: "
            f"{len(matrix.columns)}"
        )

    check_no_infinity(matrix, "V3")

    print(
        f"  V3: {len(matrix):,} filas, "
        f"{len(matrix.columns)} columnas"
    )

    return matrix


# ============================================================
# INTEGRACIÓN
# ============================================================

def replace_family(matrix, block, family_name):

    """
    Sustituye únicamente columnas ya existentes.

    Nunca añade silenciosamente variables
    pertenecientes a las familias J/K/L/M.
    """

    feature_names = [
        c for c in block.columns
        if c != "frame"
    ]

    missing = [
        c for c in feature_names
        if c not in matrix.columns
    ]

    if missing:
        raise ValueError(
            f"{family_name}: faltan columnas en V3: "
            f"{missing}"
        )

    changes = {}

    for column in feature_names:

        changes[column] = count_differences(
            matrix[column],
            block[column],
        )

        # Se asigna por posición después de
        # validar que los frames coinciden.
        matrix[column] = block[
            column
        ].to_numpy(copy=True)

    return changes


def build_matrix_v4():

    print("\n" + "=" * 65)
    print("S23 — CONSTRUCCIÓN DE MATRIZ V4")
    print("=" * 65)

    v3 = load_v3()

    original_columns = v3.columns.tolist()

    print("\nCargando tracking...")

    tracking, _ = load_tracking()

    check_frame_alignment(
        tracking, "Tracking"
    )

    print("\nReconstruyendo balón S17...")

    reconstruction = build_ball_reconstruction(
        tracking,
        strategy="temporal",
        context_weight=0.0,
    )

    check_frame_alignment(
        reconstruction, "S17"
    )

    # Reconstruir solo una vez.
    print("\n[J] Calculando S19...")

    block_j, _ = build_reconstructed_ball_proximity(
        tracking,
        reconstruction,
    )

    block_j = block_j.rename(
        columns=S19_RENAMES
    )

    check_block(
        block_j,
        "S19",
        FAMILY_EXPECTED_FEATURES["J_S19"],
    )

    print("\n[K] Calculando S20...")

    block_k, _ = build_reconstructed_ball_approach(
        tracking,
        reconstruction,
    )

    check_block(
        block_k,
        "S20",
        FAMILY_EXPECTED_FEATURES["K_S20"],
    )

    print("\n[L] Calculando S21...")

    block_l, _ = build_reconstructed_ball_kinematics(
        tracking,
        reconstruction,
    )

    check_block(
        block_l,
        "S21",
        FAMILY_EXPECTED_FEATURES["L_S21"],
    )

    print("\n[M] Calculando S22...")

    block_m, _ = build_reconstructed_global_proximity(
        tracking,
        reconstruction,
    )

    check_block(
        block_m,
        "S22",
        FAMILY_EXPECTED_FEATURES["M_S22"],
    )

    blocks = {
        "J_S19": block_j,
        "K_S20": block_k,
        "L_S21": block_l,
        "M_S22": block_m,
    }

    # No permitir que dos bloques actualizados
    # pretendan escribir la misma columna.
    all_updated_names = []

    for block in blocks.values():
        all_updated_names.extend(
            c for c in block.columns
            if c != "frame"
        )

    duplicates = pd.Index(
        all_updated_names
    )[
        pd.Index(
            all_updated_names
        ).duplicated()
    ].unique().tolist()

    if duplicates:
        raise ValueError(
            "Columnas duplicadas entre bloques: "
            f"{duplicates}"
        )

    print("\nSustituyendo familias en V3...")

    matrix = v3.copy()

    changes = {}

    for family, block in blocks.items():

        changes[family] = replace_family(
            matrix,
            block,
            family,
        )

        print(
            f"  {family}: "
            f"{len(block.columns) - 1} variables"
        )

    # ========================================================
    # METADATOS DE RECONSTRUCCIÓN
    # ========================================================

    print("\nAñadiendo cinco metadatos S17...")

    collisions = [
        c for c in METADATA_COLUMNS
        if c in matrix.columns
    ]

    if collisions:
        raise ValueError(
            "Metadatos ya existentes en V3: "
            f"{collisions}"
        )

    for column in METADATA_COLUMNS:

        if column not in reconstruction.columns:
            raise ValueError(
                f"S17 no contiene {column}"
            )

        matrix[column] = reconstruction[
            column
        ].to_numpy(copy=True)

    # ========================================================
    # VALIDACIÓN FINAL
    # ========================================================

    print("\nValidando matriz V4...")

    check_frame_alignment(
        matrix, "V4"
    )

    if matrix.shape != (
        EXPECTED_ROWS,
        EXPECTED_V4_COLUMNS,
    ):
        raise ValueError(
            "Dimensiones V4 incorrectas: "
            f"{matrix.shape}"
        )

    if (
        matrix.columns[:EXPECTED_V3_COLUMNS].tolist()
        != original_columns
    ):
        raise ValueError(
            "Se alteró el orden de columnas V3."
        )

    if (
        matrix.columns[EXPECTED_V3_COLUMNS:].tolist()
        != METADATA_COLUMNS
    ):
        raise ValueError(
            "Orden de metadatos incorrecto."
        )

    check_no_infinity(
        matrix, "V4"
    )

    # Comprobar que todas las columnas no
    # reemplazadas siguen exactamente iguales.
    updated_names = set(
        all_updated_names
    )

    untouched_columns = [
        c for c in original_columns
        if c not in updated_names
    ]

    altered_untouched = []

    for column in untouched_columns:

        if not v3[column].equals(
            matrix[column]
        ):
            altered_untouched.append(column)

    if altered_untouched:
        raise ValueError(
            "Se modificaron columnas ajenas "
            "a S19-S22: "
            f"{altered_untouched}"
        )

    # Comprobar flags de S17.
    observed = matrix[
        "ball_observed"
    ].to_numpy(dtype=int)

    imputed = matrix[
        "ball_imputed"
    ].to_numpy(dtype=int)

    relevant = reconstruction[
        "ball_reconstruction_relevant"
    ].to_numpy(dtype=bool)

    if not np.isin(
        observed, [0, 1]
    ).all():
        raise ValueError(
            "ball_observed no es binario."
        )

    if not np.isin(
        imputed, [0, 1]
    ).all():
        raise ValueError(
            "ball_imputed no es binario."
        )

    if np.any(
        (observed == 1)
        & (imputed == 1)
    ):
        raise ValueError(
            "Un frame es observado e imputado "
            "simultáneamente."
        )

    if not np.array_equal(
        imputed.astype(bool),
        relevant & ~(observed.astype(bool)),
    ):
        raise ValueError(
            "Flags de imputación inconsistentes."
        )

    confidence = matrix[
        "ball_estimation_confidence"
    ].to_numpy(dtype=float)

    if not np.isfinite(
        confidence[relevant]
    ).all():
        raise ValueError(
            "Confianza no finita en juego."
        )

    if not (
        (confidence[relevant] >= 0)
        & (confidence[relevant] <= 1)
    ).all():
        raise ValueError(
            "Confianza fuera de [0,1]."
        )

    # Cobertura y cambios.
    changed_summary = {
        family: {
            "columns_changed": sum(
                count > 0
                for count in family_changes.values()
            ),
            "total_changed_cells": sum(
                family_changes.values()
            ),
        }
        for family, family_changes in changes.items()
    }

    audit = {
        "rows": len(matrix),
        "columns": len(matrix.columns),
        "original_columns": EXPECTED_V3_COLUMNS,
        "new_metadata": len(METADATA_COLUMNS),
        "untouched_columns": len(untouched_columns),
        "observed_in_game": int(
            (
                relevant
                & (observed == 1)
            ).sum()
        ),
        "imputed_in_game": int(
            imputed.sum()
        ),
        "excluded_frames": int(
            (~relevant).sum()
        ),
        "nan_count": int(
            matrix.isna().sum().sum()
        ),
        "changed_summary": changed_summary,
        "changes": changes,
    }

    # ========================================================
    # EXPORTACIÓN
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nGuardando matrix_v4.csv...")

    matrix.to_csv(
        V4_PATH,
        index=False,
    )

    write_report(audit)

    return matrix, audit


# ============================================================
# INFORME
# ============================================================

def write_report(audit):

    lines = [
        "# ETAPA 23 — Matriz V4 integrada",
        "",
        "## Configuración",
        "- Matriz de partida: matrix_v3.csv.",
        "- Reconstrucción: S17 temporal.",
        "- Peso contextual: 0.00.",
        "- Familias actualizadas: J, K, L, M.",
        "- Scripts: S19, S20, S21, S22.",
        "",
        "## Integridad",
        f"- Frames: {audit['rows']:,}.",
        f"- Columnas V4: {audit['columns']}.",
        f"- Columnas originales V3: "
        f"{audit['original_columns']}.",
        f"- Metadatos nuevos: "
        f"{audit['new_metadata']}.",
        f"- Columnas no actualizadas e intactas: "
        f"{audit['untouched_columns']}.",
        "- Orden original de V3: preservado.",
        "- Columnas duplicadas: ninguna.",
        "- Valores infinitos: ninguno.",
        f"- Valores NaN: {audit['nan_count']:,}.",
        "",
        "## Cobertura S17",
        f"- Observaciones durante juego: "
        f"{audit['observed_in_game']:,}.",
        f"- Imputaciones durante juego: "
        f"{audit['imputed_in_game']:,}.",
        f"- Frames excluidos: "
        f"{audit['excluded_frames']:,}.",
        "",
        "## Sustitución de familias",
        "",
        "| Familia | Variables cambiadas | "
        "Celdas cambiadas |",
        "|---|---:|---:|",
    ]

    for family, summary in (
        audit["changed_summary"].items()
    ):
        lines.append(
            f"| {family} "
            f"| {summary['columns_changed']} "
            f"| {summary['total_changed_cells']:,} |"
        )

    lines.extend([
        "",
        "## Metadatos añadidos",
        "",
        "| Variable | Interpretación |",
        "|---|---|",
        "| ball_observed | Balón original observado |",
        "| ball_imputed | Balón reconstruido |",
        "| ball_estimation_confidence | "
        "Indicador heurístico de calidad, "
        "no probabilidad calibrada |",
        "| ball_gap_length | Longitud del hueco "
        "de observación en frames |",
        "| ball_context_support | Soporte contextual "
        "disponible en S17 |",
        "",
        "## Criterios",
        "- Las columnas no relacionadas con el "
        "balón permanecen intactas.",
        "- Las columnas relacionadas con el balón "
        "conservan sus nombres originales.",
        "- No se recortan velocidades extremas.",
        "- Los NaN se conservan.",
        "- La imputación no se interpreta como "
        "observación real.",
        "- Los metadatos permiten distinguir "
        "estimaciones y observaciones.",
        "",
        "## Archivos",
        f"- Matriz: `{V4_PATH}`",
        f"- Informe: `{REPORT_PATH}`",
        "",
        "## Próxima etapa",
        "S24 — Auditoría integral de la matriz V4.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    matrix, audit = build_matrix_v4()

    print("\n========== ETAPA 23 ==========")

    print(
        f"Frames: {audit['rows']:,}"
    )

    print(
        f"Columnas: {audit['columns']}"
    )

    print(
        "Columnas originales preservadas:",
        audit["original_columns"],
    )

    print(
        "Metadatos nuevos:",
        audit["new_metadata"],
    )

    print(
        "Columnas no actualizadas intactas:",
        audit["untouched_columns"],
    )

    print(
        "Observaciones durante juego:",
        f"{audit['observed_in_game']:,}",
    )

    print(
        "Imputaciones durante juego:",
        f"{audit['imputed_in_game']:,}",
    )

    print(
        "Frames excluidos:",
        f"{audit['excluded_frames']:,}",
    )

    print("\nCambios por familia:")

    for family, summary in (
        audit["changed_summary"].items()
    ):
        print(
            f"  {family}: "
            f"{summary['columns_changed']} variables, "
            f"{summary['total_changed_cells']:,} "
            "celdas"
        )

    print("\nValidación estructural: True")
    print("Orden de V3 preservado: True")
    print("Columnas ajenas a S19-S22 intactas: True")
    print("Flags S17 consistentes: True")
    print("Sin columnas duplicadas: True")
    print("Sin valores infinitos: True")

    print(f"\nMatriz: {V4_PATH}")
    print(f"Informe: {REPORT_PATH}")


if __name__ == "__main__":
    main()
