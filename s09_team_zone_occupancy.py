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

ZONE_X = ("X1", "X2", "X3")
ZONE_Y = ("Y1", "Y2", "Y3")


# ---------------------------------------------------------
# CÁLCULO DE ZONAS POR EQUIPO
# ---------------------------------------------------------


def calculate_team_zones(
    df,
    team_name,
    slots,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
):
    x_cols = [f"{slot}_x" for slot in slots]
    y_cols = [f"{slot}_y" for slot in slots]

    x = df[x_cols].to_numpy(dtype=float) * pitch_length_m
    y = df[y_cols].to_numpy(dtype=float) * pitch_width_m

    valid = np.isfinite(x) & np.isfinite(y)

    inside = (
        valid
        & (x >= 0) & (x <= pitch_length_m)
        & (y >= 0) & (y <= pitch_width_m)
    )

    result = {
        f"{team_name}_valid_position_count": valid.sum(axis=1),
        f"{team_name}_inside_position_count": inside.sum(axis=1),
    }

    x_safe = np.where(inside, x, 0.0)
    y_safe = np.where(inside, y, 0.0)

    x_zone = np.minimum(
        (x_safe / (pitch_length_m / 3)).astype(int),
        2,
    )
    y_zone = np.minimum(
        (y_safe / (pitch_width_m / 3)).astype(int),
        2,
    )

    for xi, x_name in enumerate(ZONE_X):
        for yi, y_name in enumerate(ZONE_Y):
            zone_name = f"{x_name}_{y_name}"

            mask = inside & (x_zone == xi) & (y_zone == yi)
            count = mask.sum(axis=1)

            result[f"{team_name}_zone_{zone_name}_count"] = count

            inside_count = result[
                f"{team_name}_inside_position_count"
            ]

            result[f"{team_name}_zone_{zone_name}_ratio"] = np.divide(
                count,
                inside_count,
                out=np.full(len(count), np.nan, dtype=float),
                where=inside_count > 0,
            )

    return result


def build_team_zone_occupancy(
    tracking,
    pitch_length_m=105.3,
    pitch_width_m=68.0,
):
    """Construye las 49 variables de ocupación de zonas."""

    required = ["frame"]

    for slots in TEAMS.values():
        for slot in slots:
            required.extend([f"{slot}_x", f"{slot}_y"])

    missing = [col for col in required if col not in tracking.columns]
    assert not missing, f"Faltan columnas: {missing}"

    features = pd.DataFrame({
        "frame": tracking["frame"].to_numpy()
    })

    team_results = {}

    for team_name, slots in TEAMS.items():
        team_results[team_name] = calculate_team_zones(
            tracking,
            team_name,
            slots,
            pitch_length_m,
            pitch_width_m,
        )

        for column, values in team_results[team_name].items():
            features[column] = values

    for x_name in ZONE_X:
        for y_name in ZONE_Y:
            zone_name = f"{x_name}_{y_name}"

            barcelona_col = f"Barcelona_zone_{zone_name}_count"
            atletic_col = f"Atletic_Club_zone_{zone_name}_count"

            features[
                f"zone_{zone_name}_count_difference"
            ] = features[barcelona_col] - features[atletic_col]

    validate_features(features)

    return features



# ---------------------------------------------------------
# VALIDACIÓN
# ---------------------------------------------------------

def validate_features(features):
    assert len(features) == N_VIDEO_FRAMES
    assert len(features.columns) == 50  # 49 variables + frame

    assert features["frame"].iloc[0] == 0
    assert features["frame"].iloc[-1] == N_VIDEO_FRAMES - 1
    assert features["frame"].is_unique

    for team_name in TEAMS:
        inside_col = f"{team_name}_inside_position_count"

        count_cols = [
            f"{team_name}_zone_{x}_{y}_count"
            for x in ZONE_X
            for y in ZONE_Y
        ]

        ratio_cols = [
            f"{team_name}_zone_{x}_{y}_ratio"
            for x in ZONE_X
            for y in ZONE_Y
        ]

        # La suma de las nueve zonas debe coincidir
        # con el total de posiciones interiores.
        zone_sum = features[count_cols].sum(axis=1)

        assert np.array_equal(
            zone_sum.to_numpy(),
            features[inside_col].to_numpy(),
        )

        # Las proporciones deben estar entre 0 y 1.
        ratios = features[ratio_cols].to_numpy(dtype=float)
        finite_ratios = ratios[np.isfinite(ratios)]

        assert ((finite_ratios >= 0) & (finite_ratios <= 1)).all()

        # Si hay posiciones interiores, las proporciones suman 1.
        has_inside = features[inside_col] > 0
        ratio_sum = features.loc[has_inside, ratio_cols].sum(axis=1)

        assert np.allclose(ratio_sum.to_numpy(), 1.0)

    difference_cols = [
        f"zone_{x}_{y}_count_difference"
        for x in ZONE_X
        for y in ZONE_Y
    ]

    for x_name in ZONE_X:
        for y_name in ZONE_Y:
            zone_name = f"{x_name}_{y_name}"

            barcelona_col = (
                f"Barcelona_zone_{zone_name}_count"
            )
            atletic_col = (
                f"Atletic_Club_zone_{zone_name}_count"
            )
            diff_col = (
                f"zone_{zone_name}_count_difference"
            )

            expected = (
                features[barcelona_col] - features[atletic_col]
            )

            assert np.array_equal(
                features[diff_col].to_numpy(),
                expected.to_numpy(),
            )


# ---------------------------------------------------------
# DOCUMENTACIÓN
# ---------------------------------------------------------

def write_report(features, output_path):
    barcelona_inside = features[
        "Barcelona_inside_position_count"
    ].mean()

    atletic_inside = features[
        "Atletic_Club_inside_position_count"
    ].mean()

    report = f"""# ETAPA 09 — Ocupación de zonas por equipo

## Resultado
- Filas: {len(features)}
- Columnas: {len(features.columns)}
- Variables de la familia: 49
- Frames: {features['frame'].min()} a {features['frame'].max()}

## División espacial
- Campo: {PITCH_LENGTH_M} × {PITCH_WIDTH_M} metros.
- Eje X: tres tercios longitudinales (X1, X2, X3).
- Eje Y: tres franjas transversales (Y1, Y2, Y3).
- Total: nueve zonas.

## Variables por equipo
- Número de posiciones válidas.
- Número de posiciones dentro del campo.
- Conteo de posiciones en cada una de las nueve zonas.
- Proporción de posiciones interiores en cada zona.

## Variables comparativas
- Diferencia de conteo por zona: Barcelona menos Athletic.

## Resumen
- Media de posiciones interiores Barcelona: {barcelona_inside:.3f}
- Media de posiciones interiores Athletic: {atletic_inside:.3f}

## Criterios
- Solo se asignan zonas a posiciones finitas dentro del campo.
- Las proporciones usan como denominador las posiciones interiores del equipo.
- Si un equipo no tiene posiciones interiores en un frame, sus proporciones quedan como NaN.
- Los límites máximos del campo se asignan al tercer tercio correspondiente.
- X1/X3 y Y1/Y3 son referencias geométricas, no direcciones de ataque.

## Validación
- Alineación y unicidad de frames: correctas.
- Suma de conteos por zona igual al total interior: correcta.
- Proporciones dentro de rango y suma igual a 1 cuando hay posiciones: correcta.
- Diferencias Barcelona menos Athletic: correctas.

Esta etapa describe la distribución espacial de los tracks; no determina posesión.
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

    print("Calculando ocupación de zonas por equipo...")

    barcelona = calculate_team_zones(
        df, "Barcelona", TEAMS["Barcelona"]
    )
    atletic = calculate_team_zones(
        df, "Atletic_Club", TEAMS["Atletic_Club"]
    )

    features = pd.DataFrame({
        "frame": df["frame"].to_numpy(),
    })

    # Añadir variables de cada equipo.
    for team_data in (barcelona, atletic):
        for column, values in team_data.items():
            if column == "_zone_counts":
                continue
            features[column] = values

    # Diferencias de conteos por zona.
    for x_name in ZONE_X:
        for y_name in ZONE_Y:
            zone_name = f"{x_name}_{y_name}"

            barcelona_col = (
                f"Barcelona_zone_{zone_name}_count"
            )
            atletic_col = (
                f"Atletic_Club_zone_{zone_name}_count"
            )

            features[
                f"zone_{zone_name}_count_difference"
            ] = features[barcelona_col] - features[atletic_col]

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
            "Media de posiciones interiores:",
            round(
                features[
                    f"{team_name}_inside_position_count"
                ].mean(),
                3,
            ),
        )

    print("\nSuma de zonas consistente: True")
    print("Proporciones consistentes: True")
    print("Diferencias consistentes: True")
    print("Alineación correcta: True")

    report_path = ROOT_DIR / "ETAPA_09_OCUPACION_ZONAS.md"
    write_report(features, report_path)

    print(f"\nDocumento generado: {report_path.name}")
    print("\nETAPA 09 COMPLETADA.")


if __name__ == "__main__":
    main()