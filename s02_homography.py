
import numpy as np
import pandas as pd

from config import ROOT_DIR, N_VIDEO_FRAMES
from games import GAME
from starter.load_homography import load


SOURCE = "tactical"
REPORT_PATH = ROOT_DIR / "ETAPA_02_HOMOGRAFIA.md"


def load_homography_features():
    print("Cargando homografía Tactical...")

    matrices, reprojection_error, valid = load(
        game=GAME,
        source=SOURCE,
    )

    matrices = np.asarray(matrices)
    reprojection_error = np.asarray(reprojection_error)
    valid = np.asarray(valid, dtype=bool)

    expected_shape = (N_VIDEO_FRAMES, 3, 3)

    if matrices.shape != expected_shape:
        raise ValueError(
            f"Dimensión de matrices incorrecta: "
            f"{matrices.shape}; se esperaba {expected_shape}"
        )

    if len(reprojection_error) != N_VIDEO_FRAMES:
        raise ValueError("Longitud incorrecta en reprojection_error.")

    if len(valid) != N_VIDEO_FRAMES:
        raise ValueError("Longitud incorrecta en valid.")

    features = pd.DataFrame({
        "frame": np.arange(N_VIDEO_FRAMES, dtype=np.int64),
        "homography_reprojection_error": reprojection_error,
        "homography_valid": valid,
    })

    return features, matrices


def validate_homography(features, matrices):
    errors = features["homography_reprojection_error"]
    valid = features["homography_valid"]

    finite_errors = errors[np.isfinite(errors)]

    return {
        "rows": len(features),
        "columns": len(features.columns),
        "frame_min": int(features["frame"].min()),
        "frame_max": int(features["frame"].max()),
        "unique_frames": features["frame"].is_unique,
        "matrices_shape": matrices.shape,
        "valid_frames": int(valid.sum()),
        "invalid_frames": int((~valid).sum()),
        "finite_error_frames": int(np.isfinite(errors).sum()),
        "error_mean": (
            float(finite_errors.mean())
            if len(finite_errors) else None
        ),
        "error_median": (
            float(finite_errors.median())
            if len(finite_errors) else None
        ),
        "alignment_ok": (
            len(features) == N_VIDEO_FRAMES
            and features["frame"].is_unique
            and features["frame"].iloc[0] == 0
            and features["frame"].iloc[-1] == N_VIDEO_FRAMES - 1
            and matrices.shape == (N_VIDEO_FRAMES, 3, 3)
        ),
    }


def generate_report(validation):
    lines = [
        "# ETAPA 02 — HOMOGRAFÍA TÁCTICA",
        "",
        "## Fuente",
        "",
        "- Origen: Tactical",
        "- Carga: `starter.load_homography.load()`",
        "- Correspondencia: `frame` de 0 a 199318",
        "",
        "## Validación estructural",
        "",
        f"- Filas: {validation['rows']:,}",
        f"- Columnas: {validation['columns']}",
        f"- Frame mínimo: {validation['frame_min']}",
        f"- Frame máximo: {validation['frame_max']}",
        f"- Frames únicos: {validation['unique_frames']}",
        f"- Dimensión de matrices: `{validation['matrices_shape']}`",
        f"- Frames con homografía válida: {validation['valid_frames']:,}",
        f"- Frames con homografía inválida: {validation['invalid_frames']:,}",
        f"- Errores finitos: {validation['finite_error_frames']:,}",
        "",
        "## Error de reproyección",
        "",
        f"- Media: {validation['error_mean']}",
        f"- Mediana: {validation['error_median']}",
        "",
        "## Resultado",
        "",
        f"- Alineación estructural correcta: {validation['alignment_ok']}",
        "",
        "Las matrices se conservan en memoria. No se generan archivos "
        "intermedios ni se transforman coordenadas en esta etapa.",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    features, matrices = load_homography_features()
    validation = validate_homography(features, matrices)

    print("\nResultado de carga:")
    print(f"Filas: {validation['rows']:,}")
    print(f"Columnas: {validation['columns']}")
    print(
        f"Frames: {validation['frame_min']} "
        f"a {validation['frame_max']}"
    )
    print(f"Forma de matrices: {validation['matrices_shape']}")
    print(f"Homografías válidas: {validation['valid_frames']:,}")
    print(f"Homografías inválidas: {validation['invalid_frames']:,}")
    print(f"Error medio: {validation['error_mean']}")
    print(f"Error mediano: {validation['error_median']}")
    print(f"Alineación correcta: {validation['alignment_ok']}")

    generate_report(validation)

    if not validation["alignment_ok"]:
        raise RuntimeError("ETAPA 02: falló la validación estructural.")

    print(f"\nDocumento generado: {REPORT_PATH.name}")
    print("ETAPA 02 COMPLETADA.")


if __name__ == "__main__":
    main()
