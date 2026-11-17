
"""S17 v2 — Reconstrucción híbrida del balón.

- Conserva las observaciones originales.
- Estima coordenadas durante FIRST_HALF y SECOND_HALF.
- Mantiene NaN en PRE_MATCH, HALFTIME y POST_MATCH.
- Permite comparar temporal, contextual e hybrid.
- No utiliza ground truth de posesión.
- No genera CSV intermedios.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ROOT_DIR,
    FPS,
    N_VIDEO_FRAMES,
    FIRST_HALF_START,
    FIRST_HALF_END,
    SECOND_HALF_START,
    SECOND_HALF_END,
)
from games import GAMES, GAME
from s01_tracking import load_tracking


# ============================================================
# CONFIGURACIÓN
# ============================================================

PITCH_LENGTH_M, PITCH_WIDTH_M = map(
    float, GAMES[GAME]["pitch"]
)

REPORT_PATH = (
    ROOT_DIR / "ETAPA_17_RECONSTRUCCION_BALON.md"
)

SEGMENTS = (
    (FIRST_HALF_START, FIRST_HALF_END + 1),
    (SECOND_HALF_START, SECOND_HALF_END + 1),
)

SHORT_GAP = 10
MEDIUM_GAP = 75

MAX_ENDPOINT_SPEED_MPS = 35.0
CONTEXT_RADIUS_M = 15.0

DEFAULT_STRATEGY = "hybrid"
DEFAULT_CONTEXT_WEIGHT = 0.50

VALID_STRATEGIES = {
    "temporal",
    "contextual",
    "hybrid",
}

PITCH_SCALE = np.array([
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
])


# ============================================================
# JUGADORES
# ============================================================

def get_player_arrays(tracking):
    """Extrae los 46 slots de jugadores/tracks."""

    xs = []
    ys = []

    for slot in range(1, 47):
        xs.append(
            pd.to_numeric(
                tracking[f"{slot}_x"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

        ys.append(
            pd.to_numeric(
                tracking[f"{slot}_y"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

    px = np.stack(xs, axis=1)
    py = np.stack(ys, axis=1)

    valid = (
        np.isfinite(px)
        & np.isfinite(py)
        & (px >= 0)
        & (px <= 1)
        & (py >= 0)
        & (py <= 1)
    )

    return px, py, valid


# ============================================================
# MODELO TEMPORAL
# ============================================================

def endpoint_velocity(
    xy,
    observed,
    anchor,
    direction,
    segment_start,
    segment_end,
):
    """Velocidad local, limitada y sin cruzar periodos."""

    neighbor = anchor + direction

    if (
        neighbor < segment_start
        or neighbor >= segment_end
        or not observed[neighbor]
    ):
        return np.zeros(2, dtype=float)

    dt = direction / FPS

    velocity = (
        xy[neighbor] - xy[anchor]
    ) / dt

    velocity_m = velocity * PITCH_SCALE
    speed = np.linalg.norm(velocity_m)

    if speed > MAX_ENDPOINT_SPEED_MPS:
        velocity *= MAX_ENDPOINT_SPEED_MPS / speed

    return velocity


def temporal_estimate(
    xy,
    observed,
    start,
    end,
    left,
    right,
    segment_start,
    segment_end,
):
    """Hermite en huecos cortos; referencia lineal en largos."""

    length = end - start

    result = np.full(
        (length, 2),
        np.nan,
        dtype=float,
    )

    if left is not None and right is not None:

        p0 = xy[left]
        p1 = xy[right]

        t = (
            np.arange(start, end) - left
        ) / (right - left)

        linear = (
            (1.0 - t[:, None]) * p0
            + t[:, None] * p1
        )

        if length > SHORT_GAP:
            return linear

        duration = (right - left) / FPS

        v0 = endpoint_velocity(
            xy, observed, left, -1,
            segment_start, segment_end,
        )

        v1 = endpoint_velocity(
            xy, observed, right, +1,
            segment_start, segment_end,
        )

        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2

        result = (
            h00[:, None] * p0
            + h10[:, None] * duration * v0
            + h01[:, None] * p1
            + h11[:, None] * duration * v1
        )

        # Control de sobreoscilaciones respecto a la recta.
        deviation_m = np.linalg.norm(
            (result - linear) * PITCH_SCALE,
            axis=1,
        )

        result[deviation_m > 5.0] = (
            linear[deviation_m > 5.0]
        )

        return result

    if left is not None:
        result[:] = xy[left]
        return result

    if right is not None:
        result[:] = xy[right]
        return result

    return result


# ============================================================
# MODELO CONTEXTUAL
# ============================================================

def context_estimate(
    px,
    py,
    valid,
    start,
    end,
    reference,
):
    """Hipótesis basada en el jugador más cercano a la referencia.

    No implica que ese jugador posea el balón.
    El soporte es geométrico, NO una probabilidad calibrada.
    """

    length = end - start

    result = np.full(
        (length, 2),
        np.nan,
        dtype=float,
    )

    support = np.zeros(
        length,
        dtype=float,
    )

    for i, frame in enumerate(range(start, end)):

        mask = valid[frame]

        if not np.any(mask):
            continue

        players = np.column_stack((
            px[frame, mask],
            py[frame, mask],
        ))

        ref = reference[i]

        if np.all(np.isfinite(ref)):

            distances = np.linalg.norm(
                (players - ref) * PITCH_SCALE,
                axis=1,
            )

            nearest = int(np.argmin(distances))

            result[i] = players[nearest]

            support[i] = float(
                np.exp(
                    -0.5 * (
                        distances[nearest]
                        / CONTEXT_RADIUS_M
                    ) ** 2
                )
            )

        else:
            # Prior débil si no existe referencia temporal.
            result[i] = players.mean(axis=0)
            support[i] = 0.0

    return result, support


# ============================================================
# FUSIÓN DE ESTRATEGIAS
# ============================================================

def fuse_estimates(
    temporal,
    contextual,
    support,
    length,
    anchors,
    strategy,
    context_weight,
):
    """Combina hipótesis y devuelve valores por frame."""

    n = len(temporal)

    result = np.full(
        (n, 2),
        np.nan,
        dtype=float,
    )

    methods = np.full(
        n,
        "",
        dtype=object,
    )

    quality = np.zeros(n, dtype=float)

    has_temporal = np.isfinite(
        temporal
    ).all(axis=1)

    has_context = np.isfinite(
        contextual
    ).all(axis=1)

    both = has_temporal & has_context

    temporal_only = has_temporal & ~has_context
    context_only = has_context & ~has_temporal

    # Sin anclajes suficientes, reducimos confianza.
    anchor_factor = (
        1.0 if anchors == 2
        else 0.5 if anchors == 1
        else 0.25
    )

    if strategy == "temporal":

        result[has_temporal] = temporal[has_temporal]
        methods[has_temporal] = "TEMPORAL"

        quality[has_temporal] = (
            0.35 * anchor_factor
        )

        result[context_only] = contextual[context_only]
        methods[context_only] = "CONTEXT_FALLBACK"

        quality[context_only] = 0.02

    elif strategy == "contextual":

        result[has_context] = contextual[has_context]
        methods[has_context] = "CONTEXTUAL"

        quality[has_context] = (
            0.20 * anchor_factor
            * (0.5 + 0.5 * support[has_context])
        )

        result[temporal_only] = temporal[temporal_only]
        methods[temporal_only] = "TEMPORAL_FALLBACK"

        quality[temporal_only] = (
            0.10 * anchor_factor
        )

    else:

        if length <= SHORT_GAP and anchors == 2:
            weight = 0.0
            method = "HYBRID_SHORT"
            base_quality = 0.80

        elif length <= MEDIUM_GAP:
            weight = context_weight * 0.5
            method = "HYBRID_MEDIUM"
            base_quality = 0.40

        else:
            weight = context_weight
            method = "HYBRID_LONG"
            base_quality = 0.10

        result[both] = (
            (1.0 - weight) * temporal[both]
            + weight * contextual[both]
        )

        methods[both] = method

        quality[both] = (
            base_quality
            * anchor_factor
            * (0.5 + 0.5 * support[both])
        )

        result[temporal_only] = (
            temporal[temporal_only]
        )
        methods[temporal_only] = "TEMPORAL_FALLBACK"
        quality[temporal_only] = (
            0.10 * anchor_factor
        )

        result[context_only] = (
            contextual[context_only]
        )
        methods[context_only] = "CONTEXT_FALLBACK"
        quality[context_only] = (
            0.02 * anchor_factor
        )

    # Prior neutral solo para garantizar cobertura.
    # Debe interpretarse como una hipótesis muy débil.
    invalid = ~np.isfinite(result).all(axis=1)

    result[invalid] = [0.5, 0.5]
    methods[invalid] = "WEAK_SPATIAL_PRIOR"
    quality[invalid] = 0.01

    return result, methods, quality


# ============================================================
# RECONSTRUCCIÓN PRINCIPAL
# ============================================================

def build_ball_reconstruction(
    tracking,
    strategy=DEFAULT_STRATEGY,
    context_weight=DEFAULT_CONTEXT_WEIGHT,
):

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Estrategia desconocida: {strategy}"
        )

    if not 0.0 <= context_weight <= 1.0:
        raise ValueError(
            "context_weight debe estar entre 0 y 1."
        )

    n = len(tracking)

    if n != N_VIDEO_FRAMES:
        raise ValueError(
            f"Frames incorrectos: {n}"
        )

    if not np.array_equal(
        tracking["frame"].to_numpy(),
        np.arange(n),
    ):
        raise ValueError(
            "El tracking no está alineado."
        )

    x = pd.to_numeric(
        tracking["ball_x"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        tracking["ball_y"],
        errors="coerce",
    ).to_numpy(dtype=float)

    xy = np.column_stack((x, y))

    observed = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    relevant = np.zeros(n, dtype=bool)

    for start, end in SEGMENTS:
        relevant[start:end] = True

    estimated = np.full(
        (n, 2),
        np.nan,
        dtype=float,
    )

    methods = np.full(
        n,
        "EXCLUDED",
        dtype=object,
    )

    confidence = np.full(
        n,
        np.nan,
        dtype=float,
    )

    gap_length = np.zeros(
        n,
        dtype=np.int32,
    )

    context_support = np.full(
        n,
        np.nan,
        dtype=float,
    )

    px, py, player_valid = get_player_arrays(tracking)

    for segment_start, segment_end in SEGMENTS:

        indices = np.arange(
            segment_start,
            segment_end,
        )

        segment_observed = observed[indices]

        observed_indices = indices[segment_observed]

        estimated[observed_indices] = (
            xy[observed_indices]
        )

        methods[observed_indices] = "OBSERVED"
        confidence[observed_indices] = 1.0

        missing = ~segment_observed

        transitions = np.diff(
            np.r_[
                False,
                missing,
                False,
            ].astype(np.int8)
        )

        starts = np.flatnonzero(
            transitions == 1
        )

        ends = np.flatnonzero(
            transitions == -1
        )

        for relative_start, relative_end in zip(
            starts, ends
        ):

            start = (
                segment_start + int(relative_start)
            )

            end = (
                segment_start + int(relative_end)
            )

            length = end - start

            left = (
                start - 1
                if (
                    start > segment_start
                    and observed[start - 1]
                )
                else None
            )

            right = (
                end
                if (
                    end < segment_end
                    and observed[end]
                )
                else None
            )

            anchors = (
                int(left is not None)
                + int(right is not None)
            )

            gap_length[start:end] = length

            temporal = temporal_estimate(
                xy,
                observed,
                start,
                end,
                left,
                right,
                segment_start,
                segment_end,
            )

            contextual, support = context_estimate(
                px,
                py,
                player_valid,
                start,
                end,
                temporal,
            )

            result, selected_methods, quality = (
                fuse_estimates(
                    temporal,
                    contextual,
                    support,
                    length,
                    anchors,
                    strategy,
                    context_weight,
                )
            )

            estimated[start:end] = result
            methods[start:end] = selected_methods
            confidence[start:end] = quality
            context_support[start:end] = support

    imputed = relevant & ~observed

    features = pd.DataFrame({
        "frame": np.arange(n, dtype=np.int32),

        "ball_x_raw": x,
        "ball_y_raw": y,

        "ball_observed": observed.astype(np.int8),

        "ball_reconstruction_relevant":
            relevant.astype(np.int8),

        "ball_x_estimated": estimated[:, 0],
        "ball_y_estimated": estimated[:, 1],

        "ball_imputed": imputed.astype(np.int8),

        "ball_estimation_method": methods,

        "ball_estimation_confidence": confidence,

        "ball_gap_length": gap_length,

        "ball_context_support": context_support,
    })

    validate_reconstruction(features)

    return features


# ============================================================
# VALIDACIÓN
# ============================================================

def validate_reconstruction(features):

    if len(features) != N_VIDEO_FRAMES:
        raise ValueError(
            "Cantidad incorrecta de frames."
        )

    frames = features["frame"].to_numpy()

    if not np.array_equal(
        frames,
        np.arange(N_VIDEO_FRAMES),
    ):
        raise ValueError(
            "Frames desalineados."
        )

    relevant = (
        features[
            "ball_reconstruction_relevant"
        ].to_numpy(dtype=bool)
    )

    observed = (
        features[
            "ball_observed"
        ].to_numpy(dtype=bool)
    )

    imputed = (
        features[
            "ball_imputed"
        ].to_numpy(dtype=bool)
    )

    estimated = features[
        ["ball_x_estimated", "ball_y_estimated"]
    ].to_numpy(dtype=float)

    raw = features[
        ["ball_x_raw", "ball_y_raw"]
    ].to_numpy(dtype=float)

    if not np.isfinite(
        estimated[relevant]
    ).all():
        raise ValueError(
            "Hay frames de juego sin estimación."
        )

    if not np.isnan(
        estimated[~relevant]
    ).all():
        raise ValueError(
            "Hay estimaciones fuera del juego."
        )

    if not np.array_equal(
        estimated[relevant & observed],
        raw[relevant & observed],
    ):
        raise ValueError(
            "Se alteraron observaciones originales."
        )

    if not np.array_equal(
        imputed,
        relevant & ~observed,
    ):
        raise ValueError(
            "Indicadores de imputación inconsistentes."
        )

    confidence = features[
        "ball_estimation_confidence"
    ].to_numpy(dtype=float)

    if not np.isnan(
        confidence[~relevant]
    ).all():
        raise ValueError(
            "Confianza fuera de juego debe ser NaN."
        )

    if not (
        np.isfinite(confidence[relevant]).all()
        and np.all(
            (confidence[relevant] >= 0)
            & (confidence[relevant] <= 1)
        )
    ):
        raise ValueError(
            "Puntuaciones de confianza inválidas."
        )


# ============================================================
# INFORME
# ============================================================

def write_report(features, strategy, context_weight):

    relevant = (
        features["ball_reconstruction_relevant"] == 1
    )

    observed = (
        features["ball_observed"] == 1
    )

    imputed = (
        features["ball_imputed"] == 1
    )

    method_counts = (
        features.loc[
            relevant,
            "ball_estimation_method",
        ].value_counts()
    )

    lines = [
        "# ETAPA 17 — Reconstrucción híbrida del balón (v2)",
        "",
        "## Configuración",
        f"- Estrategia: {strategy}",
        f"- Peso contextual: {context_weight:.2f}",
        f"- Hueco corto: ≤{SHORT_GAP} frames",
        f"- Hueco medio: ≤{MEDIUM_GAP} frames",
        "",
        "## Integridad",
        f"- Frames totales: {len(features):,}",
        f"- Frames de juego: {int(relevant.sum()):,}",
        f"- Observados en juego: "
        f"{int((relevant & observed).sum()):,}",
        f"- Reconstruidos en juego: "
        f"{int(imputed.sum()):,}",
        f"- Frames excluidos: {int((~relevant).sum()):,}",
        "",
        "## Métodos utilizados",
    ]

    for name, count in method_counts.items():
        lines.append(
            f"- {name}: {count:,} frames"
        )

    lines.extend([
        "",
        "## Interpretación",
        "- Las observaciones originales no se modifican.",
        "- Las estimaciones fuera de los tiempos de juego son NaN.",
        "- No se fuerza el balón dentro del campo.",
        "- La estrategia contextual utiliza jugadores próximos "
        "a una referencia temporal; no identifica al poseedor.",
        "- La confianza es heurística, no probabilística calibrada.",
        "- Los pesos todavía no están ajustados mediante validación.",
        "- Los huecos largos pueden tener errores importantes.",
        "- No se utiliza ground truth de posesión.",
        "- No se generan CSV intermedios.",
        "",
        "## Próxima etapa",
        "S18: ocultamiento de observaciones, comparación "
        "de estrategias y animaciones tácticas.",
    ])

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# EJECUCIÓN
# ============================================================

def main():

    print("Cargando tracking táctico...")
    tracking, _ = load_tracking()

    print("Reconstruyendo balón con S17 v2...")

    features = build_ball_reconstruction(
        tracking,
        strategy=DEFAULT_STRATEGY,
        context_weight=DEFAULT_CONTEXT_WEIGHT,
    )

    write_report(
        features,
        DEFAULT_STRATEGY,
        DEFAULT_CONTEXT_WEIGHT,
    )

    relevant = (
        features["ball_reconstruction_relevant"] == 1
    )

    observed = (
        features["ball_observed"] == 1
    )

    imputed = (
        features["ball_imputed"] == 1
    )

    print("\n========== ETAPA 17 v2 ==========")

    print(f"Frames: {len(features):,}")
    print(
        f"Observados en juego: "
        f"{int((relevant & observed).sum()):,}"
    )
    print(
        f"Reconstruidos en juego: "
        f"{int(imputed.sum()):,}"
    )
    print(
        f"Frames excluidos: "
        f"{int((~relevant).sum()):,}"
    )

    print(
        "Estimaciones fuera de juego:",
        int(
            features.loc[
                ~relevant,
                "ball_x_estimated",
            ].notna().sum()
        ),
    )

    print("\nMétodos:")
    print(
        features.loc[
            relevant,
            "ball_estimation_method",
        ].value_counts().to_string()
    )

    print("\nValidación estructural: True")
    print(f"Informe: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
