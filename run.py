
"""
RUN.PY — MOTOR DE POSESIÓN H/A/D/X · TH01–TH15 Y RITMO COLECTIVO
FC Barcelona / Athletic Club

Flujo:
1. Cargar V4 y S27.
2. Validar frames y periodos.
3. Construir evidencia de balon muerto.
4. Decidir D o balon activo.
5. Resolver H/A con memoria y confirmacion temporal.
6. Suavizar retrospectivamente fronteras confirmadas con apoyo de candidatos.
7. Guardar submission, trace y summary.

No usa ground truth.
No genera X dentro de los periodos de juego.
"""

import json
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import config
from dead_ball_detectors import detect as detect_dead_ball_types
from g8_episode_engine import episode_evidence
from g9_episode_refinement import refine_dead


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

V4_PATH = config.OUTPUT_DIR / "matrix_v4.csv"
S27_PATH = config.OUTPUT_DIR / "s27_football_signals.csv"

# ============================================================
# CONFIGURACIÓN DE REGLAS Y UMBRALES
# ============================================================

FPS = config.FPS

CONTROL_RADIUS_M = float(os.environ.get("RAM_G4_CONTROL_RADIUS_M", "2.5"))
CONTROL_ADVANTAGE_M = float(os.environ.get("RAM_G4_CONTROL_ADVANTAGE_M", "1.5"))

# Restricciones geométricas condicionadas a la fiabilidad del balón.
RAM_G5_IMPUTED_RADIUS_FACTOR = float(os.environ.get('RAM_G5_IMPUTED_RADIUS_FACTOR','1.0'))
RAM_G5_IMPUTED_ADVANTAGE_EXTRA = float(os.environ.get('RAM_G5_IMPUTED_ADVANTAGE_EXTRA','0.0'))
RAM_G5_OBSERVED_RADIUS_FACTOR = float(os.environ.get('RAM_G5_OBSERVED_RADIUS_FACTOR','1.0'))
RAM_G5_TH_BRIDGE_MAX_SECONDS = float(os.environ.get('RAM_G5_TH_BRIDGE_MAX_SECONDS','0.0'))
RAM_G5_TH_BRIDGE_MAX_LIVE_FRACTION = float(os.environ.get('RAM_G5_TH_BRIDGE_MAX_LIVE_FRACTION','0.05'))
if not (0.3 <= RAM_G5_IMPUTED_RADIUS_FACTOR <= 1.5 and 0.3 <= RAM_G5_OBSERVED_RADIUS_FACTOR <= 1.5
        and 0. <= RAM_G5_IMPUTED_ADVANTAGE_EXTRA <= 5.0 and 0. <= RAM_G5_TH_BRIDGE_MAX_SECONDS <= 30.0
        and 0. <= RAM_G5_TH_BRIDGE_MAX_LIVE_FRACTION <= 0.4):
    raise ValueError('G5: parametros geometricos/episodicos fuera de rango')

# Corroboración opcional de interrupciones mediante ritmo colectivo, desactivada.
RAM_G6_MOTION_ENABLED = int(os.environ.get("RAM_G6_MOTION_ENABLED", "0"))
RAM_G6_MOTION_COMPONENTS = int(os.environ.get("RAM_G6_MOTION_COMPONENTS", "2"))
RAM_G6_MOTION_MAX_SIDE_SECONDS = float(os.environ.get("RAM_G6_MOTION_MAX_SIDE_SECONDS", "8.0"))
RAM_G6_MOTION_SLOW_PROB = float(os.environ.get("RAM_G6_MOTION_SLOW_PROB", "0.70"))
RAM_G6_MOTION_HOLD_SECONDS = float(os.environ.get("RAM_G6_MOTION_HOLD_SECONDS", "0.5"))
RAM_G6_MOTION_MIN_COVERAGE = float(os.environ.get("RAM_G6_MOTION_MIN_COVERAGE", "0.45"))
if not (RAM_G6_MOTION_ENABLED in (0,1)
        and RAM_G6_MOTION_COMPONENTS in (2,3)
        and 0 <= RAM_G6_MOTION_MAX_SIDE_SECONDS <= 45
        and 0.50 <= RAM_G6_MOTION_SLOW_PROB <= 0.98
        and 0.20 <= RAM_G6_MOTION_MIN_COVERAGE <= 1.0
        and 0.12 <= RAM_G6_MOTION_HOLD_SECONDS <= 5.0):
    raise ValueError("G6: parámetros del régimen de movimiento fuera de rango")

# Reglas contextuales adicionales, desactivadas en la configuración principal.
# Contextos TH (incluidos gol/pausa colectiva) no son etiquetas de D ni
# hechos probados: exigen corroboración independiente y veto de juego real.
RAM_G7_CONTEXT_ENABLED = int(os.environ.get('RAM_G7_CONTEXT_ENABLED', '0'))
RAM_G7_CONTEXT_TYPES = tuple(x.strip() for x in os.environ.get('RAM_G7_CONTEXT_TYPES','TH08,TH09,TH11,TH12,TH15').split(',') if x.strip())
if not RAM_G7_CONTEXT_TYPES or any(x not in {'TH08','TH09','TH11','TH12','TH15'} for x in RAM_G7_CONTEXT_TYPES):
    raise ValueError('G7_CONTEXT_TYPES: solo TH08,TH09,TH11,TH12,TH15')
RAM_G7_CONTEXT_MAX_SECONDS = float(os.environ.get('RAM_G7_CONTEXT_MAX_SECONDS', '30'))
RAM_G7_CONTEXT_HOLD_SECONDS = float(os.environ.get('RAM_G7_CONTEXT_HOLD_SECONDS', '1.2'))
RAM_G7_CONTEXT_NEAR_SECONDS = float(os.environ.get('RAM_G7_CONTEXT_NEAR_SECONDS', '12'))
RAM_G7_MOTION_CUTOFF_MPS = float(os.environ.get('RAM_G7_MOTION_CUTOFF_MPS', '1.8'))
RAM_G7_MOTION_SLOW_PROB = float(os.environ.get('RAM_G7_MOTION_SLOW_PROB', '.63'))
RAM_G7_MOTION_COMPONENTS = int(os.environ.get('RAM_G7_MOTION_COMPONENTS', '3'))
RAM_G7_MOTION_SEVERITY_ENABLED = int(os.environ.get('RAM_G7_MOTION_SEVERITY_ENABLED', '0'))
RAM_G7_JOINT_ENABLED = int(os.environ.get('RAM_G7_JOINT_ENABLED', '0'))
RAM_G7_JOINT_INERTIA_HA = float(os.environ.get('RAM_G7_JOINT_INERTIA_HA', '24'))
RAM_G7_JOINT_ENTER_D = float(os.environ.get('RAM_G7_JOINT_ENTER_D', '11'))
RAM_G7_JOINT_EXIT_D = float(os.environ.get('RAM_G7_JOINT_EXIT_D', '12'))
RAM_G7_JOINT_BASE_PRIOR = float(os.environ.get('RAM_G7_JOINT_BASE_PRIOR', '2.0'))
RAM_G7_JOINT_CONTEXT_REWARD = float(os.environ.get('RAM_G7_JOINT_CONTEXT_REWARD', '4.0'))
RAM_G7_JOINT_UNVERIFIED_EXIT_EXTRA = float(os.environ.get('RAM_G7_JOINT_UNVERIFIED_EXIT_EXTRA','0.0'))
RAM_G7_TOUCH_DISCOUNT = float(os.environ.get('RAM_G7_TOUCH_DISCOUNT', '0.0'))
if not (RAM_G7_CONTEXT_ENABLED in (0,1) and RAM_G7_MOTION_SEVERITY_ENABLED in (0,1)
        and RAM_G7_JOINT_ENABLED in (0,1) and RAM_G7_MOTION_COMPONENTS in (2,3)
        and 0<=RAM_G7_CONTEXT_MAX_SECONDS<=90 and .4<=RAM_G7_CONTEXT_HOLD_SECONDS<=5
        and 2<=RAM_G7_CONTEXT_NEAR_SECONDS<=35 and .7<=RAM_G7_MOTION_CUTOFF_MPS<=3
        and .45<=RAM_G7_MOTION_SLOW_PROB<=.98 and 0<=RAM_G7_TOUCH_DISCOUNT<=20
        and min(RAM_G7_JOINT_INERTIA_HA,RAM_G7_JOINT_ENTER_D,RAM_G7_JOINT_EXIT_D,
                RAM_G7_JOINT_BASE_PRIOR,RAM_G7_JOINT_CONTEXT_REWARD,
                RAM_G7_JOINT_UNVERIFIED_EXIT_EXTRA)>=0):
    raise ValueError('G7: parámetros fuera de contrato')

# Integración de evidencia física, temporal y de interrupciones antes de decidir posesión.
RAM_G8_EPISODES_ENABLED = int(os.environ.get('RAM_G8_EPISODES_ENABLED','1'))
RAM_G8_S31_ENABLED = int(os.environ.get('RAM_G8_S31_ENABLED','1'))
RAM_G8_TH_CONTEXT_ENABLED = int(os.environ.get('RAM_G8_TH_CONTEXT_ENABLED','1'))
RAM_G8_TYPES = tuple(t.strip().upper() for t in os.environ.get('RAM_G8_TYPES',
    'COOLING_BREAK').split(',') if t.strip())
RAM_G8_CONTEXT_THRESHOLD = float(os.environ.get('RAM_G8_CONTEXT_THRESHOLD','0.55'))
RAM_G8_PHYSICAL_THRESHOLD = float(os.environ.get('RAM_G8_PHYSICAL_THRESHOLD','0.40'))
RAM_G8_SEED_SECONDS = float(os.environ.get('RAM_G8_SEED_SECONDS','1.2'))
RAM_G8_PATIENCE_SECONDS = float(os.environ.get('RAM_G8_PATIENCE_SECONDS','8.0'))
RAM_G8_MAX_EPISODE_SECONDS = float(os.environ.get('RAM_G8_MAX_EPISODE_SECONDS','120.0'))
RAM_G8_RESTART_SECONDS = float(os.environ.get('RAM_G8_RESTART_SECONDS','0.7'))
RAM_G8_GMM_ENABLED = int(os.environ.get('RAM_G8_GMM_ENABLED','0'))
RAM_G8_GMM_COMPONENTS = int(os.environ.get('RAM_G8_GMM_COMPONENTS','2'))
RAM_G8_GMM_WEIGHT = float(os.environ.get('RAM_G8_GMM_WEIGHT','0.25'))
RAM_G8_SINGLE_HA_FINAL = int(os.environ.get('RAM_G8_SINGLE_HA_FINAL','0'))
# Reglas de corroboración temporal para los contextos S31.
RAM_G9_HOLD_FACTOR = float(os.environ.get('RAM_G9_HOLD_FACTOR','0.70'))
RAM_G9_ENTRY_FRACTION = float(os.environ.get('RAM_G9_ENTRY_FRACTION','0.68'))
RAM_G9_PHYSICAL_WINDOW_SECONDS = float(os.environ.get('RAM_G9_PHYSICAL_WINDOW_SECONDS','0.8'))
RAM_G9_RESTART_THRESHOLD = float(os.environ.get('RAM_G9_RESTART_THRESHOLD','0.75'))
RAM_G9_TH_NEAR_SECONDS = float(os.environ.get('RAM_G9_TH_NEAR_SECONDS','8.0'))
RAM_G9_MIN_SEED_SECONDS = float(os.environ.get('RAM_G9_MIN_SEED_SECONDS','0.6'))
RAM_G9_REFINE_ENABLED = int(os.environ.get('RAM_G9_REFINE_ENABLED','0'))
RAM_G9_REFINE_TYPES = tuple(t.strip().upper() for t in os.environ.get('RAM_G9_REFINE_TYPES','COOLING_BREAK').split(',') if t.strip())
RAM_G9_REFINE_CONTEXT = float(os.environ.get('RAM_G9_REFINE_CONTEXT','0.44'))
RAM_G9_REFINE_PHYSICS = float(os.environ.get('RAM_G9_REFINE_PHYSICS','0.35'))
RAM_G9_REFINE_EDGE_SECONDS = float(os.environ.get('RAM_G9_REFINE_EDGE_SECONDS','4.0'))
RAM_G9_REFINE_BRIDGE_SECONDS = float(os.environ.get('RAM_G9_REFINE_BRIDGE_SECONDS','8.0'))
RAM_G9_REFINE_PATIENCE_SECONDS = float(os.environ.get('RAM_G9_REFINE_PATIENCE_SECONDS','0.6'))
RAM_G9_REFINE_RESTART_SECONDS = float(os.environ.get('RAM_G9_REFINE_RESTART_SECONDS','0.6'))

G8_ALLOWED_TYPES = {'THROW_IN','CORNER','GOAL_KICK','FREE_KICK','PENALTY',
   'KICK_OFF','OFFSIDE','GOAL','CARD_INTERVENTION','VAR_REVIEW','INJURY',
   'COOLING_BREAK','SUBSTITUTION','DROPPED_BALL','UNKNOWN_INTERRUPTION'}
if not (RAM_G8_EPISODES_ENABLED in (0,1) and RAM_G8_S31_ENABLED in (0,1)
        and RAM_G8_TH_CONTEXT_ENABLED in (0,1) and RAM_G8_SINGLE_HA_FINAL in (0,1)
        and RAM_G8_GMM_ENABLED in (0,1) and RAM_G8_GMM_COMPONENTS in (2,3)
        and RAM_G8_TYPES and set(RAM_G8_TYPES)<=G8_ALLOWED_TYPES
        and .35<=RAM_G8_CONTEXT_THRESHOLD<=.95
        and .2<=RAM_G8_PHYSICAL_THRESHOLD<=.95
        and .3<=RAM_G8_SEED_SECONDS<=6.0
        and .2<=RAM_G8_RESTART_SECONDS<=3.0
        and .4<=RAM_G8_PATIENCE_SECONDS<=90
        and 3.<=RAM_G8_MAX_EPISODE_SECONDS<=240
        and .0<=RAM_G8_GMM_WEIGHT<=1.):
    raise ValueError('G8: parámetros fuera de contrato')
if not (.45<=RAM_G9_HOLD_FACTOR<=1.15 and .35<=RAM_G9_ENTRY_FRACTION<=.97
        and .2<=RAM_G9_PHYSICAL_WINDOW_SECONDS<=5.0 and .5<=RAM_G9_RESTART_THRESHOLD<=.99
        and 1<=RAM_G9_TH_NEAR_SECONDS<=20 and .2<=RAM_G9_MIN_SEED_SECONDS<=1.5
        and RAM_G9_REFINE_ENABLED in (0,1) and set(RAM_G9_REFINE_TYPES)<=G8_ALLOWED_TYPES
        and .35<=RAM_G9_REFINE_CONTEXT<=.85 and .2<=RAM_G9_REFINE_PHYSICS<=.85
        and 0<=RAM_G9_REFINE_EDGE_SECONDS<=20 and 0<=RAM_G9_REFINE_BRIDGE_SECONDS<=30
        and 0<=RAM_G9_REFINE_PATIENCE_SECONDS<=3 and .2<=RAM_G9_REFINE_RESTART_SECONDS<=3):
    raise ValueError('G9: parámetros fuera de contrato')


# V4: evidencia sostenida en ventanas, no un simple contador.
SWITCH_SHORT_WINDOW = int(os.environ.get("RAM_SWITCH_SHORT_WINDOW", "10"))  # frames
SWITCH_LONG_WINDOW = int(os.environ.get("RAM_SWITCH_LONG_WINDOW", "25"))
if not (3 <= SWITCH_SHORT_WINDOW <= SWITCH_LONG_WINDOW <= 150):
    raise ValueError("Ventanas de switch invalidas")
SWITCH_MIN_EVIDENCE_FRAMES = int(os.environ.get("RAM_SWITCH_MIN_EVIDENCE", "12"))
SWITCH_IMPUTED_MIN_FRAMES = int(os.environ.get("RAM_SWITCH_IMPUTED_MIN", "20"))
SWITCH_SCORE_THRESHOLD = float(os.environ.get("RAM_SWITCH_SCORE", "10.0"))
SWITCH_IMPUTED_SCORE_THRESHOLD = float(os.environ.get("RAM_SWITCH_IMPUTED_SCORE", "12.0"))
SWITCH_SHORT_SUPPORT = float(os.environ.get("RAM_SWITCH_SHORT_SUPPORT", "0.75"))
SWITCH_LONG_SUPPORT = float(os.environ.get("RAM_SWITCH_LONG_SUPPORT", "0.40"))
SWITCH_MAX_SCORE = 30.0
SWITCH_OWNER_VETO_FRAMES = int(os.environ.get("RAM_SWITCH_OWNER_VETO", "3"))
SWITCH_COOLDOWN_FRAMES = int(os.environ.get("RAM_SWITCH_COOLDOWN", "12"))
SWITCH_BACKTRACK_SUPPORT = float(os.environ.get("RAM_SWITCH_BACKTRACK_SUPPORT", "0.65"))
OBSERVED_CONFIRM_FRAMES = SWITCH_MIN_EVIDENCE_FRAMES
IMPUTED_CONFIRM_FRAMES = SWITCH_IMPUTED_MIN_FRAMES

# Tolerancia a interrupciones breves del candidato rival.
MAX_CANDIDATE_GAP_FRAMES = int(os.environ.get("RAM_SWITCH_GAP", "5"))

# Suavizado retrospectivo limitado.
MAX_BACKTRACK_FRAMES = int(os.environ.get("RAM_SWITCH_BACKTRACK_FRAMES", "20"))

# Evidencia de balon muerto.
OUTSIDE_MARGIN_NORM = 0.005
OUTSIDE_CONFIRM_FRAMES = 5

LOW_BALL_SPEED_MPS = 0.8
LOW_TEAM_SPEED_MPS = 1.5
STOP_CONFIRM_FRAMES = 50

# Evita tratar una interpolacion muy larga como observacion.
LONG_IMPUTATION_FRAMES = 50

# No usamos la ausencia de geometria para declarar D.
# Tampoco usamos SET PIECES por si solo.


# V5: las ventanas son escalas de evidencia, no tiempos fijos por tipo de saque.
WINDOW_FAST = 5
WINDOW_SHORT = 10
WINDOW_MEDIUM = 25
WINDOW_LONG = 50
WINDOW_CONTEXT = 125
STOP_ACC_THRESHOLD = 11.0
RESTART_ACC_THRESHOLD = 7.0
MAX_STOP_BACKTRACK = 50
MAX_RESTART_BACKTRACK = 25

# ============================================================
# UTILIDADES
# ============================================================

def numeric(df, column, default=np.nan):
    if column not in df.columns:
        return np.full(len(df), default, dtype=float)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).to_numpy(dtype=float)


def bool_column(df, column):
    return numeric(df, column, 0) > 0


def validate_frames(df, name):
    n = config.N_VIDEO_FRAMES

    if len(df) != n:
        raise ValueError(
            f"{name}: se esperaban {n} frames; "
            f"se encontraron {len(df)}"
        )

    if "frame" not in df.columns:
        raise ValueError(f"{name}: falta frame")

    frames = pd.to_numeric(
        df["frame"],
        errors="raise"
    ).to_numpy()

    if not np.array_equal(frames, np.arange(n)):
        raise ValueError(
            f"{name}: frames desalineados"
        )


def consecutive_confirm(mask, minimum):
    """
    Marca frames a partir del momento en que una condicion
    lleva 'minimum' frames consecutivos cumpliendose.
    """
    output = np.zeros(len(mask), dtype=bool)
    count = 0

    for i, value in enumerate(mask):
        count = count + 1 if value else 0

        if count >= minimum:
            output[i] = True

    return output


def get_tag(v4, name):
    col = f"smart_tagging_code_{name}_count"
    return bool_column(v4, col)


def get_team_speed(v4, team):
    # Nombre real de S07, confirmado en los scripts del proyecto.
    return numeric(v4, f"{team}_speed_median")


def get_ball_position(v4):
    # S21 publica las coordenadas reconstruidas normalizadas.
    return numeric(v4, "ball_x_normalized"), numeric(v4, "ball_y_normalized")


def runs(mask):
    """Tramos inclusivos consecutivos de True."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def stable_rolling(mask, length):
    # Porcentaje de evidencia en una ventana causal.
    return pd.Series(mask.astype(float)).rolling(
        length, min_periods=length
    ).mean().fillna(0).to_numpy()


def make_signal_audit(v4, s27, relevant):
    """Inventario de entradas efectivas; no consulta GT."""
    groups = {
        "ball_source": (v4, ["ball_observed", "ball_imputed", "ball_gap_length", "ball_estimation_confidence"]),
        "ball_geometry": (v4, ["ball_x_normalized", "ball_y_normalized", "ball_outside_normalized"]),
        "ball_kinematics": (v4, ["ball_speed_mps", "ball_displacement_m"]),
        "team_motion": (v4, ["Barcelona_speed_median", "Atletic_Club_speed_median", "Barcelona_speed_coverage", "Atletic_Club_speed_coverage"]),
        "collective_geometry": (v4, ["Barcelona_centroid_x_m", "Barcelona_centroid_y_m", "Barcelona_spread_x_m", "Barcelona_spread_y_m", "Barcelona_valid_position_count", "Atletic_Club_centroid_x_m", "Atletic_Club_centroid_y_m", "Atletic_Club_spread_x_m", "Atletic_Club_spread_y_m", "Atletic_Club_valid_position_count"]),
        "set_pieces": (v4, ["smart_tagging_code_set_pieces_count"]),
        "occupation_S09": (v4, [f"{t}_zone_X{x}_Y{y}_ratio" for t in ("Barcelona", "Atletic_Club") for x in (1,2,3) for y in (1,2,3)]),
        "control_geometry": (s27, ["home_distance_m", "away_distance_m", "distance_margin_home_m", "geometry_comparable"]),
    }
    rows = []
    for family, (df, names) in groups.items():
        for name in names:
            exists = name in df.columns
            values = pd.to_numeric(df[name], errors="coerce") if exists else None
            rows.append({
                "family": family, "column": name, "exists": int(exists),
                "game_finite_frames": int(np.isfinite(values.to_numpy(dtype=float))[relevant].sum()) if exists else 0,
                "game_frames": int(relevant.sum()),
                "min": float(values[relevant].min()) if exists and values[relevant].notna().any() else None,
                "median": float(values[relevant].median()) if exists and values[relevant].notna().any() else None,
                "max": float(values[relevant].max()) if exists and values[relevant].notna().any() else None,
            })
    audit = pd.DataFrame(rows)
    required = ["ball_observed", "ball_imputed", "ball_gap_length", "ball_x_normalized", "ball_y_normalized", "ball_speed_mps", "Barcelona_speed_median", "Atletic_Club_speed_median"]
    required += [f"{team}_zone_X{x}_Y{y}_ratio" for team in ("Barcelona", "Atletic_Club") for x in (1,2,3) for y in (1,2,3)]
    required += [
        f"{team}_{field}"
        for team in ("Barcelona", "Atletic_Club")
        for field in ("centroid_x_m", "centroid_y_m", "spread_x_m", "spread_y_m", "valid_position_count")
    ]
    missing = [x for x in required if x not in v4.columns]
    if missing:
        raise ValueError("Faltan señales críticas para V5 (revisar matriz V4 / S06): " + ", ".join(missing))
    return audit


# ============================================================
# 1. CARGA
# ============================================================

def load_inputs():
    print("[1] Cargando V4 y S27...")

    if not V4_PATH.exists():
        raise FileNotFoundError(V4_PATH)

    if not S27_PATH.exists():
        raise FileNotFoundError(S27_PATH)

    v4 = pd.read_csv(V4_PATH, low_memory=False)
    s27 = pd.read_csv(S27_PATH, low_memory=False)

    validate_frames(v4, "V4")
    validate_frames(s27, "S27")

    required = [
        "relevant",
        "ball_source",
        "geometry_comparable",
        "home_distance_m",
        "away_distance_m",
        "distance_margin_home_m",
    ]

    missing = [
        col for col in required
        if col not in s27.columns
    ]

    if missing:
        raise ValueError(
            f"S27: faltan columnas {missing}"
        )

    return v4, s27


# ============================================================
# 2. EVIDENCIA DE BALON MUERTO
# ============================================================

def trailing_mean(arr, window, min_periods=None):
    """Media causal; no mezcla periodos porque se usa solo con máscara validada."""
    # copy=True: algunos entornos de pandas/NumPy devuelven una vista
    # de solo lectura. Estas ventanas se enmascaran in-place mas abajo.
    return pd.Series(np.asarray(arr, dtype=float)).rolling(
        window, min_periods=(min_periods or window)
    ).mean().to_numpy(dtype=float, copy=True)


def period_delta(values, lag, relevant):
    """Cambio entre frames separados, sin cruzar frames no evaluables."""
    n = len(values)
    output = np.full(n, np.nan)
    if lag >= n:
        return output
    continuous = trailing_mean(relevant.astype(float), lag + 1, lag + 1) >= 1
    ok = continuous[lag:] & np.isfinite(values[lag:]) & np.isfinite(values[:-lag])
    segment = output[lag:]
    segment[ok] = values[lag:][ok] - values[:-lag][ok]
    return output


def collective_evidence(v4, relevant):
    """Familia colectiva a partir de S06/S07. No infiere 'cooling break' ni eventos arbitrales.

    Precauciones: cobertura de tracking, cambio entre centroides y bandas,
    corroboración de ambos equipos; ninguna señal aislada crea D.
    """
    n = len(v4)
    teams = ("Barcelona", "Atletic_Club")
    coverage = []
    dy = []
    centery = []
    compact = []
    speed = []
    for team in teams:
        x = numeric(v4, f"{team}_centroid_x_m")
        y = numeric(v4, f"{team}_centroid_y_m")
        sx = numeric(v4, f"{team}_spread_x_m")
        sy = numeric(v4, f"{team}_spread_y_m")
        valid = numeric(v4, f"{team}_valid_position_count", 0)
        cov = valid / (22.0 if team == "Atletic_Club" else 24.0)
        coverage.append(np.clip(cov, 0, 1))
        dy.append(period_delta(y, 50, relevant))
        centery.append(y)
        compact.append(np.isfinite(sx) & np.isfinite(sy) & (sx < 17) & (sy < 13))
        speed.append(numeric(v4, f"{team}_speed_median"))
    valid = relevant & (coverage[0] >= .40) & (coverage[1] >= .40)
    # Los centroides pueden desplazarse por visibilidad; por eso exigimos
    # cobertura y movimiento paralelo en ambos equipos, no una sola trayectoria.
    same_way = (np.sign(dy[0]) == np.sign(dy[1]))
    both_shift = valid & same_way & (np.abs(dy[0]) >= 2.5) & (np.abs(dy[1]) >= 2.5)
    toward_touchline = both_shift & (
        ((dy[0] < 0) & (dy[1] < 0) & (centery[0] < 28) & (centery[1] < 28))
        | ((dy[0] > 0) & (dy[1] > 0) & (centery[0] > 40) & (centery[1] > 40))
    )
    stationary = valid & np.isfinite(speed[0]) & np.isfinite(speed[1]) & (speed[0] < 1.1) & (speed[1] < 1.1)
    compact_both = valid & compact[0] & compact[1]
    # No dar evidencia fuerte de pausa por 'hacia banda' solamente:
    # también ocurre en una transición de ataque.
    collective_stop = valid & (
        (toward_touchline & (stationary | compact_both))
        | (stationary & compact_both)
    )
    collective_live = valid & np.isfinite(speed[0]) & np.isfinite(speed[1]) & (speed[0] >= 2.2) & (speed[1] >= 2.2)
    return {
        "collective_valid": valid,
        "collective_shift": both_shift,
        "collective_touchline": toward_touchline,
        "collective_stop": collective_stop,
        "collective_live": collective_live,
        "collective_compact": compact_both,
    }


def segment_rolling(values, relevant, window, center=False, minimum=None):
    """Agregado independiente en cada periodo; nunca atraviesa el descanso.

    Las ventanas centradas se usan solo en la decisión retrospectiva offline;
    no se presentan como detecciones causales en tiempo real.
    """
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v), dtype=float)
    for a, b in runs(relevant):
        a0 = pd.Series(v[a:b+1]).rolling(
            window, center=center,
            min_periods=(minimum if minimum is not None else max(1, window // 2))
        ).mean().fillna(0).to_numpy(dtype=float, copy=True)
        out[a:b+1] = a0
    return out


def consecutive_true(mask, count):
    """Necesita persistencia real; una señal aislada no genera un ancla."""
    return segment_rolling(mask, np.ones(len(mask), dtype=bool), count, False, count) >= 1.0


def occupation_evidence(v4, relevant, collective_valid):
    """Contexto espacial de equipos usando SOLO agregados reales de S09.

    Sin rastreo de identidad no se infiere 'jugador se dirige al córner'.
    Un patrón de concentración o lateral es contexto, nunca una falta/saque.
    """
    n = len(v4)
    teams = ('Barcelona', 'Atletic_Club')
    lateral = []
    concentration = []
    change = []
    for team in teams:
        y1 = np.stack([numeric(v4, f'{team}_zone_X{x}_Y1_count', 0) for x in (1,2,3)], axis=1).sum(axis=1)
        y3 = np.stack([numeric(v4, f'{team}_zone_X{x}_Y3_count', 0) for x in (1,2,3)], axis=1).sum(axis=1)
        inside = numeric(v4, f'{team}_inside_position_count', 0)
        frac = np.divide(np.maximum(y1, y3), inside, out=np.zeros(n), where=inside>=5)
        # max concentración en alguna de las 9 zonas, requiere cobertura suficiente
        cells = [numeric(v4, f'{team}_zone_X{x}_Y{y}_ratio') for x in (1,2,3) for y in (1,2,3)]
        mat = np.column_stack(cells)
        mat[~np.isfinite(mat)] = 0
        lateral.append(frac)
        concentration.append(np.max(mat, axis=1))
        past = np.full(n, 0.)
        past[50:] = np.abs(frac[50:] - frac[:-50])
        change.append(past)
    lateral_both = collective_valid & (lateral[0] >= .52) & (lateral[1] >= .52)
    compact_zone = collective_valid & (concentration[0] >= .46) & (concentration[1] >= .46)
    reorganization = collective_valid & ((change[0] >= .18) | (change[1] >= .18))
    # Una reconfiguración espacial aislada NO implica balón muerto.
    return {'zone_touchline_both': lateral_both & relevant,
            'zone_compact_both': compact_zone & relevant,
            'zone_reorganization': reorganization & relevant}


def tactical_context_evidence(v4, relevant, low_teams, ball_reliability, set_piece):
    """Contexto espacial por lineas geometricas, con calidad y memoria.

    S09 proporciona X1/X2/X3 (tercios del TERRENO, no demarcaciones ni
    direccion de ataque); S07 moving_ratio y speed_coverage. Los slots NO
    son identidades. Evitamos concluir que un track es defensa o delantero.
    Ningun indicador aqui significa por si solo falta, corner o juego muerto.
    """
    n = len(v4)
    teams = ('Barcelona', 'Atletic_Club')
    occ, cover, moves, speeds = [], [], [], []
    for team in teams:
        # Agrupacion por tercio longitudinal, ignorando tracks no visibles.
        total = numeric(v4, f'{team}_inside_position_count', 0)
        xzones = []
        for x in (1, 2, 3):
            counts = np.zeros(n, dtype=float)
            for y in (1, 2, 3):
                counts += numeric(v4, f'{team}_zone_X{x}_Y{y}_count', 0)
            xzones.append(np.divide(counts, total,
                out=np.full(n, np.nan), where=total >= 6))
        occ.append(np.stack(xzones, axis=1))
        cover.append((total >= 6) & (numeric(v4, f'{team}_speed_coverage', 0) >= .30))
        moves.append(numeric(v4, f'{team}_moving_ratio'))
        speeds.append(numeric(v4, f'{team}_speed_median'))
    valid = relevant & cover[0] & cover[1]
    def zone_change(z, lag):
        # Comparar en cada segmento; prohibido cruzar descanso.
        delta = np.zeros(n, dtype=float)
        for k in range(3):
            diff = period_delta(z[:,k], lag, relevant)
            delta += np.nan_to_num(np.abs(diff), nan=0.)
        return np.clip(.5 * delta, 0, 1)
    d25 = [zone_change(occ[k], 25) for k in range(2)]
    d50 = [zone_change(occ[k], 50) for k in range(2)]
    # Distancia de distribuciones de equipos: sirve para saber si hay
    # congestion en una region comun, NO si esa zona es ofensiva.
    overlap = np.sum(np.minimum(np.nan_to_num(occ[0]), np.nan_to_num(occ[1])), axis=1)
    mean_change25 = .5 * (d25[0] + d25[1])
    mean_change50 = .5 * (d50[0] + d50[1])
    moving_both = np.minimum(np.nan_to_num(moves[0], nan=0.), np.nan_to_num(moves[1], nan=0.))
    moving_either = np.maximum(np.nan_to_num(moves[0], nan=0.), np.nan_to_num(moves[1], nan=0.))
    # Evitar confundir visibilidad con desaceleracion.
    moving_valid = valid & np.isfinite(moves[0]) & np.isfinite(moves[1])
    moving_mean = .5 * (np.nan_to_num(moves[0], nan=0.) + np.nan_to_num(moves[1], nan=0.))
    preceding = segment_rolling(moving_mean, relevant, 50, False, 20)
    fall = (preceding - moving_mean) >= .20
    slow_now = moving_valid & (moving_either <= .32) & low_teams
    # Bajas velocidades con distribuciones estables son contexto de pausa,
    # pero tambien pueden ocurrir en juego lento: se exige corroboracion.
    shape_stable = valid & (mean_change25 <= .12) & (mean_change50 <= .20)
    long_still = (segment_rolling(slow_now & shape_stable, relevant, 50, False, 25) >= .70)
    no_ball_support = ball_reliability < .30
    collective_pause = valid & long_still & (set_piece | fall | no_ball_support)
    congestion_idle = valid & shape_stable & (overlap >= .46) & slow_now
    reorganization = valid & (mean_change50 >= .22) & (mean_change25 <= .10)
    # Movimiento vivo: al menos un equipo se desplaza y hay cambio
    # territorial verificable; NO basta para atribuir la posesion.
    tactical_flow = moving_valid & (moving_either >= .48) & (
        (mean_change25 >= .09) | (moving_both >= .38)
    ) & ~long_still
    # Distingue falta de jugadores observados de falta de movimiento.
    reliability = valid.astype(float) * np.clip(
        .5 * (numeric(v4, 'Barcelona_speed_coverage', 0)
             + numeric(v4, 'Atletic_Club_speed_coverage', 0)), 0, 1)
    return {
        'tactical_valid': valid,
        'tactical_reliability': reliability,
        'tactical_overlap': overlap,
        'tactical_change_25': mean_change25,
        'tactical_change_50': mean_change50,
        'tactical_moving_mean': moving_mean,
        'tactical_slowdown': valid & fall,
        'tactical_shape_stable': shape_stable,
        'tactical_collective_pause': collective_pause,
        'tactical_congestion_idle': congestion_idle,
        'tactical_reorganization': reorganization,
        'tactical_flow': tactical_flow,
    }


def infer_activity_from_anchors(relevant, stop_anchor, play_anchor,
                                stop_strength, play_strength,
                                stop_context, play_context, ball_observed,
                                min_dead_frames=10):
    """Suavizador bidireccional con costes de transición y anclas fuertes.

    Detección offline. Un ancla de juego vivo tiene prioridad sobre
    expansión de D; un ancla de D requiere corroboración previa.
    La transición se calcula por episodio y NO usa referencia humana.
    """
    n = len(relevant)
    dead = np.zeros(n, dtype=bool)
    anchor_stop_count = int(np.sum(stop_anchor & relevant))
    anchor_play_count = int(np.sum(play_anchor & relevant))
    # Coste positivo favorece juego activo; negativo favorece D.
    # Evidencia ambigua = coste ~0: no inventa cambios sin apoyo.
    obs_bias = np.where(ball_observed, float(os.environ.get("RAM_ACTIVITY_OBS_BIAS", "0.035")), 0.)
    emission = (np.clip(play_strength, 0, 5) - np.clip(stop_strength, 0, 5)) * float(os.environ.get("RAM_ACTIVITY_EMISSION_SCALE", "0.55")) + obs_bias
    emission += np.where(play_context, float(os.environ.get("RAM_ACTIVITY_PLAY_CONTEXT", "0.18")), 0.)
    emission -= np.where(stop_context, float(os.environ.get("RAM_ACTIVITY_STOP_CONTEXT", "0.18")), 0.)
    emission = np.clip(emission, -3.5, 3.5)
    # El cambio de estado paga un coste, independientemente de la duración.
    # Una ancla observada es una restricción física/temporal, no una etiqueta GT.
    transition_cost = float(os.environ.get("RAM_ACTIVITY_TRANSITION_COST", "5.5"))
    for a,b in runs(relevant):
        m = b-a+1
        e = emission[a:b+1]
        st = stop_anchor[a:b+1]
        pl = play_anchor[a:b+1] & ~st
        # Viterbi binario lineal. LIVE=0, DEAD=1.
        costs = np.zeros((m,2),dtype=float)
        back = np.zeros((m,2),dtype=np.int8)
        costs[0] = [0., 3.0]
        costs[0,1] += e[0]
        costs[0,0] -= e[0]
        if st[0]: costs[0,0] += 1e5
        if pl[0]: costs[0,1] += 1e5
        for j in range(1,m):
            prev=costs[j-1]
            # El inicio y la salida de D están sujetos a la misma fricción,
            # pero la evidencia por frame puede vencerla si persiste.
            live_from_live=prev[0]
            live_from_dead=prev[1]+transition_cost
            dead_from_dead=prev[1]
            dead_from_live=prev[0]+transition_cost
            to_live=int(live_from_dead < live_from_live)
            to_dead=int(dead_from_dead <= dead_from_live)  # 1=desde DEAD
            costs[j,0]=(live_from_dead if to_live else live_from_live)-e[j]
            costs[j,1]=(dead_from_dead if to_dead else dead_from_live)+e[j]
            back[j,0]=to_live
            back[j,1]=to_dead
            if st[j]: costs[j,0]+=1e5
            if pl[j]: costs[j,1]+=1e5
        states=np.zeros(m,dtype=np.int8)
        states[-1]=int(costs[-1,1] < costs[-1,0])
        for j in range(m-1,0,-1):
            states[j-1]=back[j, states[j]]
        # Evita anclas D muy cortas, con excepción de trayectorias observadas
        # explícitamente exteriores al campo.
        for left,right in runs(states==1):
            if (right-left+1)<min_dead_frames and not st[left:right+1].any():
                states[left:right+1]=0
        dead[a:b+1] = states==1
    # No extender D a partir de una sola ancla a tramos enormes sin evidencia.
    # Si no hay soporte nuevo ni observaciones útiles durante 4 s, mantener
    # el estado previo hasta una nueva prueba, pero señalarlo como incierto.
    return dead, anchor_stop_count, anchor_play_count, emission


def build_dead_ball_evidence(v4, s27, relevant):
    """V5.1: fusión por regímenes + anclas de certeza + retroexpansión.

    Usa exclusivamente variables V4/S27, sin mirar etiquetas de referencia.
    La corroboración colectiva utiliza S06/S07/S09 existentes.
    """
    n=len(v4)
    obs=bool_column(v4,'ball_observed') & relevant
    imp=bool_column(v4,'ball_imputed') & relevant
    gap=numeric(v4,'ball_gap_length',0)
    long_gap=imp & (gap>=LONG_IMPUTATION_FRAMES)
    conf=numeric(v4,'ball_estimation_confidence',0)
    reliability=np.zeros(n)
    reliability[obs]=1.0
    reliability[imp]=np.clip(.8*np.exp(-np.maximum(gap[imp],0)/125.),.08,.8)
    reliability[imp]*=np.where(np.isfinite(conf[imp]),np.clip(.5+.5*conf[imp],.3,1.),.75)
    x,y=get_ball_position(v4)
    valid_xy=np.isfinite(x)&np.isfinite(y)
    out=relevant & valid_xy & ((x<-.005)|(x>1.005)|(y<-.005)|(y>1.005))
    observed_out=out&obs
    observed_cross=observed_out & (segment_rolling(observed_out,relevant,5,False,3)>=.6)
    near_edge=relevant&valid_xy&((x<=.025)|(x>=.975)|(y<=.025)|(y>=.975))
    near_corner=relevant&valid_xy&((x<=.045)|(x>=.955))&((y<=.065)|(y>=.935))
    speed=numeric(v4,'ball_speed_mps')
    valid_speed=relevant&np.isfinite(speed)&(speed>=0)&(speed<=45)
    reliable_slow=valid_speed&(speed<=.8)&(reliability>=.45)&~long_gap
    reliable_fast=valid_speed&(speed>=3)&(reliability>=.55)&~long_gap
    displacement=numeric(v4,'ball_displacement_m')
    moved=relevant&np.isfinite(displacement)&(displacement>=.2)&(reliability>=.55)&~long_gap
    hs=get_team_speed(v4,'Barcelona'); ass=get_team_speed(v4,'Atletic_Club')
    hc=numeric(v4,'Barcelona_speed_coverage',0)
    ac=numeric(v4,'Atletic_Club_speed_coverage',0)
    speeds_valid=relevant&(hc>=.35)&(ac>=.35)&np.isfinite(hs)&np.isfinite(ass)
    low_teams=speeds_valid&(hs<=1.5)&(ass<=1.5)
    very_low=speeds_valid&(hs<=1.0)&(ass<=1.0)
    fast_teams=speeds_valid&(hs>=2)&(ass>=2)
    set_piece=get_tag(v4,'set_pieces')&relevant
    geom=bool_column(s27,'geometry_comparable')&relevant
    hd=numeric(s27,'home_distance_m'); ad=numeric(s27,'away_distance_m')
    near=geom&np.isfinite(hd)&np.isfinite(ad)&(np.minimum(hd,ad)<=2)
    coll=collective_evidence(v4,relevant)
    zon=occupation_evidence(v4,relevant,coll['collective_valid'])
    tact=tactical_context_evidence(v4,relevant,low_teams,reliability,set_piece)
    # Familias independientes: no sumar arbitrariamente tres proxies de velocidad.
    # Si el balón está muy imputado el peso cambia hacia el movimiento colectivo.
    ball_quality = np.clip(reliability,0,1)
    ball_stop=np.where(reliable_slow,ball_quality,0.)
    ball_live=np.where(reliable_fast,ball_quality,0.)
    observed_flow=obs&reliable_fast&near&~very_low
    # Solo proximidad no demuestra juego: exige movimiento corroborado.
    team_flow=speeds_valid&(~low_teams)&(fast_teams|coll['collective_live'])
    play_pair=observed_flow & (team_flow|moved)
    # Configuración de preparación, NO una etiqueta de saque de esquina:
    # periferia + estabilización/reorganización + ausencia de circulación confirmada.
    preparation=(coll['collective_stop']|zon['zone_touchline_both']|zon['zone_compact_both'])
    spatial_setup=preparation & (low_teams|zon['zone_reorganization'])
    # Ponderacion por regimen: la calidad del balon controla cuanto puede
    # imponerse sobre las observaciones de jugadores, y no viceversa.
    player_quality=tact['tactical_reliability']
    ball_dominant=(reliability>=.70) & obs
    player_dominant=(player_quality>=.35) & (reliability<.35)
    context_stop=set_piece | spatial_setup | tact['tactical_collective_pause']
    tactical_standstill=tact['tactical_collective_pause'] | (
        tact['tactical_congestion_idle'] & set_piece)
    # SmartTag es contexto, no prueba directa de que el juego siga detenido.
    # Tres fuentes distintas: balón, equipos, configuración/contexto.
    strong_stop=(observed_cross | (reliable_slow&low_teams&context_stop) |
                 (very_low&spatial_setup&set_piece)) & ~play_pair
    # Para episodios sin balón, pausa colectiva requiere persistencia real,
    # corroboración espacial y descenso de velocidad en ambos equipos.
    sustained_collective=segment_rolling(very_low & spatial_setup,relevant,50,False,35)>=.70
    strong_stop |= sustained_collective & (~play_pair) & (set_piece|zon['zone_touchline_both'])
    # Posible parada por actores: exige cambio de comportamiento o
    # contexto de jugada y calidad suficiente de tracking.
    tactical_stop = tactical_standstill & (player_quality>=.35) & (
        tact['tactical_slowdown'] | set_piece | spatial_setup)
    strong_stop |= tactical_stop & ~play_pair & ~observed_flow
    # Ancla de juego fluido: pelota fiable + interacción con jugadores;
    # o circulación corroborada durante varias observaciones.
    # Un juego vivo con jugadores desplazandose debe poder contradecir
    # una supuesta preparacion aunque persista la etiqueta set_piece.
    tactical_live = tact['tactical_flow'] & (player_quality>=.35)
    strong_play=(play_pair | (obs & moved & reliable_fast & (team_flow | tactical_live))) & ~observed_cross
    # La circulacion observada con equipos activos es mas fiable que una
    # configuracion geometrica inmovil inferida de tracks parciales.
    verified_live = obs & moved & reliable_fast & tactical_live
    strong_play |= verified_live
    stop_support_10=segment_rolling(strong_stop,relevant,10,False,5)
    stop_support_25=segment_rolling(strong_stop,relevant,25,False,12)
    stop_support_50=segment_rolling(strong_stop,relevant,50,False,25)
    play_support_5=segment_rolling(strong_play,relevant,5,False,3)
    play_support_10=segment_rolling(strong_play,relevant,10,False,5)
    play_support_25=segment_rolling(strong_play,relevant,25,False,12)
    # Anclas de verdad fuerte: las señales deben sostenerse.
    stop_anchor=observed_cross | ((stop_support_10>=.7)&(stop_support_25>=.48)) | (
        sustained_collective&(stop_support_50>=.45))
    play_anchor=(play_support_5>=.8)&(play_support_10>=.65)
    # Las anclas sin balon requieren persistencia muy alta y corroboracion:
    # no inferimos que la pelota este muerta solo por faltar tracking.
    stop_anchor |= (segment_rolling(tactical_stop,relevant,50,False,30)>=.80) & (
        segment_rolling(set_piece | spatial_setup,relevant,50,False,30)>=.45) & (
        segment_rolling(strong_play,relevant,25,False,12)<.12)
    # Consistencia: juego observado y sostenido debe prevalecer sobre
    # inferencias espaciales suaves; salida observada prevalece sobre play.
    play_anchor &= ~observed_cross
    stop_anchor &= ~(play_anchor & ~observed_cross)
    # Grados de pertenencia multiescala; el futuro solo sirve para ajustar
    # fronteras OFFLINE, no como evidencia causal en el frame actual.
    st25=segment_rolling(strong_stop,relevant,25,True,10)
    st50=segment_rolling(strong_stop,relevant,50,True,20)
    pl25=segment_rolling(strong_play,relevant,25,True,10)
    pl50=segment_rolling(strong_play,relevant,50,True,20)
    prepared=segment_rolling(spatial_setup,relevant,50,True,20)
    stopped=segment_rolling(low_teams,relevant,50,True,20)
    tagging=segment_rolling(set_piece,relevant,50,True,20)
    tactical_still=segment_rolling(tactical_stop,relevant,50,True,20)
    tactical_live_window=segment_rolling(verified_live,relevant,25,True,10)
    tactical_motion=segment_rolling(tactical_live,relevant,25,True,10)
    # Ventanas de contexto mas largas evitan tratar una congestion breve
    # como evidencia de pausa.
    tactical_still_long=segment_rolling(tactical_stop,relevant,125,True,50)
    # Contexto suave: nunca D solo por baja velocidad o por set_piece.
    # Score de evidencia por familias, condicionado a fiabilidad.
    # No es probabilidad ni porcentaje calibrado.
    stop_strength=(2.0*st25 + 1.2*st50 + .6*(prepared*stopped)
        + .3*(tagging*prepared)
        + (.55 + .5*player_quality)*tactical_still
        + .45*tactical_still_long)
    play_strength=(2.2*pl25 + 1.4*pl50
        + .25*segment_rolling(team_flow,relevant,25,True,10)
        + (.25 + .7*reliability)*tactical_live_window
        + .25*tactical_motion)
    # No permitimos que un episodio con circulacion observada sostenida
    # sea absorbido por una ancla colectiva poco fiable.
    play_strength += .45 * segment_rolling(verified_live,relevant,50,True,20)
    dead, nstop,nplay, emission=infer_activity_from_anchors(
        relevant,stop_anchor,play_anchor,stop_strength,play_strength,
        context_stop,team_flow,obs)
    # Si un episodio D carece de anclas y tampoco tiene evidencia espacial
    # suficiente, se elimina: evita inventar D en zonas ambiguas.
    reasons=np.full(n,'ACTIVE_EVIDENCE_OR_CONTINUITY',dtype=object)
    states=np.full(n,'EXCLUDED',dtype=object)
    states[relevant]='ACTIVE'
    for a,b in runs(dead & relevant):
        has_anchor=bool(np.any(stop_anchor[a:b+1]))
        if not has_anchor:
            dead[a:b+1]=False
            reasons[a:b+1]='UNSUPPORTED_STOP_REJECTED'
            continue
        states[a:b+1]='DEAD'
        reasons[a:b+1]='DEAD_ANCHOR_BRIDGE'
        idx=np.where(stop_anchor[a:b+1])[0]+a
        reasons[idx]='HIGH_CERTAINTY_STOP_ANCHOR'
    reasons[play_anchor&~dead]='HIGH_CERTAINTY_PLAY_ANCHOR'
    # Registro explicativo de transiciones reales, no de cambios de score.
    stop_events=sum(1 for a,b in runs(dead&relevant))
    restart_events=sum(1 for a,b in runs(dead&relevant) if b+1<n and relevant[b+1])
    age=np.zeros(n,dtype=np.int32)
    for a,b in runs(dead&relevant):
        last=a
        for i in range(a,b+1):
            if stop_anchor[i]:last=i
            age[i]=i-last
    net=stop_strength-play_strength
    return {
      'observed':obs,'imputed':imp,'long_gap':long_gap,'outside':out,
      'low_ball':reliable_slow,'low_teams':low_teams,'set_piece':set_piece,
      'dead_confirmed':dead,'dead_reason':reasons,'ball_speed':speed,
      'home_speed':hs,'away_speed':ass,'reliability':reliability,
      'near_edge':near_edge,'near_corner':near_corner,'control_near':near,
      'fast_ball':reliable_fast,'dead_score':stop_strength,'live_score':play_strength,
      'net_score':net,'activity_state':states,'stop_candidate':strong_stop,
      'restart_candidate':strong_play,'stop_events':stop_events,'restart_events':restart_events,
      'stop_accumulator':stop_strength.astype('float32'),
      'restart_accumulator':play_strength.astype('float32'),
      'stop_support_10':stop_support_10,'stop_support_25':stop_support_25,
      'stop_support_50':stop_support_50,'restart_support_5':play_support_5,
      'restart_support_10':play_support_10,'restart_support_25':play_support_25,
      'stop_evidence_age':age,'stop_raw':strong_stop,'restart_raw':strong_play,
      'stop_anchor':stop_anchor,'play_anchor':play_anchor,
      'stop_anchor_count':nstop,'play_anchor_count':nplay,
      'activity_emission':emission,'spatial_setup':spatial_setup,
      'observed_flow':observed_flow,'zone_touchline_both':zon['zone_touchline_both'],
      'zone_compact_both':zon['zone_compact_both'],
      'zone_reorganization':zon['zone_reorganization'],
      **coll, **tact,
    }


# ============================================================
# TH01-TH15 -> EVIDENCIA DE ACTIVIDAD (NO POSTPROCESADO)
# ============================================================

TH_RESTART_TYPES = frozenset(("TH01", "TH02", "TH03", "TH04", "TH05", "TH06"))

# G3: seguimiento temporal TH y decoder global H/A. Todos los controles nuevos
# son neutros en 0; permiten verificar la equivalencia del baseline.
RAM_TH_PHYSICS_LIVE_WEIGHT = float(os.environ.get('RAM_TH_PHYSICS_LIVE_WEIGHT', '0.0'))
RAM_TH_FOLLOW_PRE_SECONDS = float(os.environ.get('RAM_TH_FOLLOW_PRE_SECONDS', '0.0'))
RAM_TH_FOLLOW_POST_SECONDS = float(os.environ.get('RAM_TH_FOLLOW_POST_SECONDS', '0.0'))
RAM_TH_FOLLOW_LIVE_FRAMES = int(os.environ.get('RAM_TH_FOLLOW_LIVE_FRAMES', '12'))
RAM_GLOBAL_HA_ENABLED = int(os.environ.get('RAM_GLOBAL_HA_ENABLED', '1'))
RAM_GLOBAL_HA_PRIOR = float(os.environ.get('RAM_GLOBAL_HA_PRIOR', '0.65'))
RAM_GLOBAL_HA_CANDIDATE = float(os.environ.get('RAM_GLOBAL_HA_CANDIDATE', '1.4'))
RAM_GLOBAL_HA_SWITCH_COST = float(os.environ.get('RAM_GLOBAL_HA_SWITCH_COST', '20.0'))
RAM_GLOBAL_HA_SMOOTH_FRAMES = int(os.environ.get('RAM_GLOBAL_HA_SMOOTH_FRAMES', '101'))
RAM_GLOBAL_HA_IMPUTED_WEIGHT = float(os.environ.get('RAM_GLOBAL_HA_IMPUTED_WEIGHT', '0.35'))
RAM_GLOBAL_HA_SMOOTH_MIX = float(os.environ.get('RAM_GLOBAL_HA_SMOOTH_MIX', '0.7'))
RAM_GLOBAL_HA_MIN_COVERAGE = float(os.environ.get('RAM_GLOBAL_HA_MIN_COVERAGE', '0.02'))


def follow_th_episodes(dead, relevant, stop_anchor, play_anchor, evidence,
                       pre_seconds=0., post_seconds=0., live_frames=12):
    """Expande episodios certificados TH solamente hasta juego activo confirmado.

    Nunca crea un D sin un ancla TH ni cruza X, medio tiempo, otra ancla
    de reanudacion o un bloque de flujo observado sostenido. No usa GT.
    """
    n=len(dead)
    out=np.array(dead,dtype=bool,copy=True)
    pre=max(0,int(round(float(pre_seconds)*FPS)))
    post=max(0,int(round(float(post_seconds)*FPS)))
    if pre==0 and post==0:return out,{'extended_frames':0,'episodes_considered':0}
    if not (1 <= live_frames <= 250):raise ValueError('RAM_TH_FOLLOW_LIVE_FRAMES fuera de rango')
    physical=(np.asarray(evidence['observed_flow'],dtype=bool) & ~np.asarray(evidence['low_teams'],dtype=bool)) \
        | (np.asarray(evidence['collective_live'],dtype=bool) & np.asarray(evidence['observed'],dtype=bool))
    hard_live=np.zeros(n,dtype=bool)
    for a,b in runs(physical & relevant):
        if b-a+1>=live_frames:hard_live[a:b+1]=True
    forbidden=~relevant | np.asarray(play_anchor,dtype=bool) | hard_live
    before=int(out.sum());eligible=0
    # Cada episodio del flag anterior actua como semilla; expandir no es generar
    # reglas adicionales independientes ni cambiar la decision de H/A a posteriori.
    for a,b in runs(dead & relevant):
        if not np.any(stop_anchor[a:b+1]):continue
        eligible+=1
        for t in range(a-1,max(-1,a-pre-1),-1):
            if forbidden[t] or dead[t]:break
            out[t]=True
        for t in range(b+1,min(n,b+post+1)):
            if forbidden[t] or dead[t]:break
            out[t]=True
    return out,{'extended_frames':int(out.sum())-before,'episodes_considered':eligible,
                'hard_live_veto_frames':int(hard_live.sum())}


def bridge_th_uncertain_gaps(dead,relevant,stop_anchor,play_anchor,evidence,
                             max_seconds=0.,max_live_fraction=.05):
    """Conecta dos paradas TH solo si un hueco es corto y carece de juego confirmado.

    No infiere D a partir de la ausencia de balon por si sola. No une episodios
    separados por X/descanso o reanudaciones. Requiere anclas TH a ambos lados
    y una region ambigua (imputacion o equipos lentos). Es previo a H/A.
    """
    original=np.asarray(dead,dtype=bool)
    out=original.copy()
    stats={'enabled':bool(max_seconds),'bridged_frames':0,'bridged_gaps':0,
           'blocked_by_live':0,'blocked_by_anchor':0,'blocked_by_uncertainty':0}
    limit=int(round(float(max_seconds)*FPS))
    if limit<=0:return out,stats
    observed=np.asarray(evidence['observed'],dtype=bool)
    low=np.asarray(evidence['low_teams'],dtype=bool)
    flow=(np.asarray(evidence['observed_flow'],dtype=bool)&~low) | (np.asarray(evidence['collective_live'],dtype=bool)&observed)
    uncertain=np.asarray(evidence['imputed'],dtype=bool)|low
    for left,right in runs(relevant):
        islands=[(left+a,left+b) for a,b in runs(original[left:right+1])]
        for (_,end0),(start1,_) in zip(islands,islands[1:]):
            a,b=end0+1,start1-1
            if a>b or b-a+1>limit:continue
            # Las dos islas deben proceder de anclas TH, no del suavizador.
            prev_start=next((x for x,y in islands if y==end0),end0)
            next_end=next((y for x,y in islands if x==start1),start1)
            if not (np.any(stop_anchor[prev_start:end0+1]) and np.any(stop_anchor[start1:next_end+1])):
                stats['blocked_by_anchor']+=1;continue
            if np.any(play_anchor[a:b+1]) or float(np.mean(flow[a:b+1]))>float(max_live_fraction):
                stats['blocked_by_live']+=1;continue
            if np.mean(uncertain[a:b+1])<.20:
                stats['blocked_by_uncertainty']+=1;continue
            out[a:b+1]=True
            stats['bridged_frames']+=b-a+1
            stats['bridged_gaps']+=1
    return out,stats


def _centered_mean_segment(v,k):
    """Media movil O(n) sin contaminar tramos separados por D/X."""
    if k<=1:return np.asarray(v,dtype=float)
    v=np.asarray(v,dtype=float)
    n=len(v);cs=np.r_[0.,np.cumsum(v)]
    pos=np.arange(n)
    lo=np.maximum(0,pos-k//2)
    hi=np.minimum(n,pos+(k+1)//2)
    return (cs[hi]-cs[lo])/np.maximum(1,hi-lo)


def decode_global_ha(result, relevant, candidate, evidence,
                     *, enabled=None, prior=None, candidate_weight=None,
                     switch_cost=None, smooth_frames=None, imputed_weight=None,
                     smooth_mix=None, min_coverage=None):
    """Decodificador Viterbi binario offline, dentro de cada tramo activo.

    Combina el owner del RAM original y la geometria de control, con peso
    mayor para balón observado. La penalización por cambio afecta la SECUENCIA,
    no solo pequeñas islas; nunca altera D/X ni usa etiquetas de referencia.
    """
    enabled=RAM_GLOBAL_HA_ENABLED if enabled is None else enabled
    stats={'enabled':bool(enabled),'changed_frames':0,'changed_segments':0,
           'regions_scanned':0,'regions_decoded':0,'transitions_before':0,
           'transitions_after':0}
    if not enabled:
        result['global_ha_stats']=stats
        return stats
    prior=RAM_GLOBAL_HA_PRIOR if prior is None else float(prior)
    candidate_weight=RAM_GLOBAL_HA_CANDIDATE if candidate_weight is None else float(candidate_weight)
    switch_cost=RAM_GLOBAL_HA_SWITCH_COST if switch_cost is None else float(switch_cost)
    smooth_frames=RAM_GLOBAL_HA_SMOOTH_FRAMES if smooth_frames is None else int(smooth_frames)
    imputed_weight=RAM_GLOBAL_HA_IMPUTED_WEIGHT if imputed_weight is None else float(imputed_weight)
    smooth_mix=RAM_GLOBAL_HA_SMOOTH_MIX if smooth_mix is None else float(smooth_mix)
    min_coverage=RAM_GLOBAL_HA_MIN_COVERAGE if min_coverage is None else float(min_coverage)
    if not (0<=smooth_mix<=1 and 0<=imputed_weight<=1 and 0<=min_coverage<=1 and switch_cost>=0 and smooth_frames>=1):
        raise ValueError('Parametros globales HA invalidos')
    labels=result['labels'].copy()
    reasons=result['reasons'].copy()
    mask=relevant & np.isin(labels,('H','A'))
    observed=np.asarray(evidence['observed'],dtype=bool)
    imputed=np.asarray(evidence['imputed'],dtype=bool)
    for a,b in runs(mask):
        stats['regions_scanned']+=1
        m=b-a+1
        if m<3:continue
        old=labels[a:b+1].copy()
        cand=np.asarray(candidate[a:b+1])
        csign=np.where(cand=='H',1.,np.where(cand=='A',-1.,0.))
        if np.mean(csign!=0)<min_coverage:continue
        stats['regions_decoded']+=1
        base=np.where(old=='H',1.,-1.)
        reliability=np.where(observed[a:b+1],1.,np.where(imputed[a:b+1],imputed_weight,0.12))
        evidence_score=prior*base + candidate_weight*csign*reliability
        smoothed=_centered_mean_segment(evidence_score,smooth_frames)
        e=(1-smooth_mix)*evidence_score+smooth_mix*smoothed
        # O(n) DP; coste igual para ambos equipos, sin favorecer uno por defecto.
        cost_h=np.zeros(m,dtype=float);cost_a=np.zeros(m,dtype=float)
        back_h=np.zeros(m,dtype=np.int8);back_a=np.zeros(m,dtype=np.int8)
        cost_h[0]=-e[0]+(0. if old[0]=='H' else prior)
        cost_a[0]=e[0]+(0. if old[0]=='A' else prior)
        g4_extra=float(os.environ.get('RAM_G4_GLOBAL_IMPUTED_TRANSITION_EXTRA','0.0'))
        g4_discount=float(os.environ.get('RAM_G4_GLOBAL_OBSERVED_TRANSITION_DISCOUNT','0.0'))
        if g4_extra<0 or g4_discount<0:raise ValueError('G4: costes adicionales deben ser >=0')
        for j in range(1,m):
            # G4: transiciones más costosas si el balón está imputado;
            # menos costosas con dos observaciones consecutivas. El coste
            # original se conserva exactamente si ambos ajustes son cero.
            pair_imputed=bool(imputed[a+j] or imputed[a+j-1])
            pair_observed=bool(observed[a+j] and observed[a+j-1])
            local_cost=max(0.,switch_cost + g4_extra*pair_imputed - g4_discount*pair_observed)
            # G7: sólo candidato observado del equipo ENTRANTE y balón
            # circulando permiten reducir la fricción. No inferir un toque
            # exacto: es un proxy de control, por eso se audita como tal.
            can_discount=(RAM_G7_TOUCH_DISCOUNT>0 and pair_observed
                          and bool(evidence['observed_flow'][a+j]))
            cost_to_h=max(0.,local_cost-(RAM_G7_TOUCH_DISCOUNT if can_discount and csign[j]>0 else 0.))
            cost_to_a=max(0.,local_cost-(RAM_G7_TOUCH_DISCOUNT if can_discount and csign[j]<0 else 0.))
            ch_stay=cost_h[j-1];ch_switch=cost_a[j-1]+cost_to_h
            ca_stay=cost_a[j-1];ca_switch=cost_h[j-1]+cost_to_a
            switch_h=(ch_switch<ch_stay)
            switch_a=(ca_switch<ca_stay)
            back_h[j]=1 if switch_h else 0  # H recibe desde A cuando cambia
            back_a[j]=0 if switch_a else 1  # A recibe desde H cuando cambia
            cost_h[j]=(ch_switch if switch_h else ch_stay)-e[j]
            cost_a[j]=(ca_switch if switch_a else ca_stay)+e[j]
        states=np.zeros(m,dtype=np.int8)
        states[-1]=int(cost_a[-1]<cost_h[-1])
        for j in range(m-1,0,-1):
            states[j-1]=back_a[j] if states[j] else back_h[j]
        new=np.where(states==0,'H','A')
        changed=(new!=old)
        stats['transitions_before']+=int(np.sum(old[1:]!=old[:-1]))
        stats['transitions_after']+=int(np.sum(new[1:]!=new[:-1]))
        if changed.any():
            stats['changed_frames']+=int(changed.sum())
            stats['changed_segments']+=1
            labels[a:b+1]=new
            rr=reasons[a:b+1].copy()
            rr[changed]='GLOBAL_HA_SEQUENCE_DECODER'
            reasons[a:b+1]=rr
    result['labels']=labels
    result['reasons']=reasons
    pf=result['possession_flag']
    pf[labels=='H']=1
    pf[labels=='A']=0
    result['global_ha_stats']=stats
    return stats

TH_CONTEXT_TYPES = frozenset(("TH07", "TH08", "TH09", "TH10", "TH11", "TH12", "TH13", "TH14", "TH15"))
TH_BAD_TIERS = frozenset(("POSSIBLE_ONLY", "WEAK_CANDIDATE", "LOW_CANDIDATE", "GEOMETRIC_RISK", "UNRANKED"))


def _th_int(value):
    try:
        z = float(value)
        return int(round(z)) if np.isfinite(z) else None
    except (ValueError, TypeError, OverflowError):
        return None


def g6_fit_motion_regimes(v4, relevant, components=2, min_coverage=.45,
                          severity_cutoff_mps=None):
    """Estimación determinista del régimen de movimiento a lo largo del tiempo.

    Se modela log(1 + velocidad máxima de las medianas de ambos equipos):
    la velocidad máxima evita que un equipo parado oculte que el otro juega.
    Cada mitad tiene su propia mezcla. Una mezcla describe velocidades, no D.
    """
    n=len(relevant)
    probability=np.full(n,np.nan,dtype=float)
    valid=np.zeros(n,dtype=bool)
    fits=[]
    s1=numeric(v4,"Barcelona_speed_median")
    s2=numeric(v4,"Atletic_Club_speed_median")
    c1=numeric(v4,"Barcelona_speed_coverage",1.)
    c2=numeric(v4,"Atletic_Club_speed_coverage",1.)
    # S07 puede representar cobertura como una proporción; otros productores
    # usan contadores. Si no hay columna se verifica la velocidad finita.
    valid=(relevant & np.isfinite(s1)&np.isfinite(s2)
           &(s1>=0)&(s2>=0)&(c1>=min_coverage)&(c2>=min_coverage))
    transformed=np.log1p(np.maximum(s1,s2))
    for a,b in runs(relevant):
        ids=np.flatnonzero(valid[a:b+1])+a
        # Muestreo por tiempo para reducir autocorrelación, NO por etiquetas.
        samples=ids[::max(1,FPS)]
        if len(samples)<150:
            fits.append({"start":a,"end":b,"status":"insufficient_tracking","samples":len(samples)})
            continue
        x=transformed[samples]
        if np.std(x)<.045:
            fits.append({"start":a,"end":b,"status":"degenerate_distribution","samples":len(samples)})
            continue
        centers=np.quantile(x,np.linspace(.18,.82,components))
        variances=np.full(components,max(float(np.var(x))*.35,.012))
        weights=np.full(components,1./components)
        for _ in range(55):
            loglike=np.stack([np.log(max(weights[j],1e-8))-.5*np.log(2*np.pi*max(variances[j],.008))-
                (x-centers[j])**2/(2*max(variances[j],.008)) for j in range(components)],axis=1)
            mx=loglike.max(axis=1,keepdims=True)
            posterior=np.exp(loglike-mx)
            posterior/=np.maximum(posterior.sum(axis=1,keepdims=True),1e-12)
            mass=posterior.sum(axis=0)
            new_weights=mass/len(x)
            new_centers=(posterior*x[:,None]).sum(axis=0)/np.maximum(mass,1e-9)
            new_variances=(posterior*(x[:,None]-new_centers)**2).sum(axis=0)/np.maximum(mass,1e-9)
            if np.min(mass)<10:break
            weights=np.maximum(new_weights,1e-4);weights/=weights.sum()
            centers=new_centers
            variances=np.maximum(new_variances,.008)
        order=np.argsort(centers)
        centers=centers[order];variances=variances[order];weights=weights[order]
        z=transformed[ids]
        loglike=np.stack([np.log(max(weights[j],1e-8))-.5*np.log(2*np.pi*variances[j])-
            (z-centers[j])**2/(2*variances[j]) for j in range(components)],axis=1)
        post=np.exp(loglike-loglike.max(axis=1,keepdims=True))
        post/=np.maximum(post.sum(axis=1,keepdims=True),1e-12)
        if severity_cutoff_mps is None:
            probability[ids]=post[:,0]
        else:
            # Los dos o tres regímenes contribuyen, proporcionalmente a
            # su velocidad característica. Una tercera componente intermedia
            # no desaparece por escoger simplemente post[:,0].
            mean_speed=np.expm1(centers)
            severity=np.clip(.5+(severity_cutoff_mps-mean_speed)/1.0,0.,1.)
            probability[ids]=post@severity
        fits.append({"start":a,"end":b,"status":"ok","samples":len(samples),
                     "means_mps":[round(float(np.expm1(q)),4) for q in centers],
                     "weights":[round(float(q),4) for q in weights]})
    return probability,valid,fits


def g6_extend_th_with_motion(dead,relevant,stop_anchor,play_anchor,evidence,
                              slow_probability,motion_valid,max_seconds,
                              min_slow_probability,hold_seconds):
    """Amplia un episodio D SOLO desde D que contenga anclaje TH.

    La ausencia del balon NO es evidencia de D. Una señal fuerte de juego
    activo o un anclaje de reanudación detiene la ampliación. No cruza mitades.
    """
    result=np.asarray(dead,dtype=bool).copy()
    eligible=(np.asarray(relevant,dtype=bool)&np.asarray(motion_valid,dtype=bool)
              &(np.asarray(slow_probability)>=min_slow_probability)
              &~np.asarray(play_anchor,dtype=bool)
              &~(np.asarray(evidence['observed_flow'],dtype=bool)
                  & np.asarray(evidence['collective_live'],dtype=bool)))
    limit=int(round(max_seconds*FPS))
    minimum=max(1,int(round(hold_seconds*FPS)))
    stats={"enabled":True,"gmm_eligible_frames":int(eligible.sum()),
           "episodes_checked":0,"seeded_episodes":0,"expanded_episodes":0,
           "extended_frames":0,"veto_play_frames":int(np.sum(play_anchor&relevant))}
    if limit<=0:return result,stats
    for pa,pb in runs(relevant):
        for local_a,local_b in runs(result[pa:pb+1]):
            a,b=pa+local_a,pa+local_b
            stats['episodes_checked']+=1
            if not np.any(stop_anchor[a:b+1]):continue
            stats['seeded_episodes']+=1
            expanded=False
            # No dividir estados previos, no invadir D/X ajeno, veto hard play.
            left=a
            while left>pa and a-left<limit and eligible[left-1] and not result[left-1]:left-=1
            right=b
            while right<pb and right-b<limit and eligible[right+1] and not result[right+1]:right+=1
            if a-left>=minimum:
                result[left:a]=True;stats['extended_frames']+=a-left;expanded=True
            if right-b>=minimum:
                result[b+1:right+1]=True;stats['extended_frames']+=right-b;expanded=True
            if expanded:stats['expanded_episodes']+=1
    return result,stats


def g7_contextual_candidates(dead, relevant, context_by, evidence, v4,
                             slow_probability=None, th_play_anchor=None,
                             max_side_seconds=30.,
                             near_seconds=12., hold_seconds=1.2,
                             slow_threshold=.63):
    """Contexto TH + movimiento + evidencia espacial/ball, antes de H/A.

    No se etiqueta un gol, una falta ni una pausa de hidratación por
    proximidad geométrica aislada. TH08/TH12 son HIPÓTESIS de causa.
    Jamás inicia D sólo porque el balón esté ausente o los jugadores lentos.
    """
    n=len(relevant)
    out=np.asarray(dead,dtype=bool).copy()
    near=np.zeros(n,dtype=bool)
    by_type={}
    near_by_type={}
    for typ in RAM_G7_CONTEXT_TYPES:
        c=np.asarray(context_by.get(typ,np.zeros(n,dtype=bool)),dtype=bool)
        by_type[typ]=int(np.sum(c&relevant))
        # Dilatar evento dentro de cada mitad, NO entre mitades.
        k=max(1,int(round(near_seconds*FPS)))
        type_near=segment_rolling(c.astype(float),relevant,2*k+1,True,1)>0
        near_by_type[typ]=type_near
        near |= type_near
    if slow_probability is None:
        motion=(np.asarray(evidence['low_teams'],dtype=bool)
                & np.asarray(evidence['collective_valid'],dtype=bool))
    else:
        motion=np.nan_to_num(np.asarray(slow_probability),nan=0.)>=slow_threshold
    shape=(np.asarray(evidence['collective_stop'],bool)
           |np.asarray(evidence['zone_reorganization'],bool)
           |np.asarray(evidence['zone_touchline_both'],bool)
           |np.asarray(evidence['spatial_setup'],bool))
    # Una señal de balón estacionario observado es evidencia independiente.
    ball_idle=(np.asarray(evidence['observed'],bool)
               &np.asarray(evidence['low_ball'],bool))
    # Juego confirmado es veto; no basta un end_low de un TH para declarar LIVE.
    active=(np.asarray(evidence['observed_flow'],bool)
            &np.asarray(evidence['collective_live'],bool))
    veto=(np.asarray(th_play_anchor,dtype=bool) & active if th_play_anchor is not None
          else np.zeros(n,dtype=bool))
    corroborated=(near & motion & (shape | ball_idle) & ~active & ~veto & relevant)
    hold=max(1,int(round(hold_seconds*FPS)))
    # Aguanta huecos breves de observación, pero no atraviesa un veto.
    supported=segment_rolling(corroborated.astype(float),relevant,hold,True,
                              max(1,hold//2))>=.60
    supported&=near&~active&~veto&relevant
    stats={'enabled':True,'typed_context_frames':by_type,
           'context_near_frames':int((near&relevant).sum()),
           'motion_frames':int((motion&relevant).sum()),
           'shape_frames':int((shape&relevant).sum()),
           'observed_idle_ball_frames':int((ball_idle&relevant).sum()),
           'motion_and_context_frames':int((near&motion&relevant).sum()),
           'motion_shape_context_frames':int((near&motion&(shape|ball_idle)&relevant).sum()),
           'context_before_hold_frames':int(corroborated.sum()),
           'per_type_near_frames':{typ:int((mask&relevant).sum()) for typ,mask in near_by_type.items()},
           'per_type_supported_frames':{typ:int((mask&supported).sum()) for typ,mask in near_by_type.items()},
           'physics_supported_frames':int(supported.sum()),
           'new_episodes':0,'new_D_frames':0,'veto_frames':int((active&relevant).sum())}
    # Apertura de nuevos episodios D únicamente si la secuencia física
    # permanece corroborada; nunca toda una ventana TH por sí misma.
    limit=max(0,int(round(max_side_seconds*FPS)))
    for left,right in runs(relevant):
        for aa,bb in runs(supported[left:right+1]):
            a,b=left+aa,left+bb
            if (b-a+1)<hold:continue
            # Episodio con soporte TH/contexto; sumar frames dentro del soporte.
            old=out[a:b+1].copy()
            if not old.all():
                out[a:b+1]=True
                stats['new_D_frames']+=int((~old).sum())
                stats['new_episodes']+=1
            # Seguir en ambos sentidos sólo con corroboración física sostenida.
            if limit:
                for direction in (-1,1):
                    j=(a-1 if direction<0 else b+1);count=0
                    while left<=j<=right and count<limit and not active[j] and not veto[j]:
                        # Seguimiento: exige evidencia física incluso fuera
                        # del radio inicial de la hipótesis contextual.
                        if not motion[j] or not (shape[j] or ball_idle[j]):break
                        if not out[j]:out[j]=True;stats['new_D_frames']+=1
                        j+=direction;count+=1
    added=out&~np.asarray(dead,dtype=bool)&relevant
    stats['per_type_added_frames']={typ:int((mask&added).sum()) for typ,mask in near_by_type.items()}
    return out,supported,stats


def g7_joint_decode(result,relevant,candidate,evidence,
                    ha_cost=24.,enter_d=11.,exit_d=12.,
                    base_prior=2.,context_reward=4.,
                    unverified_exit_extra=0.):
    """Viterbi conjunto H/A/D, por mitad, sin X o uso de ground truth.

    Fuente de D nueva: sólo corroboración G7 o D previo; el HMM no puede
    inventar paradas en territorios sin evidencia. H/A usa candidato
    geométrico y observación real, conservando desacuerdo trazable.
    """
    old=np.asarray(result['labels']).copy();new=old.copy();n=len(old)
    ctx=np.asarray(evidence.get('g7_context_candidate',np.zeros(n,bool)),bool)
    prev_dead=np.asarray(evidence['dead_confirmed'],bool)
    obs=np.asarray(evidence['observed'],bool)
    play=(np.asarray(evidence['observed_flow'],bool)
          &np.asarray(evidence['collective_live'],bool))
    stop=np.asarray(evidence['stop_anchor'],bool)
    stats={'enabled':True,'new_D_frames':0,'recovered_live_frames':0,
           'HA_changed_frames':0,'transitions_before':0,'transitions_after':0}
    candidate=np.asarray(candidate)
    trans=np.array([[0.,ha_cost,enter_d],[ha_cost,0.,enter_d],
                    [exit_d,exit_d,0.]],dtype=np.float32)
    # Reanudación creíble: balón realmente observado y circulación física,
    # o control geométrico nuevo con balón observado y velocidad real.
    confirmed_restart=play | (obs & np.isin(candidate,('H','A'))
                                & np.asarray(evidence.get('fast_ball',np.zeros(n,bool)),bool))
    evidence['g7_joint_restart_confirmed']=confirmed_restart
    stats['verified_restart_frames']=int(np.sum(confirmed_restart&relevant))
    for a,b in runs(relevant):
        m=b-a+1
        # Estado 0=H, 1=A, 2=D.
        emission=np.zeros((m,3),dtype=np.float32)
        baseline=old[a:b+1]; cc=candidate[a:b+1]
        for st,k in enumerate(('H','A','D')):
            emission[:,st]+=base_prior*(baseline==k)
        # Evidencia no calibrada: se usa como potencial, NO probabilidad.
        reliability=np.where(obs[a:b+1],1.,.15)
        emission[:,0]+=2.*(cc=='H')*reliability
        emission[:,1]+=2.*(cc=='A')*reliability
        emission[:,2]+=context_reward*ctx[a:b+1]
        # No promover D sin el soporte que ha pasado el control TH+físico.
        valid_D=prev_dead[a:b+1]|ctx[a:b+1]
        emission[~valid_D,2]=-1e5
        # Confirmación positiva de circulación: no absorber juego real.
        emission[play[a:b+1],2]=-1e5
        emission[stop[a:b+1] & ~play[a:b+1],0]=-1e5
        emission[stop[a:b+1] & ~play[a:b+1],1]=-1e5
        costs=np.full((m,3),np.inf,dtype=np.float64)
        back=np.zeros((m,3),dtype=np.int8)
        costs[0]=-emission[0]
        if baseline[0] in ('H','A','D'):
            costs[0,[k for k in range(3) if ('H','A','D')[k]!=baseline[0]]]+=base_prior
        for j in range(1,m):
            if unverified_exit_extra>0 and not confirmed_restart[a+j]:
                local_trans=trans.copy()
                local_trans[2,0]+=unverified_exit_extra
                local_trans[2,1]+=unverified_exit_extra
            else:
                local_trans=trans
            z=costs[j-1][:,None]+local_trans
            parent=z.argmin(axis=0)
            back[j]=parent
            costs[j]=z[parent,np.arange(3)]-emission[j]
        states=np.zeros(m,dtype=np.int8)
        states[-1]=int(np.argmin(costs[-1]))
        for j in range(m-1,0,-1):states[j-1]=back[j,states[j]]
        decoded=np.array(('H','A','D'))[states]
        new[a:b+1]=decoded
        stats['transitions_before']+=int(np.sum(baseline[1:]!=baseline[:-1]))
        stats['transitions_after']+=int(np.sum(decoded[1:]!=decoded[:-1]))
    was_D=old=='D';now_D=new=='D'
    stats['new_D_frames']=int(np.sum(now_D&~was_D&relevant))
    stats['recovered_live_frames']=int(np.sum(was_D&~now_D&relevant))
    stats['HA_changed_frames']=int(np.sum((old!=new)&(~was_D)&(~now_D)&relevant))
    changed=(new!=old)&relevant
    result['labels']=new
    reasons=result['reasons'].copy()
    reasons[changed]='G7_JOINT_HAD_SEQUENCE'
    result['reasons']=reasons
    internal=result['states'].copy() if 'states' in result else None
    if internal is not None:
        internal[changed & (new=='D')]='DEAD'
        internal[changed & (new=='H')]='CONTROL_H'
        internal[changed & (new=='A')]='CONTROL_A'
        result['states']=internal
    result['possession_flag'][new=='H']=1
    result['possession_flag'][new=='A']=0
    result['possession_flag'][new=='D']=-1
    # Contrato real de decide_possession(): la bandera se llama 'active',
    # no 'active_flag'. Esta clave también es la que serializa save_outputs().
    result['active'][relevant & (new=='D')]=0
    result['active'][relevant & np.isin(new,('H','A'))]=1
    evidence['g7_joint_added_D']=(~was_D)&now_D&relevant
    evidence['g7_joint_removed_D']=was_D&(~now_D)&relevant
    evidence['dead_confirmed']=now_D&relevant
    evidence['stop_events']=sum(1 for _ in runs(now_D&relevant))
    evidence['restart_events']=sum(1 for a,b in runs(now_D&relevant) if b+1<n and relevant[b+1])
    evidence['dead_reason'][evidence['g7_joint_added_D']]='G7_JOINT_CONTEXTUAL_D'
    evidence['dead_reason'][evidence['g7_joint_removed_D']]='G7_JOINT_LIVE_OBSERVED_OR_PRIOR'
    evidence['activity_state'][relevant & (new=='D')]='DEAD'
    evidence['activity_state'][relevant & np.isin(new,('H','A'))]='ACTIVE'
    result['g7_joint_stats']=stats
    return stats


def th_primary_activity(evidence, th_path, relevant, v4=None):
    """Unico flag ACTIVE/DEAD: hipotesis TH01..TH15, NO V5.2 dead_confirmed.

    Emplea episodios delimitados como soporte de inactividad, los TH contextuales
    como corroboracion; una reanudacion temporalmente acotada como cierre.
    En tramos ambiguos conserva la continuidad por histéresis. No usa GT.
    """
    n=len(relevant)
    rows=pd.read_csv(th_path,low_memory=False)
    required={'detector','tier','evidence','start_low','start_high','end_low','end_high','anchor_frame'}
    if required-set(rows.columns):
        raise ValueError('TH: faltan columnas: '+str(sorted(required-set(rows.columns))))
    from collections import Counter
    accepted=Counter()
    skipped=Counter()
    seed_by={f'TH{i:02d}':np.zeros(n,dtype=bool) for i in range(1,16)}
    context_by={f'TH{i:02d}':np.zeros(n,dtype=bool) for i in range(1,16)}
    strong_by={f'TH{i:02d}':np.zeros(n,dtype=bool) for i in range(1,16)}
    bounded=np.zeros(n,dtype=bool)
    end_confirm=np.zeros(n,dtype=bool)
    # TH01-06: reanudaciones fisicas; TH07/08: causas/contextos;
    # TH09-13: factores complementarios; TH14: bote neutral potencial;
    # TH15: parada residual. Ningun tipo aporta certeza absoluta.
    primary={'TH01','TH02','TH03','TH04','TH05','TH06'}
    conditional={'TH14','TH15'}
    for row in rows.itertuples(index=False):
        k=str(row.detector);tier=str(row.tier);proof=str(row.evidence)
        if k not in seed_by:
            raise ValueError('TH desconocido: '+k)
        if ((tier in TH_BAD_TIERS and not (k=='TH10' and tier=='UNRANKED')
                                          and not ((RAM_G7_CONTEXT_ENABLED or RAM_G8_EPISODES_ENABLED) and k in ('TH08','TH12')))
                or 'CONTRADICTION' in proof.upper()):
            skipped['contradiction_or_weak']+=1
            continue
        lo=_th_int(row.start_low);hi=_th_int(row.start_high)
        el=_th_int(row.end_low);eh=_th_int(row.end_high)
        anchor=_th_int(row.anchor_frame)
        if lo is None and hi is None and anchor is None:
            skipped['no_temporal_anchor']+=1
            continue
        # El posible comienzo no es la primera D segura. Reservar el
        # intervalo incierto sin inventar un instante de silbato.
        start=hi if hi is not None else (lo if lo is not None else anchor)
        if start is None:continue
        if el is not None and el>=start:
            end=el;has_end=True
        elif eh is not None and eh>=start:
            end=eh;has_end=False
        else:
            # No tener fin observado NO permite mantener D indefinidamente.
            end=start+int(round(float(os.environ.get("RAM_TH_UNBOUNDED_SECONDS", "1.2"))*FPS));has_end=False
        a=max(0,start);b=min(n-1,end)
        if a>b or b-a>float(os.environ.get("RAM_TH_MAX_INTERVAL_SECONDS", "45"))*FPS:
            skipped['invalid_interval']+=1;continue
        if not relevant[a:b+1].any():
            skipped['outside_game']+=1;continue
        accepted[k]+=1
        if k in primary:
            seed_by[k][a:b+1]=True
            if has_end:
                bounded[a:b+1]=True
                if eh is not None and eh-el<=int(.8*FPS):
                    q=min(n,eh+1)
                    end_confirm[max(0,el):q]=True
        elif k in conditional:
            # TH14 solo es corroboracion hasta observar bote/reanudacion.
            # TH15 es residual y no implica un saque especifico.
            context_by[k][a:b+1]=True
            if k=='TH15' and 'STRUCTURAL_CANDIDATE' in tier and has_end:
                strong_by[k][a:b+1]=True
        else:
            context_by[k][a:b+1]=True
    restart_votes=np.zeros(n,dtype=np.int8)
    context_votes=np.zeros(n,dtype=np.int8)
    for k in primary: restart_votes+=seed_by[k].astype(np.int8)
    for k in context_by:context_votes+=context_by[k].astype(np.int8)
    seed=(restart_votes>0)
    # TH07,TH08 y TH09..TH15 refuerzan una interrupcion cercana,
    # sin multiplicar votos del mismo detector.
    context_near=segment_rolling(context_votes>0,relevant,25,True,12)>0
    seed_score=(.85*np.minimum(restart_votes,3)+.25*np.minimum(context_votes,3)
                +float(os.environ.get('RAM_TH_BOUNDED_BONUS', '0.20'))*bounded.astype(float)+.15*context_near.astype(float))
    seed_score*=np.where(seed,1.,0.)
    # TH15 estructural puede originar hipótesis residual, pero solo donde
    # existe delimitacion, y su evidencia es mas débil que una reanudación.
    residual=strong_by['TH15'] & ~seed
    seed_score[residual]=np.maximum(seed_score[residual],.50)
    # Conservar solo segmentos temporalmente consistentes, no fotogramas
    # aislados. El diagnostico viejo NO participa en ninguna formula.
    stop_support=(seed | residual) & relevant
    coherent=np.zeros(n,dtype=bool)
    for a,b in runs(stop_support):
        if b-a+1>=max(5,int(float(os.environ.get("RAM_TH_COHERENCE_SECONDS", "0.28"))*FPS)):
            coherent[a:b+1]=True
    # Expandir hacia atrás el comienzo probable, sin tocar el primer
    # frame fuera del juego ni extender mas allá del margen de cada TH.
    stop_anchor=np.zeros(n,dtype=bool)
    for aa,bb in runs(coherent):
        dur=bb-aa+1
        if dur>=int(float(os.environ.get("RAM_TH_ANCHOR_DURATION", "0.8"))*FPS) or (dur>=int(.5*FPS) and bounded[aa:bb+1].any()):
            stop_anchor[aa:bb+1]=seed_score[aa:bb+1]>=float(os.environ.get('RAM_TH_ANCHOR_THRESHOLD', '0.75'))
    stop_strength=np.where(coherent, .30+float(os.environ.get('RAM_TH_STOP_SCALE', '1.7'))*seed_score,0.)
    # Fuera de una hipótesis TH, la inactividad no se inicia por reglas V5.
    # Pequeña preferencia por LIVE rompe incertidumbres largas.
    play_strength=np.where(relevant,float(os.environ.get('RAM_TH_PLAY_PRIOR', '0.22')),0.)
    # Evidencia fisica independiente del antiguo dead_confirmed.
    if RAM_TH_PHYSICS_LIVE_WEIGHT>0:
        live_physical=(np.asarray(evidence['observed_flow'],dtype=bool) & ~np.asarray(evidence['low_teams'],dtype=bool)) | (np.asarray(evidence['collective_live'],dtype=bool) & np.asarray(evidence['observed'],dtype=bool))
        play_strength[live_physical & relevant]+=RAM_TH_PHYSICS_LIVE_WEIGHT
    # Un fin acotado es evidencia de REANUDACIÓN PROBABLE, no una certeza
    # de movimiento. Ancla fuerte solo si deja de haber soporte simultáneo.
    play_anchor=end_confirm & ~coherent & relevant
    play_strength[play_anchor]+=2.0
    # Fronteras con histéresis: hace falta evidencia acumulada para
    # abandonar D; gap corto ambiguo entre paradas se resuelve unido.
    dead,nstop,nplay,emission=infer_activity_from_anchors(
        relevant,stop_anchor,play_anchor,stop_strength,play_strength,
        coherent,play_anchor,evidence['observed'],min_dead_frames=int(os.environ.get('RAM_TH_MIN_DEAD', '12')))
    bridged=0;rejected=0
    for left,right in runs(relevant):
        for aa,bb in runs(~dead[left:right+1]):
            a,b=left+aa,left+bb
            if a<=left or b>=right or not dead[a-1] or not dead[b+1]:continue
            if b-a+1<=int(float(os.environ.get("RAM_ACTIVITY_BRIDGE_SECONDS", "2.0"))*FPS) and not play_anchor[a:b+1].any():
                dead[a:b+1]=True;bridged+=b-a+1
        for aa,bb in runs(dead[left:right+1]):
            a,b=left+aa,left+bb
            if b-a+1<int(float(os.environ.get("RAM_ACTIVITY_MICRO_D_SECONDS", "1.0"))*FPS) and not stop_anchor[a:b+1].any():
                dead[a:b+1]=False;rejected+=b-a+1
    dead,follow_stats=follow_th_episodes(dead,relevant,stop_anchor,play_anchor,evidence,
       pre_seconds=RAM_TH_FOLLOW_PRE_SECONDS,post_seconds=RAM_TH_FOLLOW_POST_SECONDS,
       live_frames=RAM_TH_FOLLOW_LIVE_FRAMES)
    evidence['th_follow_stats']=follow_stats
    before_g5_bridge=dead.copy()
    dead,g5_bridge_stats=bridge_th_uncertain_gaps(dead,relevant,stop_anchor,play_anchor,evidence,
        RAM_G5_TH_BRIDGE_MAX_SECONDS,RAM_G5_TH_BRIDGE_MAX_LIVE_FRACTION)
    evidence['g5_bridge_stats']=g5_bridge_stats
    evidence['g5_bridge_mask']=dead & ~before_g5_bridge
    before_g6_motion=dead.copy()
    if RAM_G6_MOTION_ENABLED:
        if v4 is None:raise ValueError("G6: requiere matrix_v4 para el régimen de movimiento")
        motion_prob,motion_valid,motion_fits=g6_fit_motion_regimes(
            v4,relevant,components=RAM_G6_MOTION_COMPONENTS,
            min_coverage=RAM_G6_MOTION_MIN_COVERAGE)
        dead,motion_stats=g6_extend_th_with_motion(
            dead,relevant,stop_anchor,play_anchor,evidence,motion_prob,motion_valid,
            RAM_G6_MOTION_MAX_SIDE_SECONDS,RAM_G6_MOTION_SLOW_PROB,
            RAM_G6_MOTION_HOLD_SECONDS)
        motion_stats['fits']=motion_fits
    else:
        motion_prob=np.full(n,np.nan,dtype=float)
        motion_stats={'enabled':False,'extended_frames':0,'fits':[]}
    evidence['g6_motion_probability']=motion_prob
    evidence['g6_motion_added_mask']=dead&~before_g6_motion
    evidence['g6_motion_stats']=motion_stats
    if RAM_G7_CONTEXT_ENABLED:
        if v4 is None:raise ValueError('G7 requiere matrix_v4')
        motion_g7,valid_g7,fits_g7=g6_fit_motion_regimes(
            v4,relevant,components=RAM_G7_MOTION_COMPONENTS,
            min_coverage=RAM_G6_MOTION_MIN_COVERAGE,
            severity_cutoff_mps=(RAM_G7_MOTION_CUTOFF_MPS
              if RAM_G7_MOTION_SEVERITY_ENABLED else None))
        before=dead.copy()
        dead,g7_context,g7_stats=g7_contextual_candidates(
            dead,relevant,context_by,evidence,v4,motion_g7,
            th_play_anchor=play_anchor,
            max_side_seconds=RAM_G7_CONTEXT_MAX_SECONDS,
            near_seconds=RAM_G7_CONTEXT_NEAR_SECONDS,
            hold_seconds=RAM_G7_CONTEXT_HOLD_SECONDS,
            slow_threshold=RAM_G7_MOTION_SLOW_PROB)
        g7_stats['fits']=fits_g7
    else:
        g7_context=np.zeros(n,dtype=bool)
        g7_stats={'enabled':False,'new_D_frames':0,'new_episodes':0}
        before=dead.copy()
    evidence['g8_th_masks']={k:(context_by[k]|seed_by[k]).copy() for k in context_by}
    evidence['g7_context_candidate']=g7_context
    evidence['g7_context_motion_probability']=(motion_g7 if RAM_G7_CONTEXT_ENABLED
                                              else np.full(n,np.nan))
    context_origin=np.full(n,'NONE',dtype=object)
    for typ in ('TH08','TH12','TH09','TH11','TH15'):
        if typ not in context_by:continue
        mark=context_by[typ]&g7_context
        context_origin[mark & (context_origin=='NONE')]=typ
        context_origin[mark & (context_origin!='NONE') & (context_origin!=typ)]='MULTI'
    evidence['g7_context_origin']=context_origin
    evidence['g7_context_added_D']=dead & ~before
    evidence['g7_context_stats']=g7_stats
    dead[~relevant]=False
    dead[play_anchor&~stop_anchor]=False
    evidence['g7_context_added_D'] &= dead
    evidence['g7_context_stats']['final_new_D_frames']=int(np.sum(evidence['g7_context_added_D']))
    # Sustituir por COMPLETO el flag antiguo: sus scores, anclas y
    # dead_confirmed no tienen efecto en la clasificacion RAM.
    legacy_dead=np.asarray(evidence['dead_confirmed'],dtype=bool)
    evidence['th_old_dead_only_diagnostic']=legacy_dead
    evidence['dead_confirmed']=dead
    evidence['dead_score']=stop_strength
    evidence['live_score']=play_strength
    evidence['net_score']=stop_strength-play_strength
    evidence['activity_emission']=emission
    evidence['stop_anchor']=stop_anchor
    evidence['play_anchor']=play_anchor
    evidence['stop_anchor_count']=int(nstop)
    evidence['play_anchor_count']=int(nplay)
    evidence['stop_accumulator']=stop_strength.astype('float32')
    evidence['restart_accumulator']=play_strength.astype('float32')
    evidence['stop_raw']=coherent
    evidence['restart_raw']=play_anchor
    for name,width in [('stop_support_10',10),('stop_support_25',25),('stop_support_50',50)]:
        evidence[name]=segment_rolling(coherent,relevant,width,False,max(1,width//2))
    for name,width in [('restart_support_5',5),('restart_support_10',10),('restart_support_25',25)]:
        evidence[name]=segment_rolling(play_anchor,relevant,width,False,max(1,width//2))
    evidence['th_restart_votes']=restart_votes
    evidence['th_context_votes']=context_votes
    evidence['th_stop_strength']=stop_strength
    evidence['th_bounded']=bounded
    evidence['th_lock_stats']={'bridged_frames':bridged,'rejected_short_D_frames':rejected}
    states=evidence['activity_state'].copy()
    reasons=evidence['dead_reason'].copy()
    states[relevant]='ACTIVE';states[dead]='DEAD'
    reasons[relevant]='TH_PRIMARY_ACTIVE_CONTINUITY'
    reasons[dead]='TH_PRIMARY_DEAD_EPISODE'
    reasons[stop_anchor&dead]='TH_PRIMARY_STOP_ANCHOR'
    reasons[evidence['g5_bridge_mask']]='G5_BRIDGED_BETWEEN_TH_ANCHORS'
    reasons[evidence['g6_motion_added_mask']]='G6_MOTION_TH_CONTINUITY'
    reasons[evidence['g7_context_added_D']]='G7_TH_CONTEXT_AND_PHYSICS'
    evidence['activity_state']=states
    evidence['dead_reason']=reasons
    age=np.zeros(n,dtype=np.int32)
    for a,b in runs(dead):
        last=a
        for i in range(a,b+1):
            if stop_anchor[i]:last=i
            age[i]=i-last
    evidence['stop_evidence_age']=age
    evidence['stop_events']=sum(1 for _ in runs(dead))
    evidence['restart_events']=sum(1 for a,b in runs(dead) if b+1<n and relevant[b+1])
    print('[TH-PRIMARIO] candidatos aceptados',dict(accepted),'omitidos',dict(skipped))
    print('[TH-PRIMARIO] D',int(np.sum(dead)),'spells',evidence['stop_events'],
          'vs bandera V5 diagnosticamente: +',int(np.sum(dead&~legacy_dead)),
          '-',int(np.sum(legacy_dead&~dead)),
          'TH simultáneos',int(np.sum(restart_votes>=2)),'histéresis',evidence['th_lock_stats'])
    return evidence,{'accepted':dict(accepted),'ignored':sum(skipped.values()),
                     'restart_votes':restart_votes,'context_votes':context_votes,
                     'bounded':bounded,'rejection_reasons':dict(skipped)}

# ============================================================
# 3. CANDIDATOS DE CONTROL
# ============================================================

def build_candidates(s27, relevant, v4=None):
    home = numeric(s27, "home_distance_m")
    away = numeric(s27, "away_distance_m")
    margin = numeric(s27, "distance_margin_home_m")

    comparable = bool_column(
        s27, "geometry_comparable"
    )

    candidate = np.full(
        len(s27), "NONE", dtype=object
    )

    radius=np.full(len(s27),CONTROL_RADIUS_M,dtype=float)
    advantage=np.full(len(s27),CONTROL_ADVANTAGE_M,dtype=float)
    if v4 is not None:
        imputed=bool_column(v4,'ball_imputed')
        observed=bool_column(v4,'ball_observed')
        radius[imputed]*=RAM_G5_IMPUTED_RADIUS_FACTOR
        radius[observed & ~imputed]*=RAM_G5_OBSERVED_RADIUS_FACTOR
        advantage[imputed]+=RAM_G5_IMPUTED_ADVANTAGE_EXTRA
    h = (
        relevant
        & comparable
        & (home <= radius)
        & (margin >= advantage)
    )

    a = (
        relevant
        & comparable
        & (away <= radius)
        & (margin <= -advantage)
    )

    # La geometría imputada muy distante puede generar falsa posesión.
    # Umbral 0: sin filtro (reproduce G3). Los valores >0 vetan
    # candidatos construidos sobre interpolaciones demasiado largas.
    g4_gap = int(os.environ.get("RAM_G4_MAX_IMPUTED_GAP_FOR_CONTROL", "0"))
    if g4_gap > 0 and v4 is not None:
        weak = bool_column(v4, "ball_imputed") & (numeric(v4,"ball_gap_length",0)>=g4_gap)
        h &= ~weak
        a &= ~weak
    candidate[h] = "H"
    candidate[a] = "A"

    return candidate, home, away, margin, comparable


# ============================================================
# 4. MOTOR DE ESTADOS
# ============================================================

# Ajustes de continuidad: se conservan valores neutros para estos correctores.
RAM_ADAPT_IMPUTED_PENALTY = float(os.environ.get("RAM_ADAPT_IMPUTED_PENALTY", "0.0"))
RAM_ADAPT_OBS_REWARD = float(os.environ.get("RAM_ADAPT_OBS_REWARD", "0.0"))
RAM_DEAD_OWNER_GRACE_SECONDS = float(os.environ.get("RAM_DEAD_OWNER_GRACE_SECONDS", "3.0"))
RAM_ISLAND_MAX_SECONDS = float(os.environ.get("RAM_ISLAND_MAX_SECONDS", "0.0"))
RAM_ISLAND_NEIGHBOR_SUPPORT = float(os.environ.get("RAM_ISLAND_NEIGHBOR_SUPPORT", "0.70"))
RAM_ISLAND_OPPOSING_SUPPORT = float(os.environ.get("RAM_ISLAND_OPPOSING_SUPPORT", "0.20"))
RAM_SWITCH_DEAD_EXIT_CONFIRM = int(os.environ.get("RAM_SWITCH_DEAD_EXIT_CONFIRM", "0"))


def decide_possession(
    relevant,
    periods,
    s27,
    evidence,
    candidate,
    margin,
    comparable,
):
    """Switch de posesion V4 con tres barreras independientes.

    1. Candidato rival sostenido (ventana corta y larga).
    2. Evidencia acumulada ponderada por fuente, contraevidencia del owner.
    3. Veto de control reciente del owner y enfriamiento tras un cambio.

    La confirmacion es causal. Solo despues se permite revisar hacia atras
    una frontera respaldada por candidatos reales. No se lee ground truth.
    """
    n = len(relevant)
    labels = np.full(n, "X", dtype="<U1")
    reasons = np.full(n, "OUTSIDE_PERIOD", dtype=object)
    states = np.full(n, "EXCLUDED", dtype=object)
    active_flag = np.full(n, -1, dtype=np.int8)
    possession_flag = np.full(n, -1, dtype=np.int8)
    uncertain = np.zeros(n, dtype=np.int8)
    pending_trace = np.full(n, "NONE", dtype=object)
    pending_count_trace = np.zeros(n, dtype=np.int32)
    switch_score_trace = np.zeros(n, dtype=np.float32)
    short_support_trace = np.zeros(n, dtype=np.float32)
    long_support_trace = np.zeros(n, dtype=np.float32)
    owner_veto_trace = np.zeros(n, dtype=np.int8)
    cooldown_trace = np.zeros(n, dtype=np.int16)
    challenger_evidence_trace = np.zeros(n, dtype=np.int8)
    owner_evidence_trace = np.zeros(n, dtype=np.int8)
    switch_ready_trace = np.zeros(n, dtype=np.int8)
    confirmed_changes = []
    blocked_by_veto = 0
    blocked_by_support = 0
    source = s27["ball_source"].fillna("NONE").astype(str).str.upper().to_numpy()

    owner = None
    challenger = None
    score = 0.0
    pending_frames = 0
    pending_observed = 0
    pending_start = None
    gap = 0
    last_change = -10**9
    recent_rival = np.zeros(SWITCH_LONG_WINDOW, dtype=np.int8)
    recent_owner = np.zeros(SWITCH_LONG_WINDOW, dtype=np.int8)
    ptr = 0
    history_count = 0

    def reset_pending():
        nonlocal challenger, score, pending_frames, pending_observed, pending_start, gap
        challenger = None
        score = 0.0
        pending_frames = 0
        pending_observed = 0
        pending_start = None
        gap = 0

    def reset_history():
        nonlocal ptr, history_count
        recent_rival.fill(0)
        recent_owner.fill(0)
        ptr = 0
        history_count = 0

    dead_streak = 0
    post_dead_live = 0
    for i in range(n):
        if not relevant[i]:
            owner = None
            reset_pending()
            reset_history()
            last_change = -10**9
            if periods[i] == "HALFTIME":
                labels[i] = "D"
                reasons[i] = "HALFTIME"
                active_flag[i] = 0
            continue

        if evidence["dead_confirmed"][i]:
            labels[i] = "D"
            reasons[i] = evidence["dead_reason"][i]
            states[i] = "DEAD"
            active_flag[i] = 0
            # Mantener titular previo durante interrupciones cortas; no
            # trasladarlo indefinidamente a una reanudación larga.
            dead_streak += 1
            if dead_streak > int(RAM_DEAD_OWNER_GRACE_SECONDS * FPS):
                owner = None
            reset_pending()
            reset_history()
            last_change = -10**9
            continue

        # La primera reanudación tras una pausa puede carecer de observación
        # del balón. Conservamos owner y exigimos soporte observado para
        # un switch precipitado durante los primeros frames reanudados.
        if dead_streak:
            post_dead_live = RAM_SWITCH_DEAD_EXIT_CONFIRM
        dead_streak = 0
        if post_dead_live > 0:
            post_dead_live -= 1
        active_flag[i] = 1
        c = candidate[i]
        if owner is None:
            if c in ("H", "A"):
                owner = c
                reasons[i] = "INITIAL_CONTROL_CANDIDATE"
            elif comparable[i] and np.isfinite(margin[i]):
                owner = "H" if margin[i] >= 0 else "A"
                reasons[i] = "INITIAL_NEAREST"
                uncertain[i] = 1
            else:
                owner = "H"
                reasons[i] = "INITIAL_NO_GEOMETRY"
                uncertain[i] = 1
            reset_pending()
            reset_history()
            states[i] = "CONTROL_" + owner
        else:
            # Si aparece un rival nuevo, empieza un nuevo expediente de cambio.
            # No acumulamos puntuacion de dos rivales distintos.
            rival_now = c in ("H", "A") and c != owner
            owner_now = c == owner
            challenger_evidence_trace[i] = int(rival_now)
            owner_evidence_trace[i] = int(owner_now)

            if rival_now and challenger != c:
                reset_pending()
                reset_history()
                challenger = c
                pending_start = i

            if challenger is None:
                if owner_now:
                    reasons[i] = "OWNER_CONTROL_EVIDENCE"
                    states[i] = "CONTROL_" + owner
                else:
                    reasons[i] = "NO_CONTROL_KEEP_OWNER" if comparable[i] else "NO_GEOMETRY_KEEP_OWNER"
                    states[i] = "LOOSE_AFTER_" + owner
                    uncertain[i] = 1
            else:
                # Ventana de evidencias, no de cambios de etiqueta.
                recent_rival[ptr] = int(rival_now and c == challenger)
                recent_owner[ptr] = int(owner_now)
                ptr = (ptr + 1) % SWITCH_LONG_WINDOW
                history_count = min(history_count + 1, SWITCH_LONG_WINDOW)

                # La ventana larga de control observa también la tendencia
                # del margen de distancia (solo si hay geometría comparable).
                # No presupone que un rebote implique recuperación.
                trend = 0.0
                if i >= 10 and relevant[i-10] and comparable[i-10] and np.isfinite(margin[i-10]) and np.isfinite(margin[i]):
                    delta = margin[i] - margin[i-10]
                    trend = float(np.clip((-delta if challenger == "A" else delta) / 3.0, -1.0, 1.0))
                if rival_now:
                    gap = 0
                    pending_frames += 1
                    if source[i] == "OBSERVED":
                        pending_observed += 1
                        increment = 1.0
                    elif source[i] == "IMPUTED":
                        increment = 0.62
                    else:
                        increment = 0.42
                    # Ventaja grande y comparabilidad real: un poco mas de peso,
                    # sin duplicar por ello el numero de frames de evidencia.
                    if np.isfinite(margin[i]) and abs(margin[i]) >= 2.0:
                        increment += 0.15
                    # Tendencia consistente suma poco; proximidad puntual
                    # sin ganancia temporal no basta para aportar esta prueba.
                    increment += max(0., trend) * .18
                    score = min(SWITCH_MAX_SCORE, score + increment)
                elif owner_now:
                    gap += 1
                    score = max(0.0, score - 2.3)
                else:
                    gap += 1
                    score = max(0.0, score - 0.20)

                short_len = min(history_count, SWITCH_SHORT_WINDOW)
                short_idx = (np.arange(short_len) + ptr - short_len) % SWITCH_LONG_WINDOW
                long_len = history_count
                short_support = float(np.sum(recent_rival[short_idx]) / max(1, short_len))
                long_support = float(np.sum(recent_rival) / max(1, long_len))
                recent_owner_count = int(np.sum(recent_owner[short_idx]))
                cooldown = max(0, SWITCH_COOLDOWN_FRAMES - (i - last_change))

                # Una secuencia puramente imputada necesita mas pruebas.
                minimum = SWITCH_MIN_EVIDENCE_FRAMES if pending_observed >= 2 else SWITCH_IMPUTED_MIN_FRAMES
                threshold = SWITCH_SCORE_THRESHOLD if pending_observed >= 2 else SWITCH_IMPUTED_SCORE_THRESHOLD
                veto = recent_owner_count >= SWITCH_OWNER_VETO_FRAMES
                support_ok = (short_len == SWITCH_SHORT_WINDOW
                              and short_support >= SWITCH_SHORT_SUPPORT
                              and long_support >= SWITCH_LONG_SUPPORT)
                # Segunda escala: si el rival aparece justo después de
                # evidencia fuerte del propietario, esperar mayor corroboración.
                # Una tendencia negativa penaliza, no impone bloqueo absoluto.
                if rival_now and trend < -.5:
                    score = max(0., score - .35)
                # Penalización calibrable por evidencia imputada, bonificación
                # por observaciones reales. Ningún término lee etiquetas GT.
                observed_ratio = pending_observed / max(1, pending_frames)
                minimum = max(1, int(round(minimum + RAM_ADAPT_IMPUTED_PENALTY * (1.0-observed_ratio)
                                            - RAM_ADAPT_OBS_REWARD * observed_ratio)))
                ready = (pending_frames >= minimum
                         and score >= threshold
                         and support_ok
                         and not veto
                         and cooldown == 0
                         and (post_dead_live == 0 or observed_ratio >= .60)
                         and rival_now)
                switch_score_trace[i] = score
                short_support_trace[i] = short_support
                long_support_trace[i] = long_support
                owner_veto_trace[i] = int(veto)
                cooldown_trace[i] = cooldown
                pending_count_trace[i] = pending_frames
                pending_trace[i] = challenger
                switch_ready_trace[i] = int(ready)

                if ready:
                    old_owner = owner
                    owner = challenger
                    confirmed_changes.append({
                        "confirmation_frame": i,
                        "start_frame": pending_start,
                        "from": old_owner,
                        "to": owner,
                        "evidence_frames": pending_frames,
                        "observed_frames": pending_observed,
                        "switch_score": round(float(score), 3),
                        "short_support": round(float(short_support), 3),
                        "long_support": round(float(long_support), 3),
                    })
                    last_change = i
                    reasons[i] = "SWITCH_MULTIWINDOW_CONFIRMED"
                    states[i] = "CONTROL_" + owner
                    reset_pending()
                    reset_history()
                else:
                    if veto:
                        reasons[i] = "SWITCH_OWNER_VETO"
                        blocked_by_veto += 1
                    elif cooldown:
                        reasons[i] = "SWITCH_COOLDOWN"
                    elif pending_frames >= minimum and not support_ok:
                        reasons[i] = "SWITCH_WINDOW_SUPPORT_LOW"
                        blocked_by_support += 1
                    else:
                        reasons[i] = "SWITCH_ACCUMULATING"
                    states[i] = "CONTESTED_AFTER_" + owner if rival_now else "LOOSE_AFTER_" + owner
                    uncertain[i] = 1

                # Un candidato que desaparece no deja su score almacenado
                # indefinidamente. El owner corroborado provoca reset rapido.
                if not ready and (gap > MAX_CANDIDATE_GAP_FRAMES or recent_owner_count >= SWITCH_OWNER_VETO_FRAMES):
                    reset_pending()
                    reset_history()

        labels[i] = owner
        possession_flag[i] = 1 if owner == "H" else 0

    return {
        "labels": labels, "reasons": reasons, "states": states,
        "active": active_flag, "possession_flag": possession_flag,
        "uncertain": uncertain, "pending": pending_trace,
        "pending_count": pending_count_trace, "changes": confirmed_changes,
        "switch_score": switch_score_trace,
        "switch_support_10": short_support_trace,
        "switch_support_25": long_support_trace,
        "switch_owner_veto": owner_veto_trace,
        "switch_cooldown": cooldown_trace,
        "switch_challenger_evidence": challenger_evidence_trace,
        "switch_owner_evidence": owner_evidence_trace,
        "switch_ready": switch_ready_trace,
        "switch_veto_count": blocked_by_veto,
        "switch_low_support_count": blocked_by_support,
    }


# ============================================================
# 5. SUAVIZADO RETROSPECTIVO
# ============================================================

def smooth_boundaries(result, relevant, candidate):
    """Retroceso condicionado por evidencia: nunca rellena 20 frames a ciegas."""
    labels = result["labels"].copy()
    reasons = result["reasons"].copy()
    corrections = 0
    changed_frames = 0
    for change in result["changes"]:
        confirmation = change["confirmation_frame"]
        first = change["start_frame"]
        old = change["from"]
        new = change["to"]
        if first is None:
            continue
        left = max(first, confirmation - MAX_BACKTRACK_FRAMES + 1)
        if left >= confirmation:
            continue
        # Buscar un prefijo cuya porcion restante este suficientemente
        # sustentada por observaciones de control rival; no invadir zonas D.
        boundary = None
        for j in range(left, confirmation):
            if not np.all(relevant[j:confirmation + 1]):
                break
            if not np.all(labels[j:confirmation] == old):
                break
            support = np.mean(candidate[j:confirmation + 1] == new)
            if support >= SWITCH_BACKTRACK_SUPPORT:
                boundary = j
                break
        if boundary is None:
            continue
        labels[boundary:confirmation] = new
        reasons[boundary:confirmation] = "RETROSPECTIVE_SWITCH_SUPPORTED"
        corrections += 1
        changed_frames += confirmation - boundary
    # Corregir islas H-A-H / A-H-A solo si ambos vecinos concuerdan y la
    # propia isla carece de control rival consistente. Nunca pisar D/X.
    island_corrections = 0
    island_frames = 0
    max_island = int(round(RAM_ISLAND_MAX_SECONDS * FPS))
    if max_island > 0:
        # Una sola pasada sobre los segmentos H/A tras el retroceso.
        segments = [(a,b,str(labels[a])) for a,b in runs(np.isin(labels,["H","A"]))]
        for lo,hi,_ in segments:
            edges = np.r_[lo, np.flatnonzero(labels[lo+1:hi+1]!=labels[lo:hi])+lo+1, hi+1]
            for z in range(1,len(edges)-2):
                a,b=int(edges[z]),int(edges[z+1])
                prior,cur,nxt=labels[a-1],labels[a],labels[b]
                if prior != nxt or cur == prior or b-a>max_island:
                    continue
                support_neighbor=float(np.mean(candidate[a:b]==prior))
                oppose=float(np.mean(candidate[a:b]==cur))
                if support_neighbor>=RAM_ISLAND_NEIGHBOR_SUPPORT and oppose<=RAM_ISLAND_OPPOSING_SUPPORT:
                    labels[a:b]=prior
                    reasons[a:b]="ISLAND_TWO_SIDED_EVIDENCE"
                    island_corrections+=1
                    island_frames+=b-a
    result["island_corrections"] = island_corrections
    result["island_frames"] = island_frames
    result["labels"] = labels
    result["reasons"] = reasons
    possession = result["possession_flag"]
    possession[labels == "H"] = 1
    possession[labels == "A"] = 0
    possession[labels == "D"] = -1
    return corrections, changed_frames


# ============================================================
# 6. SALIDAS
# ============================================================

def save_outputs(
    result,
    evidence,
    candidate,
    home,
    away,
    margin,
    relevant,
    periods,
    corrections,
    changed_frames,
    event_diag=None,
):
    config.ensure_output_dirs()

    n = config.N_VIDEO_FRAMES

    submission_path = (
        config.RUNS_DIR
        / f"submission_{RUN_ID}.csv"
    )

    trace_path = (
        config.RUNS_DIR
        / f"decision_trace_{RUN_ID}.csv"
    )

    summary_path = (
        config.RUNS_DIR
        / f"run_summary_{RUN_ID}.json"
    )

    labels = result["labels"]

    submission = pd.DataFrame({
        "frame": np.arange(n),
        "label": labels,
    })

    submission.to_csv(
        submission_path,
        index=False
    )

    # G9: modo de búsqueda rápido. Las etiquetas se calculan COMPLETAS con el
    # mismo motor; se evita el volcado de decenas de MB de trazas por ensayo.
    # Para el campeón el orquestador reejecuta la MISMA receta en modo auditoría.
    if os.environ.get('RAM_G9_FAST_IO','0') == '1':
        trace_path.write_text('frame,final_label\n',encoding='utf-8')
        counts={k:int(np.sum(labels==k)) for k in ('H','A','D','X')}
        summary={
            'run_id':RUN_ID, 'game':config.GAME,'frames':n,'fps':FPS,
            'engine_version':'G9_fast_search_same_decisions',
            'fast_io_only':True,'labels':counts,
            'g8':{'episodes':evidence.get('g8_stats',{'enabled':False}),
                  'single_final_ha':bool(RAM_G8_SINGLE_HA_FINAL)},
            'g9':{'refinement':evidence.get('g9_stats',{'enabled':False})},
            'g6':{'motion':evidence.get('g6_motion_stats',{})},
            'g7':{'context':evidence.get('g7_context_stats',{}),
                  'joint':result.get('g7_joint_stats',{'enabled':False})},
            'files':{'submission':str(submission_path),'trace':str(trace_path)},
        }
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=_json_safe),encoding='utf-8')
        return submission_path,trace_path,summary_path,counts

    trace = pd.DataFrame({
        "frame": np.arange(n),
        "match_time": [
            config.format_match_time(i)
            for i in range(n)
        ],
        "period": periods,
        "flag_active": result["active"],
        "flag_possession": result["possession_flag"],
        "flag_active_unknown": (
            relevant
            & ~evidence["dead_confirmed"]
            & ~evidence["observed"]
        ).astype(np.int8),
        "flag_ball_observed": evidence[
            "observed"
        ].astype(np.int8),
        "flag_ball_imputed": evidence[
            "imputed"
        ].astype(np.int8),
        "flag_long_gap": evidence[
            "long_gap"
        ].astype(np.int8),
        "flag_outside_candidate": evidence[
            "outside"
        ].astype(np.int8),
        "flag_low_ball_speed": evidence[
            "low_ball"
        ].astype(np.int8),
        "flag_low_team_speed": evidence[
            "low_teams"
        ].astype(np.int8),
        "flag_set_piece": evidence[
            "set_piece"
        ].astype(np.int8),
        "flag_dead_confirmed": evidence[
            "dead_confirmed"
        ].astype(np.int8),
        "th_restart_votes": evidence["th_restart_votes"],
        "th_context_votes": evidence["th_context_votes"],
        "th_stop_strength": evidence["th_stop_strength"],
        "th_has_bounded_evidence": evidence["th_bounded"].astype(np.int8),
        "ball_reliability": evidence["reliability"],
        "dead_score": evidence["dead_score"],
        "live_score": evidence["live_score"],
        "activity_net_score": evidence["net_score"],
        "stop_accumulator": evidence["stop_accumulator"],
        "restart_accumulator": evidence["restart_accumulator"],
        "stop_support_10": evidence["stop_support_10"],
        "stop_support_25": evidence["stop_support_25"],
        "stop_support_50": evidence["stop_support_50"],
        "restart_support_5": evidence["restart_support_5"],
        "restart_support_10": evidence["restart_support_10"],
        "restart_support_25": evidence["restart_support_25"],
        "stop_evidence_age": evidence["stop_evidence_age"],
        "collective_valid": evidence["collective_valid"].astype(np.int8),
        "collective_shift": evidence["collective_shift"].astype(np.int8),
        "collective_touchline": evidence["collective_touchline"].astype(np.int8),
        "collective_stop": evidence["collective_stop"].astype(np.int8),
        "collective_live": evidence["collective_live"].astype(np.int8),
        "stop_raw": evidence["stop_raw"].astype(np.int8),
        "stop_anchor": evidence["stop_anchor"].astype(np.int8),
        "play_anchor": evidence["play_anchor"].astype(np.int8),
        "activity_emission": evidence["activity_emission"],
        "spatial_setup": evidence["spatial_setup"].astype(np.int8),
        "observed_flow": evidence["observed_flow"].astype(np.int8),
        "zone_touchline_both": evidence["zone_touchline_both"].astype(np.int8),
        "zone_compact_both": evidence["zone_compact_both"].astype(np.int8),
        "zone_reorganization": evidence["zone_reorganization"].astype(np.int8),
        "tactical_valid": evidence["tactical_valid"].astype(np.int8),
        "tactical_reliability": evidence["tactical_reliability"],
        "tactical_overlap": evidence["tactical_overlap"],
        "tactical_change_25": evidence["tactical_change_25"],
        "tactical_change_50": evidence["tactical_change_50"],
        "tactical_moving_mean": evidence["tactical_moving_mean"],
        "tactical_slowdown": evidence["tactical_slowdown"].astype(np.int8),
        "tactical_shape_stable": evidence["tactical_shape_stable"].astype(np.int8),
        "tactical_collective_pause": evidence["tactical_collective_pause"].astype(np.int8),
        "tactical_congestion_idle": evidence["tactical_congestion_idle"].astype(np.int8),
        "tactical_reorganization": evidence["tactical_reorganization"].astype(np.int8),
        "tactical_flow": evidence["tactical_flow"].astype(np.int8),
        "restart_raw": evidence["restart_raw"].astype(np.int8),
        "activity_state": evidence["activity_state"],
        "flag_near_edge": evidence["near_edge"].astype(np.int8),
        "flag_near_corner": evidence["near_corner"].astype(np.int8),
        "flag_control_near": evidence["control_near"].astype(np.int8),
        "flag_fast_ball": evidence["fast_ball"].astype(np.int8),
        "flag_stop_candidate": evidence["stop_candidate"].astype(np.int8),
        "flag_restart_candidate": evidence["restart_candidate"].astype(np.int8),
        "flag_decision_uncertain": result[
            "uncertain"
        ],
        "ball_speed_mps": evidence[
            "ball_speed"
        ],
        "home_team_speed": evidence[
            "home_speed"
        ],
        "away_team_speed": evidence[
            "away_speed"
        ],
        "home_distance_m": home,
        "away_distance_m": away,
        "distance_margin_home_m": margin,
        "g5_control_radius_m": np.where(evidence['imputed'], CONTROL_RADIUS_M*RAM_G5_IMPUTED_RADIUS_FACTOR,
                            np.where(evidence['observed'], CONTROL_RADIUS_M*RAM_G5_OBSERVED_RADIUS_FACTOR,CONTROL_RADIUS_M)),
        "g5_control_advantage_m": CONTROL_ADVANTAGE_M + RAM_G5_IMPUTED_ADVANTAGE_EXTRA*evidence['imputed'].astype(float),
        "flag_g5_bridged_th": evidence.get('g5_bridge_mask',np.zeros(n,dtype=bool)).astype(np.int8),
        "g6_motion_slow_probability": evidence.get('g6_motion_probability',np.full(n,np.nan)),
        "flag_g6_motion_th_extension": evidence.get('g6_motion_added_mask',np.zeros(n,dtype=bool)).astype(np.int8),
        "flag_g7_context_support": evidence.get('g7_context_candidate',np.zeros(n,dtype=bool)).astype(np.int8),
        "g7_context_motion_probability": evidence.get('g7_context_motion_probability',np.full(n,np.nan)),
        "g7_context_origin": evidence.get('g7_context_origin',np.full(n,'NONE',dtype=object)),
        "flag_g7_context_added_D": evidence.get('g7_context_added_D',np.zeros(n,dtype=bool)).astype(np.int8),
        "flag_g7_joint_added_D": evidence.get('g7_joint_added_D',np.zeros(n,dtype=bool)).astype(np.int8),
        "flag_g7_joint_removed_D": evidence.get('g7_joint_removed_D',np.zeros(n,dtype=bool)).astype(np.int8),
        "flag_g7_joint_restart_verified": evidence.get('g7_joint_restart_confirmed',np.zeros(n,dtype=bool)).astype(np.int8),
        "flag_g8_episode_added_D": evidence.get('g8_trace',{}).get('added',np.zeros(n,bool)).astype(np.int8),
        "flag_g8_episode_seed": evidence.get('g8_trace',{}).get('seed',np.zeros(n,bool)).astype(np.int8),
        "flag_g8_verified_restart": evidence.get('g8_trace',{}).get('restart',np.zeros(n,bool)).astype(np.int8),
        "g8_episode_physical_score": evidence.get('g8_trace',{}).get('score',np.zeros(n)),
        "g8_episode_context_score": evidence.get('g8_trace',{}).get('context_score',np.zeros(n)),
        "g8_episode_type": evidence.get('g8_trace',{}).get('event_type',np.full(n,'NONE',dtype=object)),
        "g8_episode_state": evidence.get('g8_trace',{}).get('episode_state',np.full(n,'NONE',dtype=object)),
        "flag_g9_episode_added_D": evidence.get('g9_trace',{}).get('added',np.zeros(n,bool)).astype(np.int8),
        "flag_g9_edge_D": evidence.get('g9_trace',{}).get('edge',np.zeros(n,bool)).astype(np.int8),
        "flag_g9_bridge_D": evidence.get('g9_trace',{}).get('bridge',np.zeros(n,bool)).astype(np.int8),
        "g9_context_score": evidence.get('g9_trace',{}).get('context_score',np.zeros(n)),
        "g9_physical_score": evidence.get('g9_trace',{}).get('physical_score',np.zeros(n)),
        "control_candidate": candidate,
        "pending_team": result["pending"],
        "pending_frames": result["pending_count"],
        "switch_evidence_score": result["switch_score"],
        "switch_support_10": result["switch_support_10"],
        "switch_support_25": result["switch_support_25"],
        "switch_owner_veto": result["switch_owner_veto"],
        "switch_cooldown_frames": result["switch_cooldown"],
        "switch_rival_evidence": result["switch_challenger_evidence"],
        "switch_owner_evidence": result["switch_owner_evidence"],
        "switch_ready": result["switch_ready"],
        "internal_state": result["states"],
        "decision_reason": result["reasons"],
        "final_label": labels,
    })

    # S31 — exportar diagnosticos EXISTENTES, sin fabricar evidencia.
    # El productor S31 evoluciona por version: algunas familias no existen.
    # La taxonomia completa se guarda mas abajo en event_context_{RUN_ID}.csv.
    if event_diag is not None:
        tax = event_diag.frames
        if len(tax) != n:
            raise ValueError(f"S31: {len(tax)} filas vs {n} frames")
        if "frame" not in tax.columns:
            raise ValueError("S31: falta columna frame; no se puede alinear")
        frame_ids = pd.to_numeric(tax["frame"], errors="coerce").to_numpy()
        if not np.array_equal(frame_ids, np.arange(n)):
            raise ValueError("S31: indices de frame no alineados con el video")
        requested = (
            "event_primary_type", "event_primary_score", "event_confidence_tier",
            "event_episode_id", "event_episode_type",
            "event_distinct_evidence_families", "event_observed_flow", "event_team_idle",
            "event_team_idle_50", "event_team_motion_25", "event_active_team_motion_25",
            "event_temporal_collective_regime_shift", "event_temporal_collective_hold",
            "event_temporal_collective_return",
        )
        missing_optional = [name for name in requested if name not in tax.columns]
        available = [name for name in requested if name in tax.columns]
        if missing_optional:
            print("[S31] Campos de diagnostico no exportados (ausentes en productor):",
                  ", ".join(missing_optional))
        for name in available:
            trace[name] = tax[name].to_numpy(copy=True)
        print(f"[S31] Diagnosticos alineados exportados: {len(available)} de {len(requested)}")

    trace.to_csv(
        trace_path,
        index=False
    )

    counts = {
        label: int(np.sum(labels == label))
        for label in ("H", "A", "D", "X")
    }

    summary = {
        'g6': { 'motion': evidence.get('g6_motion_stats',{'enabled':False})},
        'g7': { 'context':evidence.get('g7_context_stats',{}),
                'joint':result.get('g7_joint_stats',{'enabled':False}),
                'touch_discount':RAM_G7_TOUCH_DISCOUNT},
        'g5': {
            'imputed_radius_factor':RAM_G5_IMPUTED_RADIUS_FACTOR,
            'imputed_advantage_extra_m':RAM_G5_IMPUTED_ADVANTAGE_EXTRA,
            'observed_radius_factor':RAM_G5_OBSERVED_RADIUS_FACTOR,
            'th_bridge':evidence.get('g5_bridge_stats',{}),
        },
        'g9': {'refinement':evidence.get('g9_stats',{'enabled':False})},
        'g8': {'episodes':evidence.get('g8_stats',{'enabled':False}),
               'single_final_ha':bool(RAM_G8_SINGLE_HA_FINAL)},
        'g3': {'global_ha': result.get('global_ha_stats',{}),
               'th_follow': evidence.get('th_follow_stats',{}),
               'th_physics_live_weight':RAM_TH_PHYSICS_LIVE_WEIGHT},
        "run_id": RUN_ID,
        "game": config.GAME,
        "engine_version": "G9_fast_evidence_before_possession",
        "frames": n,
        "fps": FPS,
        "labels": counts,
        "confirmed_recoveries": len(result["changes"]),
        "switch_owner_veto_events": result["switch_veto_count"],
        "switch_low_support_events": result["switch_low_support_count"],
        "switch_confirmed_details": result["changes"],
        "smoothed_boundaries": corrections,
        "smoothed_frames": changed_frames,
        "dead_frames_inside_game": int(
            np.sum(
                relevant & (labels == "D")
            )
        ),
        "observed_outside_frames": int(
            np.sum(evidence["outside"])
        ),
        "long_imputation_frames": int(
            np.sum(evidence["long_gap"])
        ),
        "activity_v5": {
            "model": "bidirectional_anchor_hysteresis",
            "windows_frames": [5, 10, 25, 50, 125],
            "stop_acc_threshold": "replaced_by_anchor_confirmation",
            "restart_acc_threshold": "replaced_by_play_anchor_confirmation",
            "collective_coverage_frames": int(np.sum(evidence["collective_valid"] & relevant)),
            "collective_stop_frames": int(np.sum(evidence["collective_stop"] & relevant)),
            "collective_live_frames": int(np.sum(evidence["collective_live"] & relevant)),
            "tactical_valid_frames": int(np.sum(evidence["tactical_valid"] & relevant)),
            "tactical_collective_pause_frames": int(np.sum(evidence["tactical_collective_pause"] & relevant)),
            "tactical_flow_frames": int(np.sum(evidence["tactical_flow"] & relevant)),
            "stop_events": evidence["stop_events"],
            "high_certainty_stop_frames": int(evidence["stop_anchor_count"]),
            "high_certainty_live_frames": int(evidence["play_anchor_count"]),
            "restart_events": evidence["restart_events"],
            "uses_ground_truth": False,
        },
        "parameters": {
            "control_radius_m": CONTROL_RADIUS_M,
            "control_advantage_m": CONTROL_ADVANTAGE_M,
            "switch_min_evidence_frames": SWITCH_MIN_EVIDENCE_FRAMES,
            "switch_imputed_min_frames": SWITCH_IMPUTED_MIN_FRAMES,
            "switch_short_window": SWITCH_SHORT_WINDOW,
            "switch_long_window": SWITCH_LONG_WINDOW,
            "switch_short_support": SWITCH_SHORT_SUPPORT,
            "switch_long_support": SWITCH_LONG_SUPPORT,
            "switch_score_threshold": SWITCH_SCORE_THRESHOLD,
            "switch_imputed_score_threshold": SWITCH_IMPUTED_SCORE_THRESHOLD,
            "switch_owner_veto_frames": SWITCH_OWNER_VETO_FRAMES,
            "switch_cooldown_frames": SWITCH_COOLDOWN_FRAMES,
            "switch_backtrack_support": SWITCH_BACKTRACK_SUPPORT,
            "candidate_gap_tolerance":
                MAX_CANDIDATE_GAP_FRAMES,
            "backtrack_frames":
                MAX_BACKTRACK_FRAMES,
            "outside_confirm_frames":
                OUTSIDE_CONFIRM_FRAMES,
            "stop_confirm_frames":
                STOP_CONFIRM_FRAMES,
        },
        "files": {
            "submission": str(submission_path),
            "trace": str(trace_path),
        },
        "limitations": [
            "Dead-ball boundaries can remain uncertain.",
            "No ground truth used.",
            "No official spellF1 calculated.",
            "Long imputed intervals remain uncertain.",
        ],
    }


    switch_path = config.RUNS_DIR / f"possession_switches_{RUN_ID}.csv"
    switch_cols = ["confirmation_frame", "confirmation_time", "start_frame", "start_time", "from", "to", "evidence_frames", "observed_frames", "switch_score", "short_support", "long_support", "retrospective_start", "retrospective_frames"]
    switch_rows = []
    for change in result["changes"]:
        confirmation = change["confirmation_frame"]
        start = change["start_frame"]
        back = confirmation
        while back > 0 and result["reasons"][back - 1] == "RETROSPECTIVE_SWITCH_SUPPORTED":
            back -= 1
        switch_rows.append({
            **change,
            "confirmation_time": config.format_match_time(confirmation),
            "start_time": config.format_match_time(start),
            "retrospective_start": back,
            "retrospective_frames": confirmation - back,
        })
    pd.DataFrame(switch_rows, columns=switch_cols).to_csv(switch_path, index=False)
    summary["files"]["possession_switches"] = str(switch_path)
    print("Switch audit:", switch_path)

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            default=_json_safe
        ),
        encoding="utf-8"
    )

    activity_path = config.RUNS_DIR / f"activity_intervals_{RUN_ID}.csv"
    activity_rows = []
    for a, b in runs(relevant & (labels == "D")):
        activity_rows.append({"frame_start": a, "frame_end": b, "start_time": config.format_match_time(a), "end_time": config.format_match_time(b), "duration_s": round((b-a+1)/FPS, 2), "start_reason": evidence["dead_reason"][a], "mean_dead_score": round(float(np.mean(evidence["dead_score"][a:b+1])), 3), "mean_ball_reliability": round(float(np.mean(evidence["reliability"][a:b+1])), 3)})
    pd.DataFrame(activity_rows, columns=["frame_start", "frame_end", "start_time", "end_time", "duration_s", "start_reason", "mean_dead_score", "mean_ball_reliability"]).to_csv(activity_path, index=False)
    print("Activity intervals:", activity_path)
    return submission_path, trace_path, summary_path, counts


# ============================================================
# MAIN
# ============================================================

def run(th_evidence_path):
    print("=" * 65)
    print("Motor de posesión — TH + contexto físico + H/A/D")
    print("=" * 65)

    v4, s27 = load_inputs()

    n = config.N_VIDEO_FRAMES

    periods = np.array([
        config.get_period(i)
        for i in range(n)
    ])

    relevant = np.isin(
        periods,
        ["FIRST_HALF", "SECOND_HALF"]
    )

    s27_relevant = bool_column(
        s27, "relevant"
    )

    if not np.array_equal(
        relevant, s27_relevant
    ):
        raise ValueError(
            "La mascara S27 no coincide con config.py"
        )

    print("[2] Validando familias de señales...")
    signal_audit = make_signal_audit(v4, s27, relevant)
    config.ensure_output_dirs()
    audit_path = config.RUNS_DIR / f"signal_audit_{RUN_ID}.csv"
    signal_audit.to_csv(audit_path, index=False)
    print("Auditoria:", audit_path)
    print("[3] Construyendo evidencia y contextos S31 antes del motor H/A...")
    evidence = build_dead_ball_evidence(v4, s27, relevant)
    event_diagnosis = detect_dead_ball_types(v4, s27, relevant, game=config.GAME, fps=FPS)
    print("[S31] Hipotesis:", len(event_diagnosis.episodes),
          "SmartTag:", event_diagnosis.status["raw_smart_tagging_status"])
    # TH01–TH15 reemplazan V5 (nunca viceversa).
    evidence, th = th_primary_activity(evidence, th_evidence_path, relevant, v4=v4)
    if RAM_G8_EPISODES_ENABLED:
        gmm=None
        if RAM_G8_GMM_ENABLED:
            gmm,_,gmm_fit=g6_fit_motion_regimes(v4,relevant,
                components=RAM_G8_GMM_COMPONENTS,min_coverage=RAM_G6_MOTION_MIN_COVERAGE)
        else:gmm_fit=[]
        new_dead,g8_trace,g8_stats=episode_evidence(event_diagnosis.frames,evidence,relevant,
            th_masks=evidence.get('g8_th_masks'),types=RAM_G8_TYPES,
            threshold=RAM_G8_CONTEXT_THRESHOLD,support=RAM_G8_PHYSICAL_THRESHOLD,
            seed_seconds=RAM_G8_SEED_SECONDS,patience_seconds=RAM_G8_PATIENCE_SECONDS,
            max_episode_seconds=RAM_G8_MAX_EPISODE_SECONDS,
            restart_seconds=RAM_G8_RESTART_SECONDS,fps=FPS,
            gmm_probability=gmm,gmm_weight=(RAM_G8_GMM_WEIGHT if RAM_G8_GMM_ENABLED else 0.),
            s31_enabled=bool(RAM_G8_S31_ENABLED),th_enabled=bool(RAM_G8_TH_CONTEXT_ENABLED),
            hold_factor=RAM_G9_HOLD_FACTOR,entry_fraction=RAM_G9_ENTRY_FRACTION,
            physical_window_seconds=RAM_G9_PHYSICAL_WINDOW_SECONDS,
            restart_threshold=RAM_G9_RESTART_THRESHOLD,th_near_seconds=RAM_G9_TH_NEAR_SECONDS,
            minimum_seed_seconds=RAM_G9_MIN_SEED_SECONDS)
        g8_stats['gmm_fits']=gmm_fit
        evidence['dead_confirmed']=new_dead
        evidence['dead_reason'][g8_trace['added']]='G8_S31_TH_EPISODE_RESTART_HYSTERESIS'
        evidence['activity_state'][g8_trace['added']]='DEAD'
        evidence['stop_events']=len(runs(new_dead&relevant))
        evidence['restart_events']=sum(1 for a,b in runs(new_dead&relevant) if b+1<n and relevant[b+1])
        evidence['g8_trace']=g8_trace
        evidence['g8_stats']=g8_stats
        print('[G8] episodios con S31/TH:',g8_stats)
    else:
        evidence['g8_stats']={'enabled':False,'new_D_frames':0,'new_episodes':0}

    if RAM_G9_REFINE_ENABLED:
        current_dead=np.asarray(evidence['dead_confirmed'],bool)
        refined,g9_trace,g9_stats=refine_dead(current_dead,relevant,event_diagnosis.frames,evidence,
            types=RAM_G9_REFINE_TYPES,context_threshold=RAM_G9_REFINE_CONTEXT,
            physics_threshold=RAM_G9_REFINE_PHYSICS,max_edge_seconds=RAM_G9_REFINE_EDGE_SECONDS,
            max_bridge_seconds=RAM_G9_REFINE_BRIDGE_SECONDS,
            patience_seconds=RAM_G9_REFINE_PATIENCE_SECONDS,
            restart_seconds=RAM_G9_REFINE_RESTART_SECONDS,fps=FPS)
        evidence['dead_confirmed']=refined
        evidence['dead_reason'][g9_trace['added']]='G9_S31_PHYSICS_EDGE_OR_BRIDGE'
        evidence['activity_state'][g9_trace['added']]='DEAD'
        evidence['stop_events']=len(runs(refined&relevant))
        evidence['restart_events']=sum(1 for a,b in runs(refined&relevant) if b+1<n and relevant[b+1])
        evidence['g9_trace']=g9_trace
        evidence['g9_stats']=g9_stats
        print('[G9] reparación de episodios:',g9_stats)
    else:
        evidence['g9_stats']={'enabled':False,'new_D_frames':0}

    print("[4] Construyendo candidatos H/A...")

    candidate, home, away, margin, comparable = (
        build_candidates(s27, relevant, v4=v4)
    )

    print("[5] Ejecutando motor de estados...")

    result = decide_possession(
        relevant,
        periods,
        s27,
        evidence,
        candidate,
        margin,
        comparable,
    )

    print("[6] Suavizando fronteras...")

    if RAM_G8_SINGLE_HA_FINAL:
        # Solo UN decodificador final offline H/A. La maquina causal de
        # posesion calcula el prior; evitamos el suavizado retrospectivo extra.
        corrections,changed_frames=0,0
        g3_stats=decode_global_ha(result,relevant,candidate,evidence,enabled=True)
    else:
        corrections, changed_frames = smooth_boundaries(result, relevant, candidate)
        g3_stats=decode_global_ha(result,relevant,candidate,evidence)
    if RAM_G7_JOINT_ENABLED:
        print('[G7] Decodificador conjunto H/A/D:',g7_joint_decode(
            result,relevant,candidate,evidence,
            ha_cost=RAM_G7_JOINT_INERTIA_HA,
            enter_d=RAM_G7_JOINT_ENTER_D,
            exit_d=RAM_G7_JOINT_EXIT_D,
            base_prior=RAM_G7_JOINT_BASE_PRIOR,
            context_reward=RAM_G7_JOINT_CONTEXT_REWARD,
            unverified_exit_extra=RAM_G7_JOINT_UNVERIFIED_EXIT_EXTRA))
    if RAM_GLOBAL_HA_ENABLED:
        print('[G3] decodificador global H/A:',g3_stats)
    if RAM_TH_FOLLOW_PRE_SECONDS>0 or RAM_TH_FOLLOW_POST_SECONDS>0:
        print('[G3] seguimiento D TH:',evidence['th_follow_stats'])

    labels = result["labels"]

    if not np.isin(
        labels, ["H", "A", "D", "X"]
    ).all():
        raise ValueError(
            "Etiquetas invalidas"
        )

    if np.any(
        relevant & (labels == "X")
    ):
        raise ValueError(
            "Se genero X dentro del partido"
        )

    if len(labels) != n:
        raise ValueError(
            "Longitud incorrecta de submission"
        )

    print("[7] Guardando CSV y trazas...")

    (
        submission_path,
        trace_path,
        summary_path,
        counts,
    ) = save_outputs(
        result,
        evidence,
        candidate,
        home,
        away,
        margin,
        relevant,
        periods,
        corrections,
        changed_frames,
        event_diag=event_diagnosis,
    )

    # Registrar procedencia y efecto del TH dentro del RAM, no como parche.
    with open(summary_path, encoding="utf-8") as handle:
        integrated_summary=json.load(handle)
    integrated_summary["th_ram_integration"]={
        "mode":"TH_PRIMARY_ACTIVITY_REPLACES_LEGACY_DEAD_FLAG",
        "source":str(Path(th_evidence_path).resolve()),
        "accepted_by_detector":th["accepted"],
        "ignored_candidates":th["ignored"],
        "restart_vote_frames":int(np.sum(th["restart_votes"]>0)),
        "cross_detector_vote_frames":int(np.sum(th["restart_votes"]>=2)),
        "hysteresis":evidence["th_lock_stats"],
        "changes_are_before_possession_engine":True,
        "gt_used":False,
        "old_dead_flag_used":False,
        "primary_vs_old_dead_added":int(np.sum(evidence["dead_confirmed"] & ~evidence["th_old_dead_only_diagnostic"])),
        "primary_vs_old_dead_removed":int(np.sum(~evidence["dead_confirmed"] & evidence["th_old_dead_only_diagnostic"])),
        "rejection_reasons":th.get("rejection_reasons",{}),
    }
    with open(summary_path,"w",encoding="utf-8") as handle:
        json.dump(integrated_summary,handle,indent=2,ensure_ascii=False,default=_json_safe)

    if os.environ.get('RAM_G9_FAST_IO','0') != '1':
        # S31: guardar diagnóstico por evento y auditoría de disponibilidad.
        event_paths = {
            "context": config.RUNS_DIR / f"event_context_{RUN_ID}.csv",
            "episodes": config.RUNS_DIR / f"event_episodes_{RUN_ID}.csv",
            "type_summary": config.RUNS_DIR / f"event_type_summary_{RUN_ID}.csv",
            "source_audit": config.RUNS_DIR / f"event_signal_coverage_{RUN_ID}.csv",
            "smart_events": config.RUNS_DIR / f"event_raw_smart_tags_{RUN_ID}.csv",
        }
        event_diagnosis.frames.to_csv(event_paths["context"],index=False)
        event_diagnosis.episodes.to_csv(event_paths["episodes"],index=False)
        event_diagnosis.type_summary.to_csv(event_paths["type_summary"],index=False)
        event_diagnosis.sources.to_csv(event_paths["source_audit"],index=False)
        event_diagnosis.tag_records.to_csv(event_paths["smart_events"],index=False)
        # Actualizar resumen JSON ya escrito, sin afectar la submission.
        with open(summary_path,encoding="utf-8") as handle:
            summary_obj=json.load(handle)
        summary_obj["event_taxonomy_s31"] = {
            **event_diagnosis.status,
            "contract":"ETAPA_31_CONTRATO_DETECCION_INACTIVIDAD.md",
            "g8_used_before_possession":bool(RAM_G8_EPISODES_ENABLED and RAM_G8_S31_ENABLED),
            "mode":"AUDIT_ONLY_DOES_NOT_CHANGE_LABELS",
            "files":{name:str(path) for name,path in event_paths.items()},
        }
        with open(summary_path,"w",encoding="utf-8") as handle:
            json.dump(summary_obj,handle,indent=2,ensure_ascii=False,default=_json_safe)
        print("Auditoria S31:",event_paths["type_summary"])
        print("Episodios S31:",event_paths["episodes"])

    else:
        print('[G9] FAST IO: omitidas trazas/S31 masivas, submission intacta')

    print("\n========== MOTOR DE POSESIÓN + SEÑALES S31 ==========")
    print(f"RUN_ID: {RUN_ID}")
    print(f"Frames: {n:,}")

    for label, total in counts.items():
        print(f"{label}: {total:,}")

    print(
        "Recuperaciones confirmadas:",
        len(result["changes"])
    )

    print("Cambios frenados por veto:", result["switch_veto_count"])
    print("Ventanas con soporte insuficiente:", result["switch_low_support_count"])
    print(
        "Fronteras suavizadas:",
        corrections
    )

    print(
        "Frames corregidos:",
        changed_frames
    )

    print(
        "D dentro del juego:",
        int(np.sum(
            relevant & (labels == "D")
        ))
    )

    print(
        "Huecos largos imputados:",
        int(np.sum(evidence["long_gap"]))
    )

    print("Eventos de parada detectados:", evidence["stop_events"])
    print("Eventos de reanudacion detectados:", evidence["restart_events"])
    print("Cobertura geometria colectiva:", int(np.sum(evidence["collective_valid"] & relevant)))
    print("Señales de pausa colectiva:", int(np.sum(evidence["collective_stop"] & relevant)))
    print("Anclas confirmadas D / juego:", evidence["stop_anchor_count"], "/", evidence["play_anchor_count"])
    print("\nSubmission:", submission_path)
    print("Trace:", trace_path)
    print("Summary:", summary_path)



def self_test_th_primary():
    """Pruebas sintéticas sin matrices ni etiquetas de referencia."""
    from tempfile import TemporaryDirectory
    n=220
    relevant=np.ones(n,dtype=bool)
    def dummy(old_dead=False):
        return {'dead_confirmed':np.full(n,old_dead,dtype=bool),
                'observed':np.zeros(n,dtype=bool),
                'activity_state':np.full(n,'ACTIVE',dtype=object),
                'dead_reason':np.full(n,'DIAGNOSTIC',dtype=object)}
    cols=['detector','tier','evidence','start_low','start_high',
          'end_low','end_high','anchor_frame']
    with TemporaryDirectory() as folder:
        path=Path(folder)/'evidencia.csv'
        pd.DataFrame([['TH01','SEQUENCE_CANDIDATE','INWARD_RETURN_OBSERVED',
                       45,50,110,115,50],
                      ['TH11','STRONG_PATTERN','STOP_CONTEXT',
                       45,55,105,113,55]],columns=cols).to_csv(path,index=False)
        a,_=th_primary_activity(dummy(False),path,relevant)
        assert a['dead_confirmed'][75], 'TH01 debe generar D sin bandera anterior'
        assert not a['dead_confirmed'][180], 'La pausa debe terminar'
        b,_=th_primary_activity(dummy(True),path,relevant)
        assert np.array_equal(a['dead_confirmed'],b['dead_confirmed']), \
            'La bandera antigua no puede influir en la nueva prediccion'
        pd.DataFrame([['TH11','STRONG_PATTERN','INJURY_ONLY',
                       45,50,110,115,50]],columns=cols).to_csv(path,index=False)
        c,_=th_primary_activity(dummy(True),path,relevant)
        assert not np.any(c['dead_confirmed']), \
            'Contexto sin reanudacion no debe crear D'
    print('SELF TEST OK: flag TH independiente, contexto no activa D, fin de episodio')

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="RAM G7 único: TH01–TH15, contexto físico y decodificador H/A/D opcionales")
    parser.add_argument("--self-test",action="store_true",help="Pruebas sintéticas TH sin matrices")
    parser.add_argument("--th-evidence", required=False,
        help="Ruta a evidencia_th_normalizada.csv generada por integrar_solo_th.py")
    args=parser.parse_args()
    if args.self_test:
        self_test_th_primary()
        raise SystemExit(0)
    if not args.th_evidence:
        parser.error("Se requiere --th-evidence para la corrida completa")
    input_path=Path(args.th_evidence).expanduser()
    if not input_path.is_file():
        parser.error(f"No existe archivo TH: {input_path}")
    run(input_path)
