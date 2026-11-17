
from pathlib import Path

# ============================================================
# PROYECTO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
GAME = "BAR-ATH"

# ============================================================
# VIDEO
# ============================================================

FPS = 25
N_VIDEO_FRAMES = 199319

# ============================================================
# LIMITES OFICIALES DE LOS PERIODOS (FRAMES DE VIDEO)
# ============================================================

FIRST_HALF_START = 11185
FIRST_HALF_END = 90651

HALFTIME_START = 90652
HALFTIME_END = 110807

SECOND_HALF_START = 110808
SECOND_HALF_END = 190197

# ============================================================
# SALIDAS LOCALES
# ============================================================

OUTPUT_DIR = ROOT_DIR / "local_outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
EVALUATIONS_DIR = OUTPUT_DIR / "evaluations"

# ============================================================
# FUNCIONES TEMPORALES
# ============================================================

def format_match_time(frame: int) -> str:
    """
    Devuelve el reloj del partido en MM:SS.

    Primer tiempo:
        00:00 desde FIRST_HALF_START.

    Segundo tiempo:
        45:00 desde SECOND_HALF_START.

    Descanso:
        HT.

    Antes y después:
        PRE / FT.
    """

    if frame < FIRST_HALF_START:
        return "PRE"

    if FIRST_HALF_START <= frame <= FIRST_HALF_END:
        elapsed_seconds = (frame - FIRST_HALF_START) // FPS

    elif HALFTIME_START <= frame <= HALFTIME_END:
        return "HT"

    elif SECOND_HALF_START <= frame <= SECOND_HALF_END:
        elapsed_seconds = (
            45 * 60
            + (frame - SECOND_HALF_START) // FPS
        )

    else:
        return "FT"

    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


def get_period(frame: int) -> str:
    """Identifica el periodo al que pertenece un frame."""

    if frame < FIRST_HALF_START:
        return "PRE_MATCH"

    if frame <= FIRST_HALF_END:
        return "FIRST_HALF"

    if frame <= HALFTIME_END:
        return "HALFTIME"

    if frame <= SECOND_HALF_END:
        return "SECOND_HALF"

    return "POST_MATCH"


def ensure_output_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
