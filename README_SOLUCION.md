# Team Ball Possession Solution — FC Barcelona Hackathon 2027

This directory contains **only our team's code** for ball possession segmentation. It is designed to be integrated into a copy of the [official competition repository](https://github.com/metrica-sports/more-than-a-hack-2027), while keeping all original competition files unchanged.

## Overview

The solution follows a rule-based football analytics pipeline:

1. **Data ingestion and standardization:** Reads and aligns player and ball tracking data, homography, and Smart Tagging annotations provided by the competition organizers.
2. **Feature engineering:** Builds families of geometric, temporal, spatial, and movement-related features, including the reconstruction of missing ball observations.
3. **Dead-ball detection:** Analyzes interruptions in play using 15 football-specific detectors (`juego_parado/th01_*.py` through `th15_*.py`).
4. **Possession inference:** Distinguishes between active and inactive play using football-specific conditions and temporal continuity. Assigns one of four labels:
   - **H:** FC Barcelona possession.
   - **A:** Athletic Club possession.
   - **D:** Ball out of play.
   - **X:** No decision.
5. **Submission generation:** Produces one prediction per tactical video frame, resulting in **199,319 rows**.

The solution's decision rules and thresholds were reviewed by the team using football expertise and sports analytics experience.

## Execution — One Command

### Requirements

Use an original copy of the competition repository without modifying `games.py`, `starter/`, `evaluation/`, or the official dependencies.

The official datasets must be located according to [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md), typically under `data/tactical/`, or configured using the `MTAH_DATA_DIR` environment variable.

### Step 1 — Add the Team's Code

Extract the solution package into the **root directory of the official repository**, alongside `games.py`.

The package does not include or replace any files provided by the competition organizers.

### Step 2 — Install Dependencies

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-equipo.txt
```

### Step 3 — Generate the Final Submission

Run the following command:

```bash
python generar_entrega.py
```

This command automatically executes the required processing stages, including the 15 dead-ball detectors and the final possession inference engine.

**No additional scripts or external evaluation tools need to be executed.**

### Output

The generated submission file is saved at:

```text
local_outputs/entrega/posesion_fc_barcelona.csv
```

The CSV is not written to the repository root.

It contains the following columns:

```csv
frame,label
0,H
1,H
2,H
```

*The rows
