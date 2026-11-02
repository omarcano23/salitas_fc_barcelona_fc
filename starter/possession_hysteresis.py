"""A conservative possession predictor built on the supplied ATD tracking.

Unlike the reference baseline, this does not call every missing-ball frame dead ball.
It keeps a team-level possession estimate across short tracking gaps and requires
repeated, clear evidence before changing the team. This is a transparent heuristic,
not a learned or ground-truth-tuned model.

Run from the repository root:
    python starter/possession_hysteresis.py

The default output is evaluation/results/possession_hysteresis.csv.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.possession_metrics import H
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402


RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
HOME, AWAY = 0, 1
UNKNOWN = -1

# These are physical heuristics, not values selected by searching the public labels.
MAX_CONTROL_DISTANCE_M = 3.0
SUPPORT_RADIUS_M = 4.0
MIN_SCORE_MARGIN = 0.12

# A team change needs repeated control evidence, but exceptionally clear evidence
# can count more strongly than a marginal nearest-player win.
SWITCH_EVIDENCE_REQUIRED = 4.0
SWITCH_EVIDENCE_SPAN_FRAMES = 20  # evidence must arrive within 0.8 seconds

# Belief in a team fades gradually when observations become uncertain. This gives
# a longer bridge for gaps without carrying a stale label forever.
GAP_CONFIDENCE_DECAY = 0.99
MIN_OUTPUT_CONFIDENCE = 0.15
SMOOTH_RADIUS_FRAMES = 125  # vote over the current frame and up to 2 s on either side
BALL_SLOW_MPS = 0.35
PLAYER_GROUP_SLOW_MPS = 0.55
INACTIVITY_TO_DEAD_FRAMES = 20  # 0.8 seconds of still players while ball is missing


def team_of_each_track(meta_path):
    """Return the team assigned to each ATD slot by the metadata XML."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    players = list(root.iter("Player"))
    if len(teams) != 2 or not players:
        raise ValueError("Expected two teams and player track slots in {}".format(meta_path))
    home_id = teams[0].get("id")
    return np.asarray([HOME if player.get("teamId") == home_id else AWAY
                       for player in players], dtype=np.int8)


def _object(group, length, width):
    """Parse one x,y,speed ATD group; coordinates are normalized 0..1."""
    if not group or group[0] == "N":
        return None
    x_text, y_text, speed_text = group.split(",", 2)
    speed = float(speed_text) if speed_text[0] != "N" else np.nan
    return float(x_text), float(y_text), speed


def tracking_observations(raw_path, team_by_track, n_frames, pitch):
    """Extract proximity, nearby support, speed, and out-of-pitch evidence.

    Team minima avoid relying on stable player identities: the tracker slots may
    fragment or switch people, while each slot still has a team assignment. Speed
    is only a weak clue because ATD gives speed magnitude, not movement direction.
    """
    length, width = pitch
    distances = np.full((n_frames, 2), np.nan, dtype=np.float32)
    nearest_speed = np.full((n_frames, 2), np.nan, dtype=np.float32)
    support = np.zeros((n_frames, 2), dtype=np.uint8)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    ball_visible = np.zeros(n_frames, dtype=bool)
    ball_outside = np.zeros(n_frames, dtype=bool)
    group_speed = np.full(n_frames, np.nan, dtype=np.float32)
    with open(raw_path) as tracking:
        for line in tracking:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD starts at frame_id 1; CSV starts at 0.
            if frame < 0 or frame >= n_frames:
                continue
            ball = _object(parts[2], length, width)
            if ball is not None:
                bx_n, by_n, ball_speed[frame] = ball
                ball_visible[frame] = True
                ball_outside[frame] = (bx_n < -0.01 or bx_n > 1.01
                                       or by_n < -0.01 or by_n > 1.01)
                bx, by = bx_n * length, by_n * width
            team_min = [float("inf"), float("inf")]
            team_nearest_speed = [np.nan, np.nan]
            team_support = [0, 0]
            player_speeds = []
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= len(team_by_track):
                    break
                player = _object(group, length, width)
                if player is None:
                    continue
                px_n, py_n, speed = player
                if np.isfinite(speed):
                    player_speeds.append(speed)
                if ball is None:
                    continue
                dx = px_n * length - bx
                dy = py_n * width - by
                distance = (dx * dx + dy * dy) ** 0.5
                team = int(team_by_track[slot])
                if distance <= SUPPORT_RADIUS_M:
                    team_support[team] += 1
                if distance < team_min[team]:
                    team_min[team] = distance
                    team_nearest_speed[team] = speed
            for team in (HOME, AWAY):
                if ball is not None and np.isfinite(team_min[team]):
                    distances[frame, team] = team_min[team]
                    nearest_speed[frame, team] = team_nearest_speed[team]
                    support[frame, team] = min(team_support[team], 3)
            if player_speeds:
                group_speed[frame] = float(np.median(player_speeds))
    return distances, nearest_speed, support, ball_speed, ball_visible, ball_outside, group_speed


def distance_trend_bonus(distances):
    """Give a small movement bonus when a team's nearest distance is closing."""
    bonus = np.zeros_like(distances, dtype=np.float32)
    last_distance = [np.nan, np.nan]
    last_frame = [-1, -1]
    for frame, row in enumerate(distances):
        for team, distance in enumerate(row):
            if not np.isfinite(distance):
                continue
            # Distances are team minima, so this trend is collective team movement,
            # not an identity claim about one temporary track slot.
            if (np.isfinite(last_distance[team]) and frame - last_frame[team] <= 10
                    and last_distance[team] - distance >= 0.25):
                bonus[frame, team] = 0.08
            last_distance[team] = float(distance)
            last_frame[team] = frame
    return bonus


def team_evidence(distances, nearest_speed, support, ball_speed, approach_bonus):
    """Score each team's control evidence and nominate a clear candidate, if any.

    Closeness is the main signal. Nearby teammates and a team closing on the ball
    add support. Similar ball/player speed adds a small bonus because a controlled
    ball often travels with its player; it is weak since speed omits direction.
    """
    candidates = np.full(len(distances), UNKNOWN, dtype=np.int8)
    strength = np.zeros(len(distances), dtype=np.float32)
    for frame, (home_d, away_d) in enumerate(distances):
        if not (np.isfinite(home_d) or np.isfinite(away_d)):
            continue
        score = np.zeros(2, dtype=np.float32)
        for team, distance in enumerate((home_d, away_d)):
            if not np.isfinite(distance):
                continue
            if distance > MAX_CONTROL_DISTANCE_M:
                continue
            closeness = max(0.0, 1.0 - distance / MAX_CONTROL_DISTANCE_M)
            support_bonus = 0.12 * min(int(support[frame, team]), 3) / 3.0
            speed_bonus = 0.0
            player_speed = nearest_speed[frame, team]
            ball_velocity = ball_speed[frame]
            if np.isfinite(player_speed) and np.isfinite(ball_velocity):
                speed_bonus = 0.12 * np.exp(-abs(float(player_speed - ball_velocity)) / 2.5)
            score[team] = closeness + support_bonus + speed_bonus + approach_bonus[frame, team]
        winner = int(np.argmax(score))
        loser = 1 - winner
        if score[winner] <= 0.0 or score[winner] - score[loser] < MIN_SCORE_MARGIN:
            continue
        if not np.isfinite(distances[frame, winner]) or distances[frame, winner] > MAX_CONTROL_DISTANCE_M:
            continue
        candidates[frame] = winner
        strength[frame] = min(float(score[winner] - score[loser]), 1.0)
    return candidates, strength


def infer_labels(candidates, strength, dead_evidence, n_frames):
    """Smooth possession belief, bridge uncertainty, and insert detected dead play.

    An opposing-team candidate is only accumulated when it is reliable. A clear
    observation for the current owner refreshes confidence and cancels pending
    switch evidence. Stronger evidence accumulates faster, while ambiguous frames
    slowly reduce confidence. Dead-ball evidence ends the live spell and resets the
    owner so the next restart can belong to either team.
    """
    codes = np.full(n_frames, "X", dtype="<U1")
    owner = UNKNOWN
    pending_team = UNKNOWN
    pending_evidence = 0.0
    pending_last_frame = -1
    confidence = 0.0

    for frame, candidate in enumerate(candidates):
        if dead_evidence[frame]:
            codes[frame] = "D"
            owner = UNKNOWN
            pending_team, pending_evidence, pending_last_frame = UNKNOWN, 0.0, -1
            confidence = 0.0
            continue

        if candidate != UNKNOWN:
            if owner == UNKNOWN:
                owner = int(candidate)
                confidence = 1.0
                pending_team, pending_evidence, pending_last_frame = UNKNOWN, 0.0, -1
            elif candidate == owner:
                confidence = 1.0
                pending_team, pending_evidence, pending_last_frame = UNKNOWN, 0.0, -1
            else:
                if (pending_team == candidate and pending_last_frame >= 0
                        and frame - pending_last_frame <= SWITCH_EVIDENCE_SPAN_FRAMES):
                    pending_evidence += float(strength[frame])
                else:
                    pending_team, pending_evidence = int(candidate), float(strength[frame])
                pending_last_frame = frame
                if pending_evidence >= SWITCH_EVIDENCE_REQUIRED:
                    owner = int(candidate)
                    confidence = 1.0
                    pending_team, pending_evidence, pending_last_frame = UNKNOWN, 0.0, -1
                else:
                    confidence = max(confidence, 0.5)
        else:
            confidence *= GAP_CONFIDENCE_DECAY

        # Confidence fades gradually instead of ending at a fixed gap length.
        if owner != UNKNOWN and confidence >= MIN_OUTPUT_CONFIDENCE:
            codes[frame] = "H" if owner == HOME else "A"
    return codes


def smooth_labels(labels, half_blocks, radius=SMOOTH_RADIUS_FRAMES):
    """Apply a centered majority vote over H/A/D, ignoring X abstentions.

    The window is clipped to each half so warm-up, halftime, or the other half
    cannot influence a scored frame. If the vote ties, keep the model's original
    label at that frame rather than introducing an arbitrary team change.
    """
    result = labels.copy()
    classes = ("H", "A", "D")
    for _half, first, last in half_blocks:
        section = labels[first:last + 1]
        cumulative = {
            label: np.concatenate(([0], np.cumsum(section == label, dtype=np.int64)))
            for label in classes
        }
        for local_frame in range(len(section)):
            low = max(0, local_frame - radius)
            high = min(len(section), local_frame + radius + 1)
            counts = {label: int(cumulative[label][high] - cumulative[label][low])
                      for label in classes}
            best_count = max(counts.values())
            if best_count == 0:
                result[first + local_frame] = "X"
                continue
            winners = [label for label in classes if counts[label] == best_count]
            if len(winners) == 1:
                result[first + local_frame] = winners[0]
            elif section[local_frame] in winners:
                result[first + local_frame] = section[local_frame]
            # For a tie that excludes the center label, preserve the center label.
    return result


def build(game=GAME, data_dir=None):
    """Predict labels on the tactical video timeline; mark non-play frames X."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    (distances, nearest_speed, support, ball_speed, ball_visible,
     ball_outside, group_speed) = tracking_observations(
        raw_path, team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"])

    approach = distance_trend_bonus(distances)
    candidates, strength = team_evidence(
        distances, nearest_speed, support, ball_speed, approach)

    # Smart Tagging gives broad restart phases, not exact possession boundaries.
    # Treat a tagged set piece as dead only while the ball is absent/stationary and
    # the tracked players are not collectively moving like play has restarted.
    set_piece = np.zeros(n_frames, dtype=bool)
    for event in load_smart_tagging(game=game, data_dir=data_dir):
        if (event["code"] or "").strip().upper() != "SET PIECES":
            continue
        lo = max(0, event["start_frame"])
        hi = min(n_frames, event["end_frame"])
        set_piece[lo:hi] = True

    inactive_gap = (~ball_visible & np.isfinite(group_speed)
                    & (group_speed <= PLAYER_GROUP_SLOW_MPS))
    inactive_run = np.zeros(n_frames, dtype=bool)
    run = 0
    for frame, inactive in enumerate(inactive_gap):
        run = run + 1 if inactive else 0
        inactive_run[frame] = run >= INACTIVITY_TO_DEAD_FRAMES

    slow_restart = (set_piece & (~ball_visible | (ball_speed <= BALL_SLOW_MPS))
                    & (~np.isfinite(group_speed) | (group_speed <= 1.5)))
    dead_evidence = ball_outside | slow_restart | inactive_run
    labels = infer_labels(candidates, strength, dead_evidence, n_frames)

    # Only the two published half blocks are graded. Avoid carrying a possession
    # into warm-up, halftime, or post-match footage.
    in_play = np.zeros(n_frames, dtype=bool)
    for _half, first, last in half_video_blocks(game):
        in_play[first:last + 1] = True
    labels[~in_play] = "X"
    return smooth_labels(labels, half_video_blocks(game), SMOOTH_RADIUS_FRAMES)


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as output:
        output.write("frame,label\n")
        output.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "possession_hysteresis.csv"),
                        help="CSV path to write")
    args = parser.parse_args()
    predictions = build()
    path = write_submission(args.out, predictions)
    unique, counts = np.unique(predictions, return_counts=True)
    print("wrote {}".format(path))
    print("label counts: {}".format(dict(zip(unique.tolist(), counts.tolist()))))
    print("Next: python evaluation/validate_submission.py {}".format(path))
