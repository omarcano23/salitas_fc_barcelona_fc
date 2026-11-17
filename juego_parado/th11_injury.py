"""TH11 — posible demora por lesión / atención médica (diagnóstico, NO adjudicación).

Un único script autónomo junto a config.py y starter/:
  python th11_injury.py --self-test
  python th11_injury.py --step 5 --out local_outputs/runs

No escribe H/A/D. No identifica jugadores ni sanitarios, ni certifica lesión.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Parameters:
    fps: int = 25
    pitch_length_m: float = 105.3    # HIPÓTESIS del proyecto, validar con DATA
    pitch_width_m: float = 68.0
    min_visible_team: int = 7
    temporal_lag_s: float = 1.0
    lag_min_factor: float = 0.65
    lag_max_factor: float = 1.55
    spatial_match_radius_m: float = 7.0  # correspondencia espacial, NO ID
    active_proxy_mps: float = 1.75
    quiet_proxy_mps: float = 1.15
    slowdown_min_mps: float = 0.48
    pre_activity_s: float = 8.0
    min_pre_samples: int = 4
    max_sample_gap_s: float = 0.85
    min_wait_s: float = 6.0
    max_wait_search_s: float = 150.0
    quiet_fraction_min: float = 0.58
    quiet_coverage_min: float = 0.72
    recovery_proxy_mps: float = 1.65
    recovery_confirm_s: float = 2.0
    recovery_recent_s: float = 5.0
    # Foco local; no se infiere postura, caída ni condición de salud.
    focal_stay_radius_m: float = 2.8
    focus_support_radius_m: float = 7.0
    min_focus_persistence_s: float = 3.0
    focus_persistence_fraction: float = 0.68
    max_focus_anchors: int = 35
    focus_snapshot_stride_s: float = 1.0
    local_crowd_min: int = 3
    local_crowd_increase_min: int = 2
    # Contexto externo no es GT.
    provider_context_s: float = 30.0
    # Grados, no probabilidades calibradas.
    score_candidate_min: int = 5
    score_strong_min: int = 8
    # Balón puede estar ausente: jamás se usa imputación para demostrar saque.
    observed_ball_still_m: float = 1.5
    max_ball_obs_gap_s: float = 1.0
    restart_ball_distance_m: float = 1.25
    restart_ball_speed_mps: float = 2.5
    max_episodes: int = 2000

P = Parameters()


def numbers(df, col, default=np.nan):
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)


def load_data(step: int):
    import config
    if step < 1: raise ValueError('--step >= 1')
    if int(config.FPS) != P.fps: raise ValueError('config.FPS y Parameters.fps no coinciden')
    n = int(config.N_VIDEO_FRAMES)
    mat = pd.read_csv(Path(config.OUTPUT_DIR) / 'matrix_v4.csv', low_memory=False)
    if len(mat) != n or not np.array_equal(numbers(mat, 'frame'), np.arange(n)):
        raise ValueError('matrix_v4.csv no se alinea a frame=0..N-1')
    frames = np.arange(0, n, step, dtype=int)
    periods = np.asarray([config.get_period(int(f)) for f in frames])
    mask = np.isin(periods, ('FIRST_HALF', 'SECOND_HALF'))
    frames, periods = frames[mask], periods[mask]
    from starter.load_atd import load
    raw = load(game=config.GAME, source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name == 'frame_id': raw = raw.reset_index()
        else: raise ValueError('ATD: frame_id ausente')
    raw = raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw) != n or not np.array_equal(numbers(raw, 'frame_id'), np.arange(1, n + 1)):
        raise ValueError('ATD: frame_id debe ser 1..N y coincidir con matriz')
    xc = [f'{i}_x' for i in range(1, 47)]
    yc = [f'{i}_y' for i in range(1, 47)]
    if any(c not in raw.columns for c in xc + yc): raise ValueError('ATD: faltan slots 1..46')
    x = raw.loc[frames, xc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32) * P.pitch_length_m
    y = raw.loc[frames, yc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32) * P.pitch_width_m
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= P.pitch_length_m) & (y >= 0) & (y <= P.pitch_width_m)
    x[~valid] = np.nan
    y[~valid] = np.nan
    bx = numbers(mat, 'ball_x_normalized')[frames] * P.pitch_length_m
    by = numbers(mat, 'ball_y_normalized')[frames] * P.pitch_width_m
    obs = (numbers(mat, 'ball_observed', 0)[frames] > .5) & (numbers(mat, 'ball_imputed', 0)[frames] < .5) & np.isfinite(bx) & np.isfinite(by)
    return dict(frames=frames, periods=periods, x=x, y=y, bx=bx, by=by, obs=obs)


def provider_events():
    import config
    try:
        from starter.load_smart_tagging import load
        events = load(game=config.GAME)
    except (ImportError, FileNotFoundError, ValueError, OSError, TypeError, KeyError):
        return []
    out = []
    for e in events:
        if not isinstance(e, dict): continue
        tags = e.get('tags') or {}
        if not isinstance(tags, dict): tags = {}
        kind = ' '.join(map(str, (tags.get('Type', ''), e.get('type', ''), e.get('code', '')))).lower()
        if not any(t in kind for t in ('injur', 'medical', 'treatment', 'stretcher', 'lesion', 'lesión', 'foul', 'substitution', 'var')):
            continue
        try: a, b = int(e['start_frame']), int(e['end_frame'])
        except (ValueError, TypeError, KeyError): continue
        if b <= a: continue
        explicit = any(t in kind for t in ('injur', 'medical', 'treatment', 'stretcher', 'lesion', 'lesión'))
        out.append(dict(start_frame=a, end_exclusive=b, raw_type=kind, explicit_injury=explicit))
    return out


def points(d, i, team=None):
    sl = slice(None) if team is None else (slice(0, 24) if team == 0 else slice(24, 46))
    z = np.column_stack((d['x'][i, sl], d['y'][i, sl]))
    return z[np.isfinite(z).all(axis=1)]


def _motion(a, b, dt):
    if min(len(a), len(b)) < P.min_visible_team or dt <= 0: return np.nan
    dd = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    v = np.r_[dd.min(axis=0), dd.min(axis=1)]
    v = v[v <= P.spatial_match_radius_m]
    if len(v) < 2 * P.min_visible_team: return np.nan
    return float(np.median(v) / dt)


def features(d):
    f, per = d['frames'], d['periods']
    n = len(f)
    motion = np.full(n, np.nan)
    team_motion = np.full((n, 2), np.nan)
    coverage = np.zeros((n, 2), dtype=int)
    centroid = np.full((n, 2, 2), np.nan)
    for i in range(n):
        here = [points(d, i, t) for t in (0, 1)]
        for t in (0, 1):
            coverage[i, t] = len(here[t])
            if len(here[t]) >= P.min_visible_team:
                centroid[i, t] = np.median(here[t], axis=0)
        j = int(np.searchsorted(f, f[i] - P.temporal_lag_s * P.fps, side='right') - 1)
        if j < 0 or j >= i or per[j] != per[i]: continue
        dt = (f[i] - f[j]) / P.fps
        if not P.temporal_lag_s * P.lag_min_factor <= dt <= P.temporal_lag_s * P.lag_max_factor: continue
        for t in (0, 1):
            team_motion[i, t] = _motion(here[t], points(d, j, t), dt)
        if np.isfinite(team_motion[i]).all(): motion[i] = team_motion[i].mean()
    valid = (coverage.min(axis=1) >= P.min_visible_team) & np.isfinite(motion)
    return dict(motion=motion, team_motion=team_motion, coverage=coverage, centroid=centroid, valid=valid)


def _contiguous(d, a, b):
    if b <= a: return False
    f = d['frames']
    return bool((d['periods'][a] == d['periods'][b]) and
                np.max(np.diff(f[a:b + 1])) / P.fps <= P.max_sample_gap_s)


def _focus(d, idx):
    """Foco espacial persistente: ninguna referencia al ID de un jugador."""
    f = d['frames']
    if len(idx) < 3: return None
    sec = float((f[idx[-1]] - f[idx[0]]) / P.fps)
    if sec < P.min_focus_persistence_s: return None
    stride_frames = max(1, int(P.focus_snapshot_stride_s * P.fps))
    sampled = []
    for k in idx:
        if not sampled or f[k] - f[sampled[-1]] >= stride_frames: sampled.append(int(k))
    if len(sampled) < 3: return None
    # Usar varios momentos para evitar sesgo hacia un único snapshot.
    seeds = sorted(set([sampled[0], sampled[len(sampled)//2], sampled[-1]]))
    anchors = []
    for k in seeds:
        anchors.extend([tuple(v) for v in points(d, k)])
    if len(anchors) > P.max_focus_anchors:
        sampled_ix = np.linspace(0, len(anchors) - 1, P.max_focus_anchors).astype(int)
        anchors = [anchors[k] for k in sampled_ix]
    best = None
    for a in anchors:
        anchor = np.asarray(a)
        # Un foco debe incluir un jugador próximo durante parte significativa de la espera.
        stay = []; crowds = []; both = []
        for k in sampled:
            ps = points(d, k)
            if not len(ps): continue
            ds = np.linalg.norm(ps - anchor, axis=1)
            occupant = bool(np.any(ds <= P.focal_stay_radius_m))
            stay.append(occupant)
            # Quitar el posible ocupante central del grupo de atención.
            crowds.append(max(0, int(np.sum(ds <= P.focus_support_radius_m)) - int(occupant)))
            t0 = points(d, k, 0); t1 = points(d, k, 1)
            both.append(bool(len(t0) and len(t1) and
                             np.min(np.linalg.norm(t0 - anchor, axis=1)) <= P.focus_support_radius_m and
                             np.min(np.linalg.norm(t1 - anchor, axis=1)) <= P.focus_support_radius_m))
        persistence = float(np.mean(stay)) if stay else 0.
        if persistence < P.focus_persistence_fraction: continue
        # Increase relative to early waiting positions; if gathered already at quiet onset, no hard veto.
        early = float(np.median(crowds[:max(1, len(crowds)//3)]))
        peak = int(max(crowds)) if crowds else 0
        increase = float(peak - early)
        has_group = peak >= P.local_crowd_min
        if not has_group: continue
        score = 3 * persistence + min(peak, 5) * .65 + min(increase, 4) * .45 + .35 * np.mean(both)
        item = dict(x_m=round(float(anchor[0]), 2), y_m=round(float(anchor[1]), 2),
                    persistence_fraction=round(persistence, 3), nearby_peak=int(peak),
                    nearby_increase=round(increase, 2), mixed_teams_fraction=round(float(np.mean(both)), 3),
                    support_score=round(float(score), 3), group_formed=bool(increase >= P.local_crowd_increase_min))
        if best is None or item['support_score'] > best['support_score']:
            best = item
    return best


def _observed_ball(d, a_frame, b_frame):
    f = d['frames']
    indices = np.flatnonzero(d['obs'] & (f >= a_frame) & (f <= b_frame))
    stationary = []; moving = []
    for u, v in zip(indices[:-1], indices[1:]):
        if d['periods'][u] != d['periods'][v]: continue
        dt = (f[v] - f[u]) / P.fps
        if dt <= 0 or dt > P.max_ball_obs_gap_s: continue
        dist = float(np.hypot(d['bx'][v] - d['bx'][u], d['by'][v] - d['by'][u]))
        if dist <= P.observed_ball_still_m: stationary.append((int(f[u]), int(f[v])))
        if dist >= P.restart_ball_distance_m and dist / dt >= P.restart_ball_speed_mps:
            moving.append(dict(from_frame=int(f[u]), to_frame=int(f[v]), meters=round(dist, 2)))
    return dict(stationary_pairs=len(stationary), movement_candidates=moving[:10])


def _context(providers, a, b):
    close = [x for x in providers if x['start_frame'] <= b + P.provider_context_s*P.fps and
             x['end_exclusive'] >= a - P.provider_context_s*P.fps]
    return close


def detect(d, providers=None, ft=None):
    providers = [] if providers is None else providers
    ft = features(d) if ft is None else ft
    f, per, n = d['frames'], d['periods'], len(d['frames'])
    if n < 5: return []
    q = ft['valid'] & (ft['motion'] <= P.quiet_proxy_mps)
    result = []
    i = 1
    while i < n and len(result) < P.max_episodes:
        if not q[i] or (q[i-1] and per[i] == per[i-1]):
            i += 1; continue
        # Actividad anterior real (sin señales previas = no basta quietud inicial).
        pre = np.flatnonzero((f < f[i]) & (f >= f[i] - P.pre_activity_s*P.fps) &
                            (per == per[i]) & ft['valid'])
        if len(pre) < P.min_pre_samples:
            i += 1; continue
        baseline = float(np.quantile(ft['motion'][pre], .75))
        if baseline < P.active_proxy_mps or baseline - ft['motion'][i] < P.slowdown_min_mps:
            i += 1; continue
        j = i
        last_quiet = i
        stop_horizon = f[i] + int(P.max_wait_search_s * P.fps)
        while j + 1 < n and f[j+1] <= stop_horizon and per[j+1] == per[i]:
            if (f[j+1] - f[j]) / P.fps > P.max_sample_gap_s or not ft['valid'][j+1]: break
            k = j + 1
            recent = np.flatnonzero((f >= f[k] - P.recovery_recent_s*P.fps) &
                                   (f <= f[k]) & (per == per[i]) & ft['valid'])
            if f[k] - f[i] >= P.min_wait_s*P.fps and len(recent) >= 3:
                if (np.mean(q[recent]) < P.quiet_fraction_min and
                        np.median(ft['motion'][recent]) >= P.recovery_proxy_mps):
                    break
            j = k
            if q[j]: last_quiet = j
        idx = np.arange(i, last_quiet + 1)
        duration = (f[last_quiet] - f[i]) / P.fps
        if duration < P.min_wait_s or not _contiguous(d, i, last_quiet):
            i = max(i+1, j+1); continue
        if np.mean(ft['valid'][idx]) < P.quiet_coverage_min or np.mean(q[idx]) < P.quiet_fraction_min:
            i = max(i+1, j+1); continue
        focus = _focus(d, idx)
        # No declarar lesión sobre quietud global o posible portero inmóvil solamente.
        if not focus:
            i = max(i+1, j+1); continue
        near_goalkeeper_zone = bool((focus['x_m'] < 7 or focus['x_m'] > P.pitch_length_m-7) and
                                    abs(focus['y_m'] - P.pitch_width_m/2) < 10)
        # Sin cambio de agrupación ni apoyo del proveedor, el foco fijo es inespecífico.
        ctx = _context(providers, int(f[i]), int(f[last_quiet]))
        explicit = any(e['explicit_injury'] for e in ctx)
        if near_goalkeeper_zone and not focus['group_formed'] and not explicit:
            i = max(i+1, j+1); continue
        # Resultado: grupo persistente + transición, no diagnóstico médico.
        score = 2 + (2 if duration >= 12 else 1) + 2 + (2 if focus['group_formed'] else 0)
        score += (1 if focus['nearby_peak'] >= 4 else 0)
        score += (1 if focus['mixed_teams_fraction'] >= .3 else 0)
        score += (2 if explicit else 0)
        if score < P.score_candidate_min:
            i = max(i+1, j+1); continue
        # Recuperación indica juego activo, no instante reglamentario del saque.
        recovery = None
        if j + 1 < n and per[j+1] == per[i]:
            r = np.flatnonzero((f >= f[j+1]) & (f <= f[j+1]+P.recovery_confirm_s*P.fps) &
                               (per == per[i]) & ft['valid'])
            if len(r) >= 3 and (f[r[-1]] - f[r[0]])/P.fps >= P.recovery_confirm_s*.7 and np.mean(ft['motion'][r] >= P.recovery_proxy_mps) >= .65:
                recovery = dict(lower_frame=int(f[last_quiet]), upper_frame=int(f[r[0]]),
                                movement_recovery_not_kick=True)
        ball = _observed_ball(d, int(f[i]), int(f[min(n-1, j+1)]))
        kind = 'INJURY_CONTEXT_SUPPORTED' if explicit else 'MEDICAL_STOPPAGE_COMPATIBLE_UNCONFIRMED'
        result.append(dict(detector='TH11_INJURY', kind=kind, injury_confirmed=False,
                           candidate_strength='STRONG_PATTERN' if score >= P.score_strong_min else 'WEAK_PATTERN',
                           evidence_score=int(score),
                           quiet_start_frame=int(f[i]), quiet_end_frame=int(f[last_quiet]),
                           quiet_duration_s=round(float(duration), 2),
                           dead_start_bracket=dict(lower_frame=int(f[pre[-1]]), upper_frame=int(f[i]), estimated=True),
                           dead_end_bracket=recovery,
                           focus=focus, provider_context=[e['raw_type'] for e in ctx][:10],
                           explicit_injury_tag=explicit, observed_ball=ball,
                           evidence=dict(prior_activity_proxy_mps=round(baseline, 3),
                                         quiet_fraction=round(float(np.mean(q[idx])), 3),
                                         local_gathering=focus['group_formed']),
                           competing_explanations=['VAR','substitution','foul/free kick','goal celebration','cooling break','tactical huddle'],
                           warning='No prueba lesión, postura corporal, entrada de sanitarios ni fronteras exactas de D'))
        i = max(i + 1, j + 1)
    return result


def self_test():
    # Secuencia inventada, NO frames del partido ni calibración con GT.
    f = np.arange(0, 1250, 5)
    n = len(f)
    rng = np.random.default_rng(10)
    xy = np.empty((n, 46, 2), dtype=float)
    base = np.column_stack((rng.uniform(9,96,46), rng.uniform(6,62,46)))
    for k in range(n):
        xy[k] = base.copy()
        xy[k, :24, 0] += .01 * k
        xy[k, 24:, 1] += .01 * k
    # Foco persistente y aproximación de varios jugadores después de iniciar la pausa.
    xy[:, 0] = [48, 30]
    xy[45:190, 1] = [50, 31]
    xy[45:190, 2] = [51, 32]
    xy[45:190, 25] = [52, 30]
    xy[45:190, 26] = [51, 29]
    # Mantener el grupo en la espera y romperlo al final.
    dd = dict(frames=f, periods=np.array(['FIRST_HALF']*n), x=xy[:,:,0], y=xy[:,:,1],
              bx=np.full(n,np.nan),by=np.full(n,np.nan),obs=np.zeros(n,bool))
    motion=np.r_[np.full(35,2.4), np.full(155,.35), np.full(n-190,2.2)]
    ft=dict(motion=motion, team_motion=np.column_stack((motion,motion)),
            coverage=np.full((n,2),11), valid=np.ones(n,bool), centroid=np.zeros((n,2,2)))
    res=detect(dd, [], ft)
    assert res and res[0]['focus'] and not res[0]['injury_confirmed'],res
    assert res[0]['kind']=='MEDICAL_STOPPAGE_COMPATIBLE_UNCONFIRMED'
    assert not detect(dd, [], {**ft,'motion':np.full(n,.3)}), 'No episodio sin transición previa'
    # Sin foco concentrado no se debe afirmar atención médica.
    empty = dict(dd); empty['x'] = np.full((n,46), np.nan);empty['y']=np.full((n,46),np.nan)
    assert not detect(empty, [], ft)
    # El balón imputado no aparece como observado.
    assert not _observed_ball(dd,0,1000)['movement_candidates']
    print('TH11 self-test OK (sintético; no valida cobertura ni precisión real)')


def main():
    ap = argparse.ArgumentParser(description='TH11: lesión/atención compatible; no confirma diagnóstico')
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--out', default='local_outputs/runs')
    ap.add_argument('--self-test', action='store_true')
    args=ap.parse_args()
    if args.self_test: self_test(); return
    d=load_data(args.step)
    episodes=detect(d,provider_events())
    payload=dict(detector='TH11_INJURY', parameters=asdict(P), step=args.step,
                 episodes=episodes,count=len(episodes),
                 disclaimer='Un episodio es compatible con atención médica, NO prueba lesión ni adjudica D.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dest=out/f'th11_injury_step{args.step}.json'
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{dest} | candidatos={len(episodes)} | lesiones confirmadas=0')

if __name__=='__main__': main()
