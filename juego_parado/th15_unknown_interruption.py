"""TH15 — detector RESIDUAL de interrupciones no explicadas (diagnóstico, no árbitro).

  python th15_unknown_interruption.py --self-test
  python th15_unknown_interruption.py --step 5 --out local_outputs/runs
  python th15_unknown_interruption.py --step 5 --evidence-dir local_outputs/runs

Dependencias: numpy, pandas, config.py, starter.load_atd y matrix_v4.csv.
NO modifica run.py, NO etiqueta H/A/D y NO puede afirmar una parada sin indicios
convergentes. El modo con --evidence-dir resta intervalos de detectores previos
para producir candidatos residuales; sin evidencias se exportan SOLO candidatos
sin resolver, nunca 'desconocidos finales'.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Parameters:
    fps: int = 25
    length_m: float = 105.3  # Parámetro del proyecto; verificar escala y homografía.
    width_m: float = 68.0
    observation_s: float = 0.6  # 15 frames a 25 fps; independiente de --step
    lag_s: float = 1.2
    lag_factor_min: float = 0.7
    lag_factor_max: float = 1.6
    min_visible_team: int = 7
    min_correspondence: int = 7
    spatial_pair_max_m: float = 7.0
    max_gap_s: float = 1.25
    pre_window_s: float = 7.0
    active_proxy_mps: float = 1.55
    quiet_proxy_mps: float = 1.15
    minimum_drop_mps: float = 0.35
    min_pre_valid: int = 3
    min_stop_s: float = 3.0
    max_episode_search_s: float = 240.0
    stable_fraction: float = 0.62
    low_motion_recovery_mps: float = 1.55
    recovery_window_s: float = 2.4
    recovery_fraction: float = 0.67
    ball_static_radius_m: float = 1.4
    ball_stable_min_s: float = 1.4
    ball_dynamic_mps: float = 2.3
    observed_ball_max_gap_s: float = 1.3
    recent_ball_s: float = 3.0
    transition_lookback_s: float = 3.0
    dedup_gap_s: float = 2.0
    evidence_overlap_fraction: float = 0.35
    evidence_margin_s: float = 2.0
    max_results: int = 1500

P = Parameters()


def col(df, name, default=np.nan):
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)


def load_data(step: int):
    import config
    if step < 1: raise ValueError('--step debe ser >= 1')
    if int(config.FPS) != P.fps: raise ValueError('FPS en config.py y TH15 difieren')
    n = int(config.N_VIDEO_FRAMES)
    matrix = pd.read_csv(Path(config.OUTPUT_DIR) / 'matrix_v4.csv', low_memory=False)
    if len(matrix) != n or not np.array_equal(col(matrix, 'frame'), np.arange(n)):
        raise ValueError('matrix_v4.csv debe tener frame 0..N-1')
    from starter.load_atd import load
    raw = load(game=config.GAME, source='tactical').to_df()
    if 'frame_id' not in raw.columns:
        if raw.index.name == 'frame_id': raw = raw.reset_index()
        else: raise ValueError('ATD sin frame_id')
    raw = raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw) != n or not np.array_equal(col(raw, 'frame_id'), np.arange(1, n+1)):
        raise ValueError('ATD debe alinear frame_id 1..N')
    stride = max(step, int(np.ceil(P.observation_s*P.fps/step))*step)
    f = np.arange(0, n, stride, dtype=int)
    periods = np.array([config.get_period(int(v)) for v in f])
    keep = np.isin(periods, ('FIRST_HALF','SECOND_HALF'))
    f, periods = f[keep], periods[keep]
    xc = [f'{k}_x' for k in range(1,47)]
    yc = [f'{k}_y' for k in range(1,47)]
    if any(c not in raw.columns for c in xc+yc):
        raise ValueError('ATD no contiene los 46 slots de coordenadas')
    x = raw.loc[f, xc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)*P.length_m
    y = raw.loc[f, yc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)*P.width_m
    valid = np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.length_m)&(y>=0)&(y<=P.width_m)
    x[~valid] = np.nan; y[~valid] = np.nan
    bx = col(matrix, 'ball_x_normalized')[f]*P.length_m
    by = col(matrix, 'ball_y_normalized')[f]*P.width_m
    observed = (col(matrix,'ball_observed',0)[f]>.5)&(col(matrix,'ball_imputed',0)[f]<.5)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=f,periods=periods,x=x,y=y,bx=bx,by=by,obs=observed, stride=stride)


def team_points(d, i, team):
    sl = slice(0,24) if team == 0 else slice(24,46)
    q = np.column_stack((d['x'][i,sl], d['y'][i,sl]))
    return q[np.isfinite(q).all(axis=1)]


def matched_motion(a, b, dt):
    """Correspondencia geométrica bidireccional, NO identidad ni velocidad exacta."""
    if min(len(a),len(b))<P.min_visible_team or dt<=0: return np.nan
    distances = np.linalg.norm(a[:,None,:]-b[None,:,:], axis=2)
    nearest = np.r_[distances.min(axis=0), distances.min(axis=1)]
    nearest = nearest[nearest<=P.spatial_pair_max_m]
    if len(nearest)<2*P.min_correspondence:return np.nan
    return float(np.median(nearest)/dt)


def derive_features(d):
    f, per = d['frames'], d['periods']; n=len(f)
    team_motion=np.full((n,2),np.nan)
    coverage=np.zeros((n,2),int)
    for i in range(n):
        ps = [team_points(d,i,t) for t in (0,1)]
        for t in (0,1): coverage[i,t]=len(ps[t])
        j = int(np.searchsorted(f,f[i]-P.lag_s*P.fps,side='right')-1)
        if j<0 or j>=i or per[j]!=per[i]:continue
        dt=(f[i]-f[j])/P.fps
        if not P.lag_s*P.lag_factor_min<=dt<=P.lag_s*P.lag_factor_max:continue
        for t in (0,1):team_motion[i,t]=matched_motion(ps[t],team_points(d,j,t),dt)
    valid=(coverage.min(axis=1)>=P.min_visible_team)&np.isfinite(team_motion).all(axis=1)
    motion=np.full(n,np.nan)
    motion[valid]=np.median(team_motion[valid],axis=1)
    # Anclas de balón: solo observado y pares con diferencia temporal admisible.
    stable_ball=np.zeros(n,bool);moving_ball=np.zeros(n,bool)
    last=None
    for i in range(n):
        if not d['obs'][i]:continue
        if last is not None and per[i]==per[last]:
            dt=(f[i]-f[last])/P.fps
            if 0<dt<=P.observed_ball_max_gap_s:
                dist=float(np.hypot(d['bx'][i]-d['bx'][last],d['by'][i]-d['by'][last]))
                stable_ball[i] = dist<=P.ball_static_radius_m
                moving_ball[i] = dist/dt>=P.ball_dynamic_mps and dist>=P.ball_static_radius_m
        last=i
    return dict(motion=motion,team_motion=team_motion,valid=valid,coverage=coverage,
                stable_ball=stable_ball,moving_ball=moving_ball)


def _indices(d, start, end, period):
    f=d['frames'];return np.flatnonzero((f>=start)&(f<=end)&(d['periods']==period))


def _fraction(arr, valid):
    return float(np.mean(arr[valid])) if np.any(valid) else 0.


def _stable_ball_run(d,ft, start_i, end_i):
    """Exigir persistencia del balón, no un único par quieto."""
    f=d['frames']; idx=np.arange(start_i,end_i+1)
    ok=idx[ft['stable_ball'][idx]]
    if len(ok)<2:return False
    return (f[ok[-1]]-f[ok[0]])/P.fps>=P.ball_stable_min_s and len(ok)/max(1,len(idx))>=0.4


def detect(d, ft=None):
    """Candidatos de interrupción. No asignar D por quietud aislada."""
    ft=derive_features(d) if ft is None else ft
    f,per=d['frames'],d['periods']; n=len(f)
    if n<8:return []
    quiet=ft['valid']&(ft['motion']<=P.quiet_proxy_mps)
    episodes=[]; i=1
    while i<n and len(episodes)<P.max_results:
        if not quiet[i] or (quiet[i-1] and per[i]==per[i-1]):
            i+=1;continue
        previous=_indices(d,int(f[i]-P.pre_window_s*P.fps),int(f[i]-1),per[i])
        pv=previous[ft['valid'][previous]]
        if len(pv)<P.min_pre_valid:i+=1;continue
        baseline=float(np.nanquantile(ft['motion'][pv],0.75))
        if baseline<P.active_proxy_mps or baseline-ft['motion'][i]<P.minimum_drop_mps:
            i+=1;continue
        last_i=i;max_f=int(f[i]+P.max_episode_search_s*P.fps)
        gap_recovery=0
        j=i
        while j+1<n and f[j+1]<=max_f and per[j+1]==per[i]:
            k=j+1
            if (f[k]-f[j])/P.fps>P.max_gap_s or not ft['valid'][k]:break
            # Confirmar reactivación mantenida, evitar que un sprint aislado corte D.
            if ft['motion'][k]>=P.low_motion_recovery_mps:
                gap_recovery+=f[k]-f[j]
                if gap_recovery>=P.recovery_window_s*P.fps:
                    break
            else:
                gap_recovery=0
            j=k;last_i=k
        inds=np.arange(i,last_i+1)
        duration=(f[last_i]-f[i])/P.fps
        quiet_fraction=_fraction(quiet[inds],ft['valid'][inds])
        stable_ball=_stable_ball_run(d,ft,i,last_i)
        independent=(quiet_fraction>=P.stable_fraction and duration>=P.min_stop_s)
        # Dos niveles de evidencia: contracción de movimiento + espera; balón estacionario corrobora,
        # pero no se exige dado que el tracking del balón tiene ausencias.
        if independent:
            tail=_indices(d,int(f[last_i]),int(f[last_i]+P.recovery_window_s*P.fps),per[i])
            recovered=(len(tail)>=2 and
                       _fraction((ft['motion'][tail]>=P.low_motion_recovery_mps), ft['valid'][tail])>=P.recovery_fraction)
            # Primer balón móvil al aproximarse a la recuperación. No es una prueba de golpeo.
            ball_idx=_indices(d,int(f[last_i]-P.recent_ball_s*P.fps),
                              int(f[last_i]+P.recent_ball_s*P.fps),per[i])
            ball_move=[int(f[k]) for k in ball_idx if ft['moving_ball'][k]]
            prev_transition=_indices(d,int(f[i]-P.transition_lookback_s*P.fps),int(f[i]),per[i])
            last_active=[int(f[k]) for k in prev_transition if ft['valid'][k] and ft['motion'][k]>=P.active_proxy_mps]
            # La 'frontera D' siempre es intervalo, nunca frame atribuido a silbato/golpeo.
            start_lo=last_active[-1] if last_active else int(f[pv[-1]])
            score=2+int(quiet_fraction>=.78)+int(stable_ball)+int(recovered)+int(bool(ball_move))
            episodes.append(dict(
                id=f'TH15-{len(episodes)+1:04d}',kind='STOPPAGE_CANDIDATE_UNRESOLVED',
                period=str(per[i]),start_frame=int(f[i]),end_frame=int(f[last_i]),
                duration_s=round(float(duration),2),score_ordinal=score,
                evidence=dict(pre_activity_proxy_mps=round(baseline,3),
                              pause_motion_proxy_mps=round(float(np.nanmedian(ft['motion'][inds])),3),
                              quiet_fraction=round(quiet_fraction,3),
                              ball_observed_stable=bool(stable_ball),
                              recovery_collective_compatible=bool(recovered),
                              ball_movement_observed_near_end=bool(ball_move),
                              ball_movement_frames=ball_move[:6],
                              min_team_coverage=coverage_min(ft,inds)),
                dead_start_bracket=dict(lower_frame=int(start_lo),upper_frame=int(f[i]),estimated=True),
                dead_end_bracket=dict(lower_frame=int(f[last_i]),
                                      upper_frame=int(f[tail[-1]]) if recovered else None,
                                      estimated=True,not_proven_restart=True),
                confidence='STRUCTURAL_CANDIDATE' if score>=4 else 'WEAK_CANDIDATE',
                limitations=['No whistle or referee tracking','Slow play may resemble stoppage',
                             'Motion recovery does not prove legal restart']))
        i=max(i+1,last_i+1)
    return episodes


def coverage_min(ft,inds):
    return [int(ft['coverage'][inds,t].min()) for t in (0,1)]


# Formatos aceptados de detector: formatos diagnosticados, NO leer columnas arbitrarias
# como si fueran intervalos de juego parado. La conexión final deberá normalizar
# la semántica de cada evidencia (inicio de configuración != inicio exacto de D).
INTERVAL_FIELDS={
    'TH01': [('dead_start_lower_frame','dead_end_upper_frame'),('start_frame','end_frame')],
    'TH02': [('dead_start_lower_frame','dead_end_upper_frame'),('start_frame','end_frame')],
    'TH03': [('dead_start_lower_frame','dead_end_upper_frame'),('start_frame','end_frame')],
    'TH04': [('dead_start_lower_frame','dead_end_upper_frame'),('candidate_start_frame','candidate_end_frame')],
    'TH05': [('dead_start_lower_frame','dead_end_upper_frame'),('candidate_start_frame','candidate_end_frame')],
    'TH06': [('dead_start_lower_frame','dead_end_upper_frame'),('candidate_start_frame','candidate_end_frame')],
    'TH07': [('dead_start_lower_frame','dead_end_upper_frame')],
    'TH08': [('dead_start_lower_frame','dead_end_upper_frame')],
    'TH09': [], # atributo, no mecanismo ni adjudicador
    'TH10': [], # pausa genérica NO adjudica causa
    'TH11': [], # atributo, no mecanismo
    'TH12': [], # contexto de pausa colectiva, no reanudación
    'TH13': [], # atributo, no mecanismo
    'TH14': [('dead_start_lower_frame','dead_end_upper_frame')],
}


def _as_int(v):
    try:
        if v is None or (isinstance(v,str) and not v.strip()):return None
        return int(float(v))
    except (TypeError,ValueError,OverflowError):return None


def _bracket_interval(e):
    """Única vía segura: fronteras declaradas, no tiempos de etiquetas/eventos."""
    st=e.get('dead_start_bracket');en=e.get('dead_end_bracket')
    if not isinstance(st,dict) or not isinstance(en,dict):return None
    a=_as_int(st.get('lower_frame'));b=_as_int(en.get('upper_frame'))
    if a is None or b is None or b<a:return None
    return a,b


def read_prior_evidence(folder: Path):
    """Lee SOLO intervalos explícitos; evita inferir semántica de CSV de TH01–14.

    --evidence-dir NO implica cobertura completa: registrar qué detectores faltan.
    """
    if not folder.exists():return [],[],['Carpeta de evidencias inexistente']
    intervals=[];found=set();warnings=[]
    for path in sorted(folder.glob('th[0-1][0-9]*.json')):
        name=path.stem.upper();th=name[:4]
        if th not in INTERVAL_FIELDS or th=='TH15':continue
        found.add(th)
        try:payload=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:warnings.append(f'{path.name}: {exc}');continue
        records=payload.get('episodes',[]) if isinstance(payload,dict) else []
        if not isinstance(records,list):continue
        count=0
        for e in records:
            if not isinstance(e,dict):continue
            v=_bracket_interval(e)
            if v is None:continue
            # Nunca restar candidatos no adjudicados de pausa genérica.
            if th in ('TH09','TH10','TH11','TH12','TH13'):continue
            if e.get('confirmed') is False or e.get('is_confirmed') is False:continue
            # En fase diagnóstica, incluso intervalo válido es solo soporte externo.
            intervals.append(dict(source=th,start=v[0],end=v[1],path=path.name))
            count+=1
        if count==0:warnings.append(f'{path.name}: sin intervalo de D explícito utilizable; no se resta')
    # Formatos *_episodes.csv no contienen de manera uniforme dos límites de D.
    # No se usan: se normalizarán al integrar TH01–TH14.
    return intervals,sorted(found),warnings


def explain_overlap(episodes,prior):
    for e in episodes:
        a,b=e['start_frame'],e['end_frame'];duration=max(1,b-a)
        links=[];covered=[]
        for p in prior:
            overlap=max(0,min(b,p['end'])-max(a,p['start']))
            frac=overlap/duration
            if frac>0:
                links.append(dict(source=p['source'],start_frame=p['start'],end_frame=p['end'],
                                  overlap_fraction=round(frac,3)))
                if frac>=P.evidence_overlap_fraction:
                    covered.append(p['source'])
        e['prior_explanations']=links[:30]
        e['supported_by_prior_detectors']=sorted(set(covered))
        e['residual_status']='EXPLAINED_OR_OVERLAPPING' if covered else 'UNRESOLVED_NEEDS_ARBITRATION'
        e['not_certified_unknown']=True
    return episodes


def self_test():
    assert matched_motion(np.array([[x,0.] for x in range(7)]),
                          np.array([[x,0.] for x in range(7)]),1.0)==0
    f=np.arange(0,1400,15);n=len(f)
    q=np.r_[np.full(16,2.2),np.full(42,.48),np.full(n-58,2.1)]
    d=dict(frames=f,periods=np.array(['FIRST_HALF']*n),
           x=np.full((n,46),np.nan),y=np.full((n,46),np.nan),
           bx=np.full(n,np.nan),by=np.full(n,np.nan),obs=np.zeros(n,bool))
    ft=dict(motion=q,valid=np.ones(n,bool),coverage=np.full((n,2),10),
            stable_ball=np.zeros(n,bool),moving_ball=np.zeros(n,bool))
    result=detect(d,ft)
    assert len(result)==1 and result[0]['start_frame']>=f[16],result
    assert result[0]['dead_end_bracket']['not_proven_restart']
    q2=np.full(n,.4)
    ft['motion']=q2
    assert detect(d,ft)==[], 'Quietud sin actividad previa => NO interrumpir'
    ft['motion']=np.full(n,2.1)
    assert detect(d,ft)==[], 'Juego activo => NO interrumpir'
    explain_overlap(result,[dict(source='TH04',start=result[0]['start_frame'],end=result[0]['end_frame'])])
    assert result[0]['residual_status']=='EXPLAINED_OR_OVERLAPPING'
    assert result[0]['not_certified_unknown']
    print('TH15: pruebas sintéticas OK; NO es prueba de precisión sobre partido real')


def main():
    ap=argparse.ArgumentParser(description='TH15: detector residual sin modificar run.py')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',type=Path,default=Path('local_outputs/runs'))
    ap.add_argument('--evidence-dir',type=Path,default=None,
                    help='Carpeta con JSON de otros detectores; solo cruces conservadores')
    args=ap.parse_args()
    if args.self_test:self_test();return
    d=load_data(args.step)
    candidate=detect(d)
    prior,loaded,warnings=read_prior_evidence(args.evidence_dir) if args.evidence_dir else ([],[],[])
    episodes=explain_overlap(candidate,prior)
    result=dict(detector='TH15_UNKNOWN_INTERRUPTION',schema_version='1.0',
                parameters=asdict(P),input_step=int(args.step),sample_stride=int(d['stride']),
                episodes=episodes,counts=dict(total=len(episodes),
                   unresolved=sum(e['residual_status']=='UNRESOLVED_NEEDS_ARBITRATION' for e in episodes),
                   overlapping=sum(e['residual_status']=='EXPLAINED_OR_OVERLAPPING' for e in episodes)),
                prior_evidence=dict(dir=str(args.evidence_dir) if args.evidence_dir else None,
                   detected_json_sources=loaded,usable_intervals=len(prior),warnings=warnings,
                   integration_complete=False),
                warning='Candidatos no adjudicados. NO escribir D directamente ni interpretar unknown como causa confirmada.')
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/f'th15_unknown_interruption_step{args.step}.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{target} | candidatos={len(episodes)} | irresueltos={result["counts"]["unresolved"]} | integración pendiente')

if __name__=='__main__':main()
