"""TH13: sustituciones COMPATIBLES a partir de tracking sin identidades.

Requisitos: numpy, pandas, scipy; config.py, starter.load_atd y matrix_v4.csv
Uso:
    python th13_substitution.py --self-test
    python th13_substitution.py --step 5 --out local_outputs/runs

No escribe H/A/D. No demuestra identidad ni acto reglamentario de sustitución.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

@dataclass(frozen=True)
class Params:
    fps: int = 25
    length_m: float = 105.3  # hipótesis de escala; verificar contra metadatos
    width_m: float = 68.0
    observation_s: float = 0.8  # ventanas de observación genuinamente separadas
    max_gap_s: float = 1.3
    max_match_speed_mps: float = 9.0
    match_extra_m: float = 1.2
    min_track_s: float = 2.4
    min_visible_team: int = 7
    max_visible_team: int = 14  # no confundir duplicados con sustitutos
    boundary_near_m: float = 4.5
    touchline_near_m: float = 4.5
    midfield_entry_x_m: float = 9.0
    entry_near_line_m: float = 4.5
    entry_inward_m: float = 5.0
    exit_edge_progress_m: float = 3.0
    exit_lookback_s: float = 9.0
    entry_lookahead_s: float = 10.0
    leave_disappearance_s: float = 1.6
    entry_persistence_s: float = 2.4
    pair_min_delay_s: float = -3.0  # ambigüedad de tracking oclusiones
    pair_max_delay_s: float = 25.0
    halftime_guard_s: float = 30.0
    period_end_guard_s: float = 10.0
    local_radius_m: float = 12.0
    other_team_migration_min_m: float = 8.0
    other_team_migration_window_s: float = 12.0
    restart_ball_dist_m: float = 1.4
    restart_ball_speed_mps: float = 2.8
    restart_max_obs_gap_s: float = 1.0
    restart_search_s: float = 30.0
    dedup_s: float = 7.0
    max_events: int = 200

P = Params()


def numeric(df, name, default=np.nan):
    if name not in df: return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)


def load_data(step:int):
    import config
    if step < 1: raise ValueError('--step debe ser >= 1')
    if int(config.FPS) != P.fps: raise ValueError('FPS no coincide')
    n=int(config.N_VIDEO_FRAMES)
    mat=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    if len(mat)!=n or not np.array_equal(numeric(mat,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv no se alinea 0..N-1')
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw.columns:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('ATD sin frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(numeric(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD y matriz desalineados')
    # No duplicar observaciones: el muestreo efectivo ocurre cada 0.8s o más.
    stride=max(step,int(np.ceil(P.observation_s*P.fps/step))*step)
    frames=np.arange(0,n,stride,dtype=int)
    periods=np.array([config.get_period(int(f)) for f in frames])
    use=np.isin(periods,['FIRST_HALF','SECOND_HALF'])
    frames,periods=frames[use],periods[use]
    xx=[f'{k}_x' for k in range(1,47)]
    yy=[f'{k}_y' for k in range(1,47)]
    if not all(k in raw.columns for k in xx+yy):raise ValueError('Slots 1..46 incompletos')
    x=raw.loc[frames,xx].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.length_m
    y=raw.loc[frames,yy].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.width_m
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.length_m)&(y>=0)&(y<=P.width_m)
    x[~valid]=np.nan;y[~valid]=np.nan
    bx=numeric(mat,'ball_x_normalized')[frames]*P.length_m
    by=numeric(mat,'ball_y_normalized')[frames]*P.width_m
    obs=(numeric(mat,'ball_observed',0)[frames]>.5)&(numeric(mat,'ball_imputed',0)[frames]<.5)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=frames,periods=periods,x=x,y=y,bx=bx,by=by,obs=obs)


def pts(d,i,team):
    s=slice(0,24) if team==0 else slice(24,46)
    a=np.column_stack([d['x'][i,s],d['y'][i,s]])
    return a[np.isfinite(a).all(axis=1)]


def boundary_distance(p):
    x,y=float(p[0]),float(p[1])
    return min(x,P.length_m-x,y,P.width_m-y)


def entry_gate_distance(p):
    return min(abs(float(p[0])-P.length_m/2),1e6) if min(float(p[1]),P.width_m-float(p[1]))<=P.entry_near_line_m else np.inf


def inward_y(p):return min(float(p[1]),P.width_m-float(p[1]))


def match_pairs(a,b,dt):
    """Emparejamiento de proximidad 1:1, sin afirmar identidad personal."""
    if not len(a) or not len(b):return [],list(range(len(a))),list(range(len(b)))
    dist=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    cutoff=P.match_extra_m+P.max_match_speed_mps*dt
    cost=np.where(dist<=cutoff,dist,1e6)
    row,col=linear_sum_assignment(cost)
    pairs=[(int(i),int(j)) for i,j in zip(row,col) if cost[i,j]<1e5]
    used_a={i for i,j in pairs};used_b={j for i,j in pairs}
    return pairs,[i for i in range(len(a)) if i not in used_a],[j for j in range(len(b)) if j not in used_b]


def build_tracklets(d,team):
    """Tracklets locales discontinuos: el identificador es interno, nunca el slot ATD."""
    frames=d['frames']; periods=d['periods']; tracks=[];active={};next_id=0
    for i in range(len(frames)):
        a=pts(d,i,team)
        if i==0 or periods[i]!=periods[i-1] or (frames[i]-frames[i-1])/P.fps>P.max_gap_s:
            active={}
        else:
            prev_ids=list(active)
            prev=np.asarray([active[k][-1][1] for k in prev_ids]).reshape(-1,2)
            dt=(frames[i]-frames[i-1])/P.fps
            pairs,_,new=match_pairs(prev,a,dt)
            fresh={}
            for pi,bi in pairs:
                tid=prev_ids[pi];active[tid].append((i,a[bi].copy()));fresh[tid]=active[tid]
            active=fresh
            for bi in new:
                tid=next_id;next_id+=1
                t=[(i,a[bi].copy())];tracks.append(t);active[tid]=t
            continue
        for q in a:
            tid=next_id;next_id+=1
            t=[(i,q.copy())];tracks.append(t);active[tid]=t
    return tracks


def track_features(d,tracks):
    """Señales de salida en cualquier límite y entrada desde línea media por banda."""
    f=d['frames']; per=d['periods']; exits=[];entries=[]
    for t in tracks:
        start,end=t[0][0],t[-1][0]
        duration=(f[end]-f[start])/P.fps
        if duration<P.min_track_s:continue
        p0,pe=t[0][1],t[-1][1]
        seg_exit=[(idx,p) for idx,p in t if f[end]-f[idx]<=P.exit_lookback_s*P.fps]
        exit_old=seg_exit[0][1]
        toward=boundary_distance(exit_old)-boundary_distance(pe)
        near=boundary_distance(pe)
        at_period_end=(end==len(f)-1 or end<len(f)-1 and per[end+1]!=per[end])
        enough_after=(len(f)>end+1 and f[end+1]-f[end]<=P.max_gap_s*P.fps and
                      f[-1]-f[end]>=P.leave_disappearance_s*P.fps)
        if near<=P.boundary_near_m and toward>=P.exit_edge_progress_m and not at_period_end and enough_after:
            exits.append(dict(frame=int(f[end]),position=[float(v) for v in pe],
                              progress_m=round(float(toward),2),track_duration_s=round(duration,1),
                              t=t,period=str(per[end])))
        close_to_half=abs(float(p0[0])-P.length_m/2)<=P.midfield_entry_x_m
        close_to_side=inward_y(p0)<=P.entry_near_line_m
        seg_entry=[(idx,p) for idx,p in t if f[idx]-f[start]<=P.entry_lookahead_s*P.fps]
        deepest=max(inward_y(p) for _,p in seg_entry)
        entry_progress=deepest-inward_y(p0)
        guard=(f[start]-f[0])/P.fps>=P.halftime_guard_s
        # Comprobar distancia al inicio de cada período, no al principio del array.
        period_idxs=np.flatnonzero(per==per[start])
        guard=(f[start]-f[period_idxs[0]])/P.fps>=P.halftime_guard_s
        if close_to_half and close_to_side and duration>=P.entry_persistence_s and entry_progress>=P.entry_inward_m and guard:
            entries.append(dict(frame=int(f[start]),position=[float(v) for v in p0],
                                progress_m=round(float(entry_progress),2),track_duration_s=round(duration,1),
                                t=t,period=str(per[start])))
    return exits,entries


def context_counts(d,team,frame,span_s=6):
    f=d['frames'];idx=int(np.searchsorted(f,frame))
    lo=max(0,int(np.searchsorted(f,frame-span_s*P.fps)))
    hi=min(len(f),int(np.searchsorted(f,frame+span_s*P.fps))+1)
    cnt=[len(pts(d,i,team)) for i in range(lo,hi) if d['periods'][i]==d['periods'][idx] if idx<len(f)]
    return dict(min=min(cnt) if cnt else None,max=max(cnt) if cnt else None,
                median=float(np.median(cnt)) if cnt else None)


def other_team_migration(d,team,start_frame,end_frame):
    """Veto SUAVE: evitar confundir descanso/hidratación con sustitución."""
    f=d['frames'];a=int(np.searchsorted(f,start_frame));b=int(np.searchsorted(f,end_frame))
    a=min(a,len(f)-1);b=min(b,len(f)-1)
    other=1-team
    x=pts(d,a,other);y=pts(d,b,other)
    if min(len(x),len(y))<P.min_visible_team:return None
    return float(np.linalg.norm(np.median(y,axis=0)-np.median(x,axis=0)))


def ball_restart(d,frame):
    f=d['frames'];start=int(np.searchsorted(f,frame))
    end=int(np.searchsorted(f,frame+P.restart_search_s*P.fps,side='right'))
    obs=np.flatnonzero(d['obs'][start:end])+start
    for a,b in zip(obs[:-1],obs[1:]):
        dt=(f[b]-f[a])/P.fps
        if dt<=0 or dt>P.restart_max_obs_gap_s or d['periods'][a]!=d['periods'][b]:continue
        dist=float(np.hypot(d['bx'][b]-d['bx'][a],d['by'][b]-d['by'][a]))
        if dist>=P.restart_ball_dist_m and dist/dt>=P.restart_ball_speed_mps:
            return dict(from_frame=int(f[a]),to_frame=int(f[b]),dist_m=round(dist,2))
    return None


def detect(d):
    events=[]
    for team in (0,1):
        tracks=build_tracklets(d,team)
        exits,entries=track_features(d,tracks)
        possible=[]
        for ex in exits:
            for en in entries:
                dt=(en['frame']-ex['frame'])/P.fps
                if ex['period']!=en['period'] or not P.pair_min_delay_s<=dt<=P.pair_max_delay_s:continue
                # El jugador sustituido no necesariamente sale por mitad de campo.
                counts=context_counts(d,team,en['frame'])
                if counts['median'] is None or counts['median']<P.min_visible_team:continue
                mig=other_team_migration(d,team,ex['frame'],en['frame'])
                restart=ball_restart(d,en['frame'])
                # Sin vídeo ni IDs solo se puede hablar de compatibilidad.
                score=2 + int(ex['progress_m']>=5) + int(en['progress_m']>=7)
                score+=int(abs(dt)<=12)+int(mig is not None and mig<P.other_team_migration_min_m)
                score+=int(counts['max']<=P.max_visible_team)+int(bool(restart))
                level='SUBSTITUTION_PATTERN_STRONG' if score>=6 else 'SUBSTITUTION_PATTERN_CANDIDATE'
                possible.append(dict(type='TH13_SUBSTITUTION',classification=level,score_heuristic=int(score),
                                     team='HOME' if team==0 else 'AWAY',
                                     exit_frame=ex['frame'],entry_frame=en['frame'],
                                     exit_position_m=ex['position'],entry_position_m=en['position'],
                                     exit_toward_boundary_m=ex['progress_m'],entry_inward_m=en['progress_m'],
                                     seconds_exit_to_entry=round(dt,2),other_team_centroid_shift_m=(round(mig,2) if mig is not None else None),
                                     team_visible_counts=counts,observed_ball_motion_after=restart,
                                     possible_D_start_window=None,possible_D_end_window=None,
                                     warning='No hay IDs persistentes: trayectorias locales no certifican sustitución. Movimiento del balón NO confirma el instante de saque.'))
        possible.sort(key=lambda z:(-z['score_heuristic'],z['entry_frame']))
        accepted=[]
        for q in possible:
            if any(abs(q['entry_frame']-k['entry_frame'])<P.dedup_s*P.fps and
                   abs(q['exit_frame']-k['exit_frame'])<P.dedup_s*P.fps for k in accepted):continue
            accepted.append(q)
        events.extend(accepted)
    events.sort(key=lambda z:z['entry_frame'])
    return events[:P.max_events]


def synth():
    """Secuencia artificial: un jugador se acerca a banda y desaparece; otro surge en medular."""
    fps=P.fps;frames=np.arange(0,75*fps,20,dtype=int);n=len(frames)
    x=np.full((n,46),np.nan);y=np.full((n,46),np.nan)
    for i,f in enumerate(frames):
        sec=f/fps
        for t,offset in ((0,0),(24,0)):
            for j in range(10):
                x[i,t+j]=13+j*8 + (4 if t else 0)
                y[i,t+j]=16+(j%5)*7+(2 if t else 0)
        # Saliente: desde banda central hasta fuera de la cobertura.
        if sec<30:
            x[i,10]=52
            y[i,10]=max(1,25-max(0,sec-20)*3.0)
        # Entrante: arranca después y se interna en campo.
        if sec>=34:
            x[i,11]=52
            y[i,11]=min(30,1+(sec-34)*2.0)
    periods=np.array(['FIRST_HALF']*n)
    d=dict(frames=frames,periods=periods,x=x,y=y,bx=np.full(n,np.nan),by=np.full(n,np.nan),obs=np.zeros(n,bool))
    return d


def self_test():
    d=synth();events=detect(d)
    assert any(e['team']=='HOME' for e in events), 'No detectó patrón positivo'
    z=synth();z['x'][:,11]=np.nan;z['y'][:,11]=np.nan
    assert not detect(z),'Falso positivo sin entrada'
    w=synth();w['x'][:,10]=np.nan;w['y'][:,10]=np.nan
    assert not detect(w),'Falso positivo sin salida'
    print('TH13 self-test OK: salida + entrada; negativos independientes')


def main():
    ap=argparse.ArgumentParser(description='TH13 diagnóstico de sustituciones, sin IDs persistentes')
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',type=Path,default=Path('local_outputs/runs'))
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    d=load_data(args.step);ev=detect(d)
    args.out.mkdir(parents=True,exist_ok=True)
    outfile=args.out/'th13_substitution_diagnostic.json'
    payload=dict(detector='TH13',params=asdict(P),observations=len(d['frames']),events=ev,
                 caveat='Diagnóstico con tracklets, nunca prueba de identidad, acto arbitral ni fronteras D')
    outfile.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'TH13: {len(ev)} candidatos; {outfile}')

if __name__=='__main__':main()
