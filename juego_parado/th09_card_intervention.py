"""TH09 — Intervención arbitral/disciplinaria COMPATIBLE (no tarjeta confirmada).

Uso:
  python th09_card_intervention.py --self-test
  python th09_card_intervention.py --step 5 --out local_outputs/runs

Sólo diagnóstico JSON; no modifica run.py ni asigna H/A/D.
Requiere config.py, starter.load_atd, matrix_v4.csv, numpy, pandas.
Los slots ATD no son identidades estables. Una aglomeración NO prueba tarjeta.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import config

FPS = int(config.FPS)

@dataclass(frozen=True)
class Settings:
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0
    min_visible_each_team: int = 7
    lag_s: float = 1.0
    spatial_match_max_m: float = 7.0
    quiet_speed_mps: float = 1.25
    baseline_active_speed_mps: float = 1.75
    slowdown_drop_mps: float = 0.55
    before_window_s: float = 6.0
    after_window_s: float = 4.0
    min_preparation_s: float = 2.0
    max_snapshot_gap_s: float = 0.8
    # Congestión local; no identifica árbitro porque no figura como slot conocido.
    cluster_radius_m: float = 7.0
    cluster_min_count: int = 4
    cluster_min_both_teams: int = 1
    gathering_increase: int = 2
    crowd_search_radius_m: float = 14.0
    crowd_near_ball_m: float = 12.0
    ball_still_radius_m: float = 1.3
    ball_still_s: float = 1.2
    ball_max_obs_gap_s: float = 1.0
    observed_kick_min_m: float = 1.15
    observed_kick_min_speed_mps: float = 2.0
    recovery_speed_mps: float = 1.7
    restart_horizon_s: float = 45.0
    start_lookback_s: float = 12.0
    max_episode_s: float = 75.0
    dedup_s: float = 9.0
    provider_near_s: float = 12.0
    # Scores: entrada solo con transición, no con quietud absoluta.
    min_score: int = 4
    strong_score: int = 7

P=Settings()

def number(df,col,fill=np.nan):
    if col not in df:return np.full(len(df),fill,dtype=float)
    return pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)

def load_tracking(step):
    if step < 1:raise ValueError('step >=1')
    mat=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    n=int(config.N_VIDEO_FRAMES)
    if len(mat)!=n or not np.array_equal(number(mat,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv no alinea frame 0..N-1')
    frames=np.arange(0,n,step,dtype=int)
    periods=np.array([config.get_period(int(f)) for f in frames])
    good_period=np.isin(periods,['FIRST_HALF','SECOND_HALF'])
    frames=frames[good_period];periods=periods[good_period]
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('ATD sin frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(number(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD no corresponde a frames 1..N')
    xc=[f'{i}_x' for i in range(1,47)];yc=[f'{i}_y' for i in range(1,47)]
    if any(c not in raw for c in xc+yc):raise ValueError('ATD sin slots 1..46')
    x=raw.loc[frames,xc].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_length_m
    y=raw.loc[frames,yc].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_width_m
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.pitch_length_m)&(y>=0)&(y<=P.pitch_width_m)
    x[~valid]=np.nan;y[~valid]=np.nan
    bx=number(mat,'ball_x_normalized')[frames]*P.pitch_length_m
    by=number(mat,'ball_y_normalized')[frames]*P.pitch_width_m
    obs=(number(mat,'ball_observed',0)[frames]>.5)&(number(mat,'ball_imputed',0)[frames]<.5)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=frames,periods=periods,x=x,y=y,valid=valid,bx=bx,by=by,obs=obs)

def load_provider():
    try:
        from starter.load_smart_tagging import load
        raw=load(game=config.GAME)
    except (ImportError,FileNotFoundError,KeyError,ValueError,OSError,TypeError):return []
    out=[]
    for e in raw:
        if not isinstance(e,dict):continue
        tags=e.get('tags') or {}
        kind=' '.join(str(z) for z in (tags.get('Type',''),e.get('type',''),e.get('code',''))).lower()
        # Sólo coincidencia explícita; el proveedor puede no registrar tarjetas.
        if not any(s in kind for s in ('yellow card','red card','booking','caution','sending off','send off','tarjeta','foul','free kick')):continue
        try:a=int(e['start_frame']);b=int(e['end_frame'])
        except (KeyError,ValueError,TypeError):continue
        if b<=a:continue
        out.append(dict(start_frame=a,end_exclusive=b,raw_type=kind,source='SMART_TAGGING_NOT_GT'))
    return sorted(out,key=lambda x:x['start_frame'])

def _points(d,i,team=None):
    sl=slice(None) if team is None else (slice(0,24) if team==0 else slice(24,46))
    xx=d['x'][i,sl];yy=d['y'][i,sl]
    points=np.column_stack((xx,yy));return points[np.isfinite(points).all(axis=1)]

def _nn_speed(a,b,seconds):
    # Correspondencia ESPACIAL local: no identifica futbolistas. Simetrización reduce
    # anomalías por swaps/slots vacíos. No convertir un slot en trayectoria real.
    if min(len(a),len(b))<P.min_visible_each_team or seconds<=0:return np.nan
    dif=a[:,None,:]-b[None,:,:]
    dist=np.linalg.norm(dif,axis=2)
    m1=np.min(dist,axis=1);m2=np.min(dist,axis=0)
    matches=np.r_[m1[m1<=P.spatial_match_max_m],m2[m2<=P.spatial_match_max_m]]
    if len(matches)<2*P.min_visible_each_team:return np.nan
    return float(np.median(matches)/seconds)

def spatial_features(d,lag_s=None):
    """Features por observación, usando tiempo real y secuencias. Sin duplicar frames."""
    lag_s=P.lag_s if lag_s is None else lag_s
    frames=d['frames'];n=len(frames)
    speed=np.full((n,2),np.nan)
    visible=np.zeros((n,2),dtype=int)
    # Centro de la aglomeración más grande y recuento mixto, NO referencia arbitral.
    cluster_count=np.zeros(n,int);cluster_mixed=np.zeros(n,bool)
    centers=np.full((n,2),np.nan)
    for i in range(n):
        teams=[_points(d,i,0),_points(d,i,1)]
        visible[i,:]=[len(x) for x in teams]
        if min(visible[i])<P.min_visible_each_team:continue
        j=np.searchsorted(frames,frames[i]-round(lag_s*FPS),side='right')-1
        if j>=0 and j<i and d['periods'][j]==d['periods'][i]:
            dt=(frames[i]-frames[j])/FPS
            if lag_s*.65<=dt<=lag_s*1.55:
                for t in (0,1):speed[i,t]=_nn_speed(teams[t],_points(d,j,t),dt)
        pts=np.vstack(teams)
        affiliation=np.r_[np.zeros(len(teams[0]),int),np.ones(len(teams[1]),int)]
        # Centro candidato entre puntos observados; densidad local de 7 m.
        distance=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
        is_near=distance<=P.cluster_radius_m
        counts=is_near.sum(axis=1)
        mix=((is_near& (affiliation[None,:]==0)).sum(axis=1)>=P.cluster_min_both_teams)&((is_near&(affiliation[None,:]==1)).sum(axis=1)>=P.cluster_min_both_teams)
        ix=int(np.argmax(counts+mix.astype(int)*0.25))
        cluster_count[i]=int(counts[ix]);cluster_mixed[i]=bool(mix[ix]);centers[i]=pts[ix]
    return dict(speed=speed,visible=visible,cluster_count=cluster_count,cluster_mixed=cluster_mixed,centers=centers)

def _window_indices(frames,i,seconds,before=True):
    if before:
        return np.flatnonzero((frames>=frames[i]-seconds*FPS)&(frames<frames[i]))
    return np.flatnonzero((frames>frames[i])&(frames<=frames[i]+seconds*FPS))

def _observed_ball_stable(d,i):
    f=d['frames'];inds=np.flatnonzero(d['obs']&(np.abs(f-f[i])<=P.ball_still_s*FPS))
    if len(inds)<3:return False
    if (f[inds[-1]]-f[inds[0]])/FPS < P.ball_still_s*.75:return False
    if np.max(np.diff(f[inds]))/FPS>P.ball_max_obs_gap_s:return False
    xy=np.column_stack((d['bx'][inds],d['by'][inds]));center=np.median(xy,axis=0)
    return bool(np.quantile(np.linalg.norm(xy-center,axis=1),.85)<=P.ball_still_radius_m)

def _ball_restart(d,i,last):
    f=d['frames'];inds=np.flatnonzero(d['obs']&(np.arange(len(f))>=i)&(np.arange(len(f))<=last))
    for a,b in zip(inds[:-1],inds[1:]):
        if d['periods'][a]!=d['periods'][b]:continue
        dt=(f[b]-f[a])/FPS
        if dt<=0 or dt>P.ball_max_obs_gap_s:continue
        displacement=float(np.hypot(d['bx'][b]-d['bx'][a],d['by'][b]-d['by'][a]))
        if displacement>=P.observed_kick_min_m and displacement/dt>=P.observed_kick_min_speed_mps:
            # Movimiento de balón no demuestra un saque; consultar fase y alternativas.
            return dict(lower_frame=int(f[a]),upper_frame=int(f[b]),observed_ball_displacement_m=round(displacement,2),
                        interpretation='OBSERVED_BALL_MOTION_NOT_PROVEN_RESTART')
    return None

def detect(d,providers=None):
    providers=providers or []
    ft=spatial_features(d);frames=d['frames'];n=len(frames)
    available=np.isfinite(ft['speed'])
    speed_count=available.sum(axis=1)
    mean_speed=np.divide(np.nansum(ft['speed'],axis=1),speed_count,
                         out=np.full(n,np.nan),where=speed_count==2)
    # Evitar RuntimeWarning por ausencia de tracking en filas: serie conservadora.
    valid=np.isfinite(mean_speed)&np.all(ft['visible']>=P.min_visible_each_team,axis=1)
    quiet=valid&(mean_speed<=P.quiet_speed_mps)
    crowd=valid&(ft['cluster_count']>=P.cluster_min_count)&ft['cluster_mixed']
    items=[];prev_end=-1
    # Candidatos cronológicos: COMIENZO de una espera, no cada frame del episodio.
    for i in range(1,n):
        if frames[i]<=prev_end+P.dedup_s*FPS:continue
        if not (quiet[i] and (crowd[i] or _observed_ball_stable(d,i))):continue
        if quiet[i-1] and d['periods'][i]==d['periods'][i-1]:continue
        before=_window_indices(frames,i,P.before_window_s,True)
        before=before[d['periods'][before]==d['periods'][i]]
        if len(before)<3:continue
        b_speed=mean_speed[before];b_speed=b_speed[np.isfinite(b_speed)]
        if len(b_speed)<3:continue
        active_before=float(np.quantile(b_speed,.75))
        slowed=active_before>=P.baseline_active_speed_mps and active_before-mean_speed[i]>=P.slowdown_drop_mps
        # No activar evento sólo por gente quieta o grupos estáticos.
        if not slowed:continue
        end_search=min(n-1,np.searchsorted(frames,frames[i]+P.max_episode_s*FPS,side='right')-1)
        end_search=max(i,end_search)
        # Formación/quietud persistente en al menos 2s y varias observaciones independientes.
        hold=np.flatnonzero((frames>=frames[i])&(frames<=frames[i]+P.min_preparation_s*FPS)&(d['periods']==d['periods'][i]))
        if len(hold)<3 or (frames[hold[-1]]-frames[hold[0]])/FPS<P.min_preparation_s*.9:continue
        if np.mean(quiet[hold])<.6:continue
        mixed_hold=np.mean(crowd[hold])>=.4
        still_hold=np.mean([_observed_ball_stable(d,int(j)) for j in hold[::max(1,len(hold)//5)]])>=.5
        if not (mixed_hold or still_hold):continue
        cluster_delta=int(ft['cluster_count'][i]-np.median(ft['cluster_count'][before[-min(8,len(before)):]]))
        grew=cluster_delta>=P.gathering_increase
        # Para aglomeración: centro relativamente estable durante preparación.
        cc=ft['centers'][hold];cc=cc[np.isfinite(cc).all(axis=1)]
        stable_center=bool(len(cc)>=3 and np.median(np.linalg.norm(cc-np.median(cc,axis=0),axis=1))<=P.crowd_search_radius_m/2)
        stable_ball=bool(still_hold)
        nearby=[e for e in providers if abs(e['start_frame']-int(frames[i]))<=P.provider_near_s*FPS]
        provider_card=any(any(k in p['raw_type'] for k in ('yellow card','red card','booking','caution','sending off','send off','tarjeta')) for p in nearby)
        score=2*int(slowed)+int(mixed_hold)+int(grew)+int(stable_center)+2*int(stable_ball)+2*int(provider_card)
        if score<P.min_score:continue
        # Estimar final: primer retorno sostenido a actividad, preferentemente con balón.
        last=min(end_search,np.searchsorted(frames,frames[i]+P.restart_horizon_s*FPS,side='right')-1)
        moving=np.flatnonzero((np.arange(n)>i)&(np.arange(n)<=last)&valid&(mean_speed>=P.recovery_speed_mps)&(d['periods']==d['periods'][i]))
        recovery=None
        if len(moving):
            for j in moving:
                ahead=np.flatnonzero((frames>=frames[j])&(frames<=frames[j]+FPS)&(d['periods']==d['periods'][j]))
                if len(ahead)>=3 and np.mean((mean_speed[ahead]>=P.recovery_speed_mps)&valid[ahead])>=.5:
                    recovery=int(j);break
        upto=recovery if recovery is not None else last
        ball_move=_ball_restart(d,i,upto)
        # Observed movement during preparation may be a moved ball, not restart.
        if ball_move and ball_move['upper_frame']<frames[hold[-1]]:ball_move=None
        # Bounding de inicio: entre último tramo activo e inicio de quietud, no un -N fijo.
        past=np.flatnonzero((frames>=frames[i]-P.start_lookback_s*FPS)&(frames<frames[i])&(d['periods']==d['periods'][i])&valid&(mean_speed>=P.baseline_active_speed_mps))
        start_bracket=dict(lower_frame=int(frames[past[-1]]) if len(past) else None,upper_frame=int(frames[i]),
                           basis='LAST_ACTIVE_TO_FIRST_STABLE',exact=False)
        # No afirmar fin de D sin evidencia de saque. Recuperación = indicio, balón = bracket compatible.
        end_bracket=ball_move or (dict(lower_frame=int(frames[max(i,recovery-1)]),upper_frame=int(frames[recovery]),
                                            basis='COLLECTIVE_REACTIVATION_ONLY',exact=False) if recovery is not None else None)
        classification='DISCIPLINARY_CONTEXT_SUPPORTED_NOT_CARD_PROVEN' if provider_card else ('INTERVENTION_COMPATIBLE' if score>=P.strong_score else 'STOPPAGE_CANDIDATE')
        output=dict(type='CARD_INTERVENTION',classification=classification,
                    anchor_frame=int(frames[i]),period=str(d['periods'][i]),score=int(score),
                    crowd_size=int(ft['cluster_count'][i]),crowd_mixed=bool(mixed_hold),
                    cluster_growth=int(cluster_delta),cluster_center_stable=stable_center,
                    mean_spatial_speed_mps=round(float(mean_speed[i]),3),active_before_mps=round(active_before,3),
                    ball_still_observed=stable_ball,provider_card_mention=bool(provider_card),
                    provider_context=nearby,dead_start_bracket=start_bracket,dead_end_bracket=end_bracket,
                    potential_stop_end_frame=int(frames[recovery]) if recovery is not None else None,
                    card_seen=False,referee_seen=False,verified_card=False,
                    evidence=['ACTIVITY_TO_QUIET_TRANSITION']+(['MIXED_TEAM_GATHERING'] if mixed_hold else [])+
                             (['CROWD_GROWTH'] if grew else [])+(['STABLE_CLUSTER'] if stable_center else [])+
                             (['OBSERVED_STABLE_BALL'] if stable_ball else [])+
                             (['PROVIDER_DISCIPLINARY_CONTEXT'] if provider_card else []),
                    caveat='No referee/card/whistle track; grouping is non-specific. Restart type must be arbitrated with TH01–TH08.')
        items.append(output)
        # saltar la pausa detectada pero no futuras intervenciones separadas
        prev_end=int(frames[recovery]) if recovery is not None else int(frames[i]+P.min_preparation_s*FPS)
    stats=dict(samples=n,valid_speed=int(valid.sum()),quiet_samples=int(quiet.sum()),crowded_samples=int(crowd.sum()),
               provider_context_events=len(providers),candidates=len(items))
    return items,stats

def self_test():
    f=np.arange(0,20*FPS,5);n=len(f)
    # jugadores constantes; no falsa intervención por quietud.
    x=np.tile(np.r_[np.linspace(5,95,24),np.linspace(7,97,22)],(n,1))
    y=np.tile(np.r_[np.linspace(8,60,24),np.linspace(9,59,22)],(n,1))
    d=dict(frames=f,x=x,y=y,valid=np.ones((n,46),bool),bx=np.full(n,np.nan),by=np.full(n,np.nan),obs=np.zeros(n,bool),periods=np.array(['FIRST_HALF']*n))
    feats=spatial_features(d)
    assert np.nanmax(feats['speed'])<.01
    detected,_=detect(d,[])
    assert not detected, 'Una disposición inmóvil NO demuestra sanción'
    # Balón moviéndose pero sin recuperación de velocidad no es un final confirmado.
    d['obs'][20:23]=True;d['bx'][20:23]=[30.,31.,33.];d['by'][20:23]=34.
    assert _ball_restart(d,20,22) is not None
    print('TH09 synthetic tests OK')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',default='local_outputs/runs')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    d=load_tracking(args.step);providers=load_provider();events,stats=detect(d,providers)
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'th09_card_intervention_step{args.step}.json'
    payload=dict(detector='TH09_CARD_INTERVENTION_V1',fps=FPS,config=asdict(P),stats=stats,
                 warning='NO CERTIFICA TARJETA NI DECISIÓN ARBITRAL; NO ASIGNA D',candidates=events)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda z:z.item() if isinstance(z,np.generic) else str(z)),encoding='utf-8')
    print(f'TH09: {len(events)} candidatos -> {path}')
    print(stats)
if __name__=='__main__':main()
