"""TH05: detector diagnostico de PENALTI. NO altera H/A/D.

Uso: python NOMBRE_SCRIPT.py [--step 5] [--out DIR]
Requiere numpy, pandas, config.py, starter.load_atd y (opcional) starter.load_smart_tagging.
Coordenadas ATD normalizadas -> metros; verificar orientacion, escala y campo.
ATD slots 1..24 Barcelona y 25..46 Athletic: no identidades permanentes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import config

L, W = 105.3, 68.0
FPS = 25
GOAL_HALF = 3.66
GOAL_AREA_HALF = GOAL_HALF + 5.5 # 9.16m desde centro
PENALTY_AREA_HALF = GOAL_HALF + 16.5 # 20.16m desde centro


def runs(mask):
    mask=np.asarray(mask,dtype=bool)
    changes=np.diff(np.r_[False,mask,False].astype('int8'))
    return list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)-1))


def asfloat(df, col, fill=np.nan):
    if col not in df: return np.full(len(df),fill,dtype=float)
    return pd.to_numeric(df[col],errors='coerce').to_numpy(dtype=float)


def nearest_before(ix, indices, horizon):
    p=np.searchsorted(indices,ix,side='right')-1
    if p>=0 and ix-indices[p]<=horizon:return int(indices[p])
    return None


def nearest_after(ix, indices, horizon):
    p=np.searchsorted(indices,ix,side='left')
    if p<len(indices) and indices[p]-ix<=horizon:return int(indices[p])
    return None


def valid_period(frame):
    return config.get_period(int(frame)) in ('FIRST_HALF','SECOND_HALF')


def load_data(step):
    if step<1:raise ValueError('step >= 1')
    file=Path(config.OUTPUT_DIR)/'matrix_v4.csv'
    v4=pd.read_csv(file,low_memory=False)
    n=len(v4)
    if n!=config.N_VIDEO_FRAMES or 'frame' not in v4 or not np.array_equal(asfloat(v4,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv debe coincidir exactamente con frames de vídeo 0..N-1')
    # Only actual game snapshots; never join events across halftime.
    frames=np.arange(0,n,step,dtype=int)
    frames=frames[np.array([valid_period(f) for f in frames])]
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('ATD sin frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(asfloat(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD no está alineado con los frames; no asumir correspondencia')
    col_x=[f'{j}_x' for j in range(1,47)];col_y=[f'{j}_y' for j in range(1,47)]
    missing=[c for c in col_x+col_y if c not in raw]
    if missing:raise ValueError('ATD sin slots necesarios: '+repr(missing[:4]))
    x=raw.loc[frames,col_x].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)*L
    y=raw.loc[frames,col_y].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float32)*W
    good=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=L)&(y>=0)&(y<=W)
    x[~good]=np.nan;y[~good]=np.nan
    bx=asfloat(v4,'ball_x_normalized')[frames]*L
    by=asfloat(v4,'ball_y_normalized')[frames]*W
    obs=asfloat(v4,'ball_observed',0)[frames]>.5
    imp=asfloat(v4,'ball_imputed',0)[frames]>.5
    gap=asfloat(v4,'ball_gap_length',np.nan)[frames]
    # V4 may clip to [0,1]. Never prove an off-pitch crossing from clipped positions.
    obs &=np.isfinite(bx)&np.isfinite(by)&~imp
    ball_quality=np.where(obs,1,np.where(imp,np.clip(.35*np.exp(-np.nan_to_num(gap,nan=100)/50),0,.35),0))
    period=np.array([config.get_period(int(f)) for f in frames])
    return {'frames':frames,'x':x,'y':y,'valid':good,'bx':bx,'by':by,'obs':obs,'imp':imp,
            'quality':ball_quality,'period':period,'full_len':n,'step':step,
            'pitch_outside_observable':bool(np.any(obs&((bx<0)|(bx>L)|(by<0)|(by>W))))}


def load_provider(kind):
    """Eventos proveedor como contexto, NO GT, nunca sus rangos se convierten en D."""
    try:
        from starter.load_smart_tagging import load
        raw=load(game=config.GAME)
    except (ImportError,OSError,ValueError,KeyError,FileNotFoundError) as exc:
        return [],'unavailable: '+str(exc)
    aliases={'THROW_IN':('throw-in','throw in'),'CORNER':('corner kick',),'GOAL_KICK':('goal kick',),'FREE_KICK':('free kick','free-kick','freekick',),'PENALTY':('penalty','penalty kick','penalty-kick')}
    records=[]
    for e in raw:
        meta=e.get('tags') or {}
        typ=str(meta.get('Type',e.get('type',''))).lower().strip()
        if typ not in aliases[kind]:continue
        try:a=int(e['start_frame']);b=int(e['end_frame'])
        except (ValueError,KeyError,TypeError):continue
        if b<=a:continue
        records.append({'start_frame':a,'end_exclusive':b,'type':kind,
            'team':str(meta.get('Team','')),'side':str(meta.get('Side',''))})
    return records,'available'


def team_counts(d,mask):
    valid=d['valid']
    return np.stack([(mask[:,:24]&valid[:,:24]).sum(axis=1),
                     (mask[:,24:]&valid[:,24:]).sum(axis=1)],axis=1)


def team_centroids(d):
    """Centroide sólo cuando hay cobertura suficiente; NO equivale a la trayectoria de un jugador."""
    x,y=d['x'],d['y']
    out=np.full((len(x),2,2),np.nan,dtype=float)
    cov=np.zeros((len(x),2),dtype=int)
    for team,sl in enumerate((slice(0,24),slice(24,46))):
        z=np.isfinite(x[:,sl])&np.isfinite(y[:,sl]); cov[:,team]=z.sum(axis=1)
        good=cov[:,team]>=6
        out[good,team,0]=np.nansum(x[good,sl],axis=1)/cov[good,team]
        out[good,team,1]=np.nansum(y[good,sl],axis=1)/cov[good,team]
    return out,cov


def lagged(d,a,seconds):
    """a[t] - a[t-lag], incompatible periods and low coverage -> NaN."""
    lag=max(1,int(round(FPS*seconds/d['step'])))
    result=np.full_like(np.asarray(a,dtype=float),np.nan,dtype=float)
    result[lag:]=np.asarray(a[lag:],float)-np.asarray(a[:-lag],float)
    result[lag:][d['period'][lag:]!=d['period'][:-lag]]=np.nan
    return result


def rolling_fraction(mask,k):
    """Fracción de snapshots reales, no frames repetidos."""
    return pd.Series(np.asarray(mask,float)).rolling(max(1,k),min_periods=1).mean().to_numpy()


def smooth_sum(mask,k):
    return pd.Series(np.asarray(mask,float)).rolling(max(1,k),min_periods=1).sum().to_numpy()


def independent_phases(d,mask,seconds,minimum=2):
    k=max(1,int(round(FPS*seconds/d['step'])))
    return smooth_sum(mask,k)>=minimum


def observed_ball_motion(d,limit_s=1.):
    """Derivadas basadas EXCLUSIVAMENTE en observaciones reales contiguas (< limit_s).
    El balón ausente/imputado NO permite inferir velocidad exacta ni cruce.
    """
    n=len(d['frames']); vx=np.full(n,np.nan);vy=np.full(n,np.nan)
    idx=np.flatnonzero(d['obs'])
    if len(idx)<2:return vx,vy
    current=idx[1:]; previous=idx[:-1]
    dt=(d['frames'][current]-d['frames'][previous])/FPS
    ok=(dt>0)&(dt<=limit_s)&(d['period'][current]==d['period'][previous])
    c=current[ok];p=previous[ok];t=dt[ok]
    vx[c]=(d['bx'][c]-d['bx'][p])/t
    vy[c]=(d['by'][c]-d['by'][p])/t
    bad=np.hypot(vx,vy)>45
    vx[bad]=np.nan;vy[bad]=np.nan
    return vx,vy


def local_activity(d,center_x,center_y,radius=12):
    return np.sum(d['valid']&(np.hypot(d['x']-center_x[:,None],d['y']-center_y[:,None])<radius),axis=1)


def movement_to_point(d,px,py,seconds=2.):
    """Cambio de mínima distancia del conjunto al objetivo en dos snapshots:
    negativo=aproximación; NO asume identidad individual persistente.
    """
    x,y=d['x'],d['y'];dist=np.hypot(x-px[:,None],y-py[:,None])
    dist[~d['valid']]=np.nan
    # Use finite-safe minimum and preserve missing as NaN.
    mini=np.min(np.where(np.isfinite(dist),dist,np.inf),axis=1)
    mini[~np.isfinite(mini)]=np.nan
    return mini,lagged(d,mini,seconds)


def ranges_from_anchor(d,idx,lookback_s,lookahead_s):
    return max(0,idx-round(FPS*lookback_s/d['step'])),min(len(d['frames'])-1,idx+round(FPS*lookahead_s/d['step']))


def matched_provider(records,t0,t1,tolerance=5*FPS):
    return [r for r in records if r['start_frame']<=t1+tolerance and r['end_exclusive']>=t0-tolerance]


def merge_candidates(candidates,d,merge_s=2):
    """Merge only same LOCATION, PERIOD, and intervals separated <= gap; never fuse arbitrary D."""
    if not candidates:return []
    candidates.sort(key=lambda z:(z['period'],z['location_id'],z['anchor_frame']))
    out=[]
    for ev in candidates:
        if out and ev['location_id']==out[-1]['location_id'] and ev['period']==out[-1]['period'] \
            and ev['anchor_frame']-out[-1]['anchor_last_frame']<=merge_s*FPS and ev['anchor_frame']-out[-1]['anchor_frame']<=35*FPS:
            last=out[-1]
            last['anchor_last_frame']=ev['anchor_frame']
            last['evidence'].update(ev['evidence'])
            last['anchor_count']+=1
            continue
        out.append(ev)
    return out


def save_result(kind,d,episodes,features,provider,status,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    stem=kind.lower()+'_'+stamp
    f=pd.DataFrame({'frame':d['frames'],'period':d['period'],
       'ball_observed':d['obs'].astype('int8'),'ball_imputed':d['imp'].astype('int8'),
       'ball_quality':np.round(d['quality'],3)})
    for k,val in features.items():
        if len(val)!=len(f):raise ValueError(k+' length mismatch')
        f[k]=val
    paths={'features':out_dir/(stem+'_features.csv'),
           'episodes':out_dir/(stem+'_episodes.csv'),
           'provider':out_dir/(stem+'_provider.csv'),
           'summary':out_dir/(stem+'_summary.json')}
    f.to_csv(paths['features'],index=False)
    pd.DataFrame(episodes).to_csv(paths['episodes'],index=False)
    pd.DataFrame(provider).to_csv(paths['provider'],index=False)
    status={**status,'type':kind,'frames_analyzed':len(f),'candidate_episodes':len(episodes),
            'provider_events':len(provider),'outside_ball_observable':d['pitch_outside_observable'],
            'outputs':{k:str(v) for k,v in paths.items()},'warning':'NO altera H/A/D; los scores son heurísticos'}
    paths['summary'].write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in status.items() if k!='outputs'},indent=2,ensure_ascii=False))
    for k,path in paths.items():print(k,':',path)
    return status


def cli(detector):
    pa=argparse.ArgumentParser(description='Detector aislado de balón parado; NO genera D')
    pa.add_argument('--step',type=int,default=5,help='cada cuantos frames se toma una observacion independiente')
    pa.add_argument('--out',default=str(config.RUNS_DIR))
    args=pa.parse_args()
    d=load_data(args.step)
    tags,tag_status=load_provider(KIND)
    episodes,features,meta=detector(d,tags)
    save_result(KIND,d,episodes,features,tags,{**meta,'provider_status':tag_status,'step':args.step},args.out)



KIND = 'PENALTY'
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Params:
    # Umbrales de ingeniería exploratorios; unidades metros, segundos y proporciones.
    kicker_spot_m: float = 3.2
    keeper_goal_line_m: float = 2.6
    keeper_lateral_m: float = 4.7  # postes +/-3.66 + tolerancia de cobertura
    ball_spot_m: float = 1.4
    area_tolerance_m: float = 0.8
    minimum_other_visible: int = 10
    maximum_other_inside: int = 2
    outside_behind_fraction: float = 0.70
    outside_arc_fraction: float = 0.72
    stability_seconds: float = 2.0
    stable_snapshot_ratio: float = 0.55
    motion_threshold_mps: float = 1.8
    prior_seconds: float = 14.0
    future_seconds: float = 12.0
    max_retro_seconds: float = 18.0
    kick_max_gap_s: float = 0.8
    kick_min_displacement_m: float = 0.8
    kick_min_speed_mps: float = 1.8
    ball_observed_still_mps: float = 0.9

P = Params()

def local_occupancy_motion(d, seconds=1.0, cutoff=9.0):
    """Cambios de ocupación SIN atribuir identidad a slots de seguimiento.
    La velocidad de ocupación NO es la velocidad individual de un futbolista.
    """
    lag = max(1, int(round(seconds*FPS/d['step'])))
    out = np.full((len(d['frames']), 2), np.nan, float)
    for i in range(lag, len(out)):
        j=i-lag
        if d['period'][i] != d['period'][j]:continue
        dt = (d['frames'][i]-d['frames'][j])/FPS
        for team, sl in enumerate((slice(0,24),slice(24,46))):
            a=np.column_stack((d['x'][i,sl],d['y'][i,sl])); b=np.column_stack((d['x'][j,sl],d['y'][j,sl]))
            a=a[np.isfinite(a).all(axis=1)];b=b[np.isfinite(b).all(axis=1)]
            if len(a)<6 or len(b)<6 or abs(len(a)-len(b))>2: continue
            nearest=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2).min(axis=1)
            ok=nearest<cutoff
            if ok.sum()<6:continue
            out[i,team]=float(np.median(nearest[ok])/dt)
    return out

def geometric_features(d, goal_x, attacking_team, movement):
    """Evalúa posición de ejecutor, guardameta, zona vacía y espera exterior.
    Las selecciones son por proximidad en cada snapshot; nunca se atribuyen identidades.
    """
    n=len(d['frames']);x=d['x'];y=d['y'];valid=d['valid']
    spot_x=11. if goal_x==0 else L-11.
    py=W/2
    attack=slice(0,24) if attacking_team==0 else slice(24,46)
    defend=slice(24,46) if attacking_team==0 else slice(0,24)
    # Penalti hacia izquierda: los otros jugadores esperan a la derecha del punto; viceversa.
    behind=(x>=spot_x-0.6) if goal_x==0 else (x<=spot_x+0.6)
    # Área penal exacta, con tolerancia hacia fuera exclusivamente para diagnóstico.
    in_x=(x<=16.5+P.area_tolerance_m) if goal_x==0 else (x>=L-16.5-P.area_tolerance_m)
    in_area=in_x & (np.abs(y-py)<=20.16+P.area_tolerance_m) & valid
    outside_arc=np.hypot(x-spot_x,y-py)>=9.15-1.0
    kicker_d=np.hypot(x[:,attack]-spot_x,y[:,attack]-py)
    keeper_d=np.hypot(x[:,defend]-goal_x,y[:,defend]-py)
    kicker_d=np.where(valid[:,attack],kicker_d,np.inf)
    keeper_d=np.where(valid[:,defend],keeper_d,np.inf)
    kicker_local=kicker_d.argmin(axis=1); keeper_local=keeper_d.argmin(axis=1)
    kicker_dist=kicker_d[np.arange(n),kicker_local];keeper_dist=keeper_d[np.arange(n),keeper_local]
    kicker_index=kicker_local+(0 if attacking_team==0 else 24)
    keeper_index=keeper_local+(24 if attacking_team==0 else 0)
    near_kicker=(kicker_dist<=P.kicker_spot_m)
    keeper_x=np.abs(x[np.arange(n),keeper_index]-goal_x)
    keeper_y=np.abs(y[np.arange(n),keeper_index]-py)
    keeper_present=(keeper_x<=P.keeper_goal_line_m)&(keeper_y<=P.keeper_lateral_m)
    # Excluir el candidato ejecutor y guardameta del cómputo de todos los demás.
    others=valid.copy()
    others[np.arange(n),kicker_index]=False
    others[np.arange(n),keeper_index]=False
    n_other=others.sum(axis=1)
    area_other=(others & in_area).sum(axis=1)
    frac_behind=np.divide((others & behind).sum(axis=1),n_other,out=np.zeros(n,float),where=n_other>0)
    frac_arc=np.divide((others & outside_arc).sum(axis=1),n_other,out=np.zeros(n,float),where=n_other>0)
    area_clear=(n_other>=P.minimum_other_visible)&(area_other<=P.maximum_other_inside)
    outside_ok=(frac_behind>=P.outside_behind_fraction)&(frac_arc>=P.outside_arc_fraction)
    # Quietud en ambos equipos; si no tenemos cobertura no declaramos estabilidad.
    still=np.all(np.isfinite(movement),axis=1)&np.all(movement<P.motion_threshold_mps,axis=1)
    geometry=near_kicker&keeper_present&area_clear&outside_ok
    setup=geometry&still
    ball_spot=d['obs']&(np.hypot(d['bx']-spot_x,d['by']-py)<=P.ball_spot_m)
    return {'spot_x':spot_x,'kicker_distance_m':kicker_dist,'keeper_distance_m':keeper_dist,
         'keeper_line_distance_m':keeper_x,'others_in_area':area_other,
         'other_visible':n_other,'others_behind_fraction':frac_behind,
         'others_outside_arc_fraction':frac_arc,'geometry':geometry,'setup':setup,
         'still':still,'ball_at_spot_observed':ball_spot}

def observed_penalty_kicks(d, spot_x):
    """Golpeo sólo con dos observaciones REALES consecutivas, desde punto penal.
    El frame verdadero queda acotado, no se conoce exactamente entre snapshots.
    """
    n=len(d['frames']);hit=np.zeros(n,bool);before=np.full(n,-1,dtype=int)
    obs=np.flatnonzero(d['obs'])
    for a,b in zip(obs[:-1],obs[1:]):
        if d['period'][a]!=d['period'][b]:continue
        dt=(d['frames'][b]-d['frames'][a])/FPS
        if dt>P.kick_max_gap_s or dt<=0:continue
        if np.hypot(d['bx'][a]-spot_x,d['by'][a]-W/2)>P.ball_spot_m:continue
        dx=d['bx'][b]-d['bx'][a]; dy=d['by'][b]-d['by'][a]
        delta=np.hypot(dx,dy)
        # Al patear penalti, el balón ha de moverse hacia delante, no hacia atrás.
        forward=(dx<-.10) if spot_x < L/2 else (dx>.10)
        if forward and delta>=P.kick_min_displacement_m and delta/dt>=P.kick_min_speed_mps:
            hit[b]=True;before[b]=a
    return hit,before

def detect(d, tags):
    n=len(d['frames']);mov=local_occupancy_motion(d)
    features={'BAR_occupancy_motion_mps':mov[:,0], 'ATH_occupancy_motion_mps':mov[:,1]}
    episodes=[]
    for goal_x, side in ((0.,'LEFT'),(L,'RIGHT')):
      spot_x=11. if goal_x==0. else L-11.
      kicks,prev=observed_penalty_kicks(d,spot_x)
      for attacking in (0,1):
        defender=1-attacking
        z=geometric_features(d,goal_x,attacking,mov)
        tag=('BAR' if attacking==0 else 'ATH')+'_TO_'+side
        for name in ('kicker_distance_m','keeper_line_distance_m','others_in_area',
                     'other_visible','others_behind_fraction','others_outside_arc_fraction'):
            features[tag+'_'+name]=z[name]
        features[tag+'_geometry']=z['geometry'].astype('int8')
        features[tag+'_setup']=z['setup'].astype('int8')
        features[tag+'_ball_spot_observed']=z['ball_at_spot_observed'].astype('int8')
        stable_k=max(2,round(P.stability_seconds*FPS/d['step']))
        fraction=rolling_fraction(z['setup'],stable_k)
        # Requiere mínimo dos snapshots reales de geometría en la ventana.
        confirmed=(fraction>=P.stable_snapshot_ratio)&(smooth_sum(z['setup'],stable_k)>=2)
        begins=np.flatnonzero(confirmed & ~np.r_[False,confirmed[:-1]])
        # Smart Tagging sólo ofrece posible ancla; NO convierte un falso positivo en penalti.
        for k in begins:
            fr=int(d['frames'][k]);period=str(d['period'][k])
            ia,ib=ranges_from_anchor(d,int(k),P.prior_seconds,P.future_seconds)
            rr=np.arange(ia,ib+1);rr=rr[d['period'][rr]==period]
            pre=rr[rr<=k];post=rr[rr>=k]
            if len(pre)<2:continue
            good=pre[z['geometry'][pre]]
            if len(good)<2:continue
            # No inferir inicio exacto de la infracción desde una fotografía de preparación.
            first_setup=int(good[0]);look=max(0,fr-round(P.max_retro_seconds*FPS))
            prior=rr[(d['frames'][rr]>=look)&(rr<=first_setup)]
            # Para inicio, localizar primer declive de ocupación interior y primera quietud.
            area=z['others_in_area']; onset=first_setup
            for j in prior:
                future=min(j+max(1,round(2*FPS/d['step'])),first_setup)
                if future>j and area[j]-area[future]>=2:
                    onset=int(j);break
            # Las pruebas de ejecución deben ser DESPUÉS del establecimiento de la formación.
            endings=post[kicks[post]]
            end=int(endings[0]) if len(endings) else None
            provider=matched_provider(tags,fr,fr,5*FPS)
            observed_ball=bool(np.any(z['ball_at_spot_observed'][pre]))
            ev={'event_type':KIND,'subtype':'REGULATION_SHAPE','period':period,
                'location_id':side+'_PENALTY_MARK','attacking_team_candidate':'Barcelona' if attacking==0 else 'Athletic',
                'defending_team_candidate':'Athletic' if attacking==0 else 'Barcelona',
                'target_goal_candidate':side,'anchor_frame':fr,'anchor_last_frame':fr,'anchor_count':1,
                'd_start_low':int(d['frames'][prior[0]]) if len(prior) else max(fr-round(P.max_retro_seconds*FPS),0),
                'd_start_high':int(d['frames'][onset]),
                'd_start_method':'UNCERTAIN_RETROSPECTIVE_CLEARING_AND_SETUP',
                'd_end_low':int(d['frames'][prev[end]]) if end is not None else None,
                'd_end_high':int(d['frames'][end]) if end is not None else None,
                'd_end_method':'OBSERVED_BALL_KICK_INTERVAL' if end is not None else 'NOT_LOCALIZED',
                'stable_snapshots':int(np.count_nonzero(z['setup'][pre])),
                'other_players_inside_min':int(np.min(z['others_in_area'][pre])),
                'other_players_visible_median':float(np.median(z['other_visible'][pre])),
                'external_behind_median':float(np.median(z['others_behind_fraction'][pre])),
                'external_arc_median':float(np.median(z['others_outside_arc_fraction'][pre])),
                'ball_at_spot_observed':observed_ball, 'provider_events':len(provider),
                'tier':'GEOMETRY_AND_OBSERVED_KICK' if end is not None else 'GEOMETRY_FORMATION_ONLY',
                'evidence':'EXECUTOR_AT_SPOT;KEEPER_ON_GOAL_LINE;AREA_CLEAR;OTHERS_BEHIND_AND_OUTSIDE_ARC;COLLECTIVE_WAIT'+
                          (';BALL_SPOT_OBSERVED' if observed_ball else '')+
                          (';KICK_OBSERVED' if end is not None else '')+
                          (';PROVIDER_CONTEXT' if provider else '')}
            episodes.append(ev)
    # Duplicación en snapshots sucesivos y dos hipótesis de equipo: elegir mejor configuración.
    episodes.sort(key=lambda e:(e['anchor_frame'], -e['stable_snapshots'],
                 -int(e['ball_at_spot_observed']),e['attacking_team_candidate']))
    result=[]
    for ev in episodes:
        if any(ev['period']==prior['period'] and ev['location_id']==prior['location_id'] and
               abs(ev['anchor_frame']-prior['anchor_frame'])<=8*FPS for prior in result):continue
        result.append(ev)
    return result,features,{'rule':'TH05_PENALTY_GEOMETRY_AND_TEMPORAL_FORMATION',
       'params':asdict(P),'parameters':'EXPLORATORY_NOT_CALIBRATED',
       'slots_have_no_persistent_identity':True,'ball_missing_does_not_veto_formation':True,
       'no_ball_no_exact_kick_time':True,'not_HAD_output':True}

if __name__=='__main__':
    cli(detect)
