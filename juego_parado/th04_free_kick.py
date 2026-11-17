"""Detector especializado de balón parado. Diagnóstico: NO altera H/A/D.

Uso: python NOMBRE_SCRIPT.py [--step 5] [--out DIR]
Requiere numpy, pandas, config.py, starter.load_atd y (opcional) starter.load_smart_tagging.
Coordenadas ATD normalizadas -> metros; validar dimensiones de campo.
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
    aliases={'THROW_IN':('throw-in','throw in'),'CORNER':('corner kick',),'GOAL_KICK':('goal kick',),'FREE_KICK':('free kick','free-kick','freekick',)}
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


KIND='FREE_KICK'

# Umbrales exploratorios en metros, segundos y proporciones, no calibrados.
from dataclasses import dataclass, asdict
@dataclass(frozen=True)
class Params:
    candidate_radius_m: float = 3.2
    near_ball_radius_m: float = 7.0
    wall_ball_min_m: float = 4.0
    wall_ball_max_m: float = 17.0
    wall_width_m: float = 5.0
    wall_depth_m: float = 3.2
    wall_min_players: int = 3
    setup_seconds: float = 2.0
    history_seconds: float = 12.0
    restart_seconds: float = 18.0
    low_motion_mps: float = 1.2
    kick_speed_mps: float = 2.2
    kick_displacement_m: float = 1.0
    max_ball_gap_s: float = 0.8
    max_start_lookback_s: float = 15.0
    step_min_s: float = 0.15

P=Params()

def safe_count_near(d,px,py,radius):
    dist=np.hypot(d['x']-px[:,None],d['y']-py[:,None])
    mask=d['valid']&np.isfinite(px[:,None])&np.isfinite(py[:,None])&(dist<=radius)
    return team_counts(d,mask)

def local_matching_movement(d,seconds=1.0,cutoff=9.0):
    """Distancia al vecino más cercano del mismo equipo entre snapshots.
    Es movimiento de ocupación espacial, NO trayectoria de identidad real.
    Cobertura baja u oclusiones se invalidan. Matriz (tiempo, equipo).
    """
    lag=max(1,round(seconds*FPS/d['step']))
    out=np.full((len(d['frames']),2),np.nan)
    for i in range(lag,len(out)):
        j=i-lag
        if d['period'][i]!=d['period'][j]:continue
        dt=(d['frames'][i]-d['frames'][j])/FPS
        for team,sl in enumerate((slice(0,24),slice(24,46))):
            a=np.column_stack([d['x'][i,sl],d['y'][i,sl]])
            b=np.column_stack([d['x'][j,sl],d['y'][j,sl]])
            a=a[np.isfinite(a).all(axis=1)];b=b[np.isfinite(b).all(axis=1)]
            if len(a)<6 or len(b)<6 or abs(len(a)-len(b))>2:continue
            dist=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
            nearest=np.min(dist,axis=1)
            good=nearest<cutoff
            if good.sum()<6:continue
            out[i,team]=float(np.median(nearest[good])/dt)
    return out

def ball_speed_and_wait(d):
    vx,vy=observed_ball_motion(d,limit_s=P.max_ball_gap_s)
    speed=np.hypot(vx,vy)
    still=d['obs']&np.isfinite(speed)&(speed<0.8)
    # No movement claim during imputed/missing ball.
    return vx,vy,speed,still

def wall_geometry(d,px,py,team,goal_x):
    """Barrera aproximada: rivales alineados transversalmente al eje balón->portería.
    Se permiten paredes de 1/2 jugadores como apoyo, 3+ como señal fuerte.
    Coordenadas de cada snapshot, sin identidades persistentes.
    """
    n=len(d['frames']);count=np.zeros(n,dtype=int);score=np.zeros(n)
    sl=slice(0,24) if team==0 else slice(24,46)
    for i in np.flatnonzero(np.isfinite(px)&np.isfinite(py)):
        # vector longitudinal hacia portería objetivo; el balón puede estar en cualquier sector
        direction=np.array([goal_x-px[i],W/2-py[i]],float)
        norm=np.linalg.norm(direction)
        if norm<3:continue
        u=direction/norm;v=np.array([-u[1],u[0]])
        xy=np.column_stack((d['x'][i,sl],d['y'][i,sl]))
        xy=xy[np.isfinite(xy).all(axis=1)]
        if len(xy)<1:continue
        rel=xy-np.array([px[i],py[i]])
        forward=rel@u;cross=rel@v
        eligible=(forward>=P.wall_ball_min_m)&(forward<=P.wall_ball_max_m)
        ids=np.flatnonzero(eligible)
        best=0
        # sliding compact aligned rectangle; max count without assuming 9.15m exact
        for z in ids:
            compact=(np.abs(forward-forward[z])<=P.wall_depth_m/2)&(np.abs(cross-cross[z])<=P.wall_width_m/2)&eligible
            best=max(best,int(compact.sum()))
        count[i]=best
        score[i]=min(1.,best/max(P.wall_min_players,1))
    return count,score

def observed_kick(d,px,py,speed):
    """Golpeo localizado entre dos observaciones reales; evita inventarlo en huecos."""
    n=len(d['frames']);kick=np.zeros(n,dtype=bool)
    obs=np.flatnonzero(d['obs'])
    for a,b in zip(obs[:-1],obs[1:]):
        if d['period'][a]!=d['period'][b]:continue
        dt=(d['frames'][b]-d['frames'][a])/FPS
        if dt>P.max_ball_gap_s:continue
        if not np.isfinite(px[a]) or not np.isfinite(py[a]):continue
        dist0=np.hypot(d['bx'][a]-px[a],d['by'][a]-py[a])
        delta=np.hypot(d['bx'][b]-d['bx'][a],d['by'][b]-d['by'][a])
        if dist0<=2.2 and delta>=P.kick_displacement_m and speed[b]>=P.kick_speed_mps:
            kick[b]=True
    return kick

def detect(d,tags):
    n=len(d['frames']);mov=local_matching_movement(d,1.0)
    vx,vy,speed,still=ball_speed_and_wait(d)
    counts=safe_count_near(d,d['bx'],d['by'],P.candidate_radius_m)
    features={'ball_speed_observed_mps':speed,'ball_stationary_observed':still.astype(int),
              'BAR_occupancy_motion_mps':mov[:,0],'ATH_occupancy_motion_mps':mov[:,1],
              'BAR_near_ball_count':counts[:,0],'ATH_near_ball_count':counts[:,1]}
    episodes=[]
    # Evaluar ambas hipótesis de equipo ejecutor y ambas porterías objetivo.
    # El lado de ataque es una hipótesis geométrica, NO se presupone por periodo.
    for executing in (0,1):
      opponent=1-executing
      # Ancla de balón: posición observada y relativamente estable. Si no hay balón,
      # un evento sin barrera no puede localizarse geométricamente con seguridad.
      px=np.where(d['obs'],d['bx'],np.nan);py=np.where(d['obs'],d['by'],np.nan)
      # Hold last REAL ball observation for up to 3 s ONLY as a spatial hypothesis.
      # Never treat held position as an observed stationary ball or a kick.
      last=-1
      for ii in range(n):
          if d['obs'][ii]:last=ii
          elif last>=0 and d['period'][ii]==d['period'][last] and d['frames'][ii]-d['frames'][last]<=3*FPS:
              px[ii]=d['bx'][last];py[ii]=d['by'][last]
      near=safe_count_near(d,px,py,P.candidate_radius_m)[:,executing]>=1
      opp_near=counts[:,opponent]
      for goal_x in (0.,L):
        wall_count,wall_score=wall_geometry(d,px,py,opponent,goal_x)
        wall=wall_count>=P.wall_min_players
        # Espera: ocupación lenta en ambos equipos; no se requiere inmovilidad absoluta.
        slow=(np.isfinite(mov[:,executing])&np.isfinite(mov[:,opponent])&
              (mov[:,executing]<P.low_motion_mps)&(mov[:,opponent]<P.low_motion_mps))
        # La concentración sin balón y sin caída de movimiento NO activa candidato.
        formation=wall&near&slow
        formation_short=(wall_count>=2)&near&slow
        kick=observed_kick(d,np.where(d['obs'],d['bx'],np.nan),np.where(d['obs'],d['by'],np.nan),speed)
        tag=f'{"BAR" if executing==0 else "ATH"}_TO_{"LEFT" if goal_x==0 else "RIGHT"}'
        features[tag+'_wall_count']=wall_count
        features[tag+'_wall_score']=wall_score
        features[tag+'_slow_formation']=formation.astype(int)
        features[tag+'_kick_observed']=kick.astype(int)
        # Semillas por formación sostenida, o contexto de proveedor cercano al balón.
        setup_len=max(2,round(P.setup_seconds*FPS/d['step']))
        stable=rolling_fraction(formation,setup_len)>=.55
        starts=np.flatnonzero(stable & ~np.r_[False,stable[:-1]])
        # Anclas proveedor: sin convertir su rango en D, sólo contexto.
        provider_anchors=[]
        for e in tags:
            idx=int(np.searchsorted(d['frames'],e['start_frame']))
            if idx<n and d['period'][idx] in ('FIRST_HALF','SECOND_HALF'):
                provider_anchors.append(idx)
        anchors=sorted(set(starts.tolist()+provider_anchors))
        last_frame=-10**9
        for k in anchors:
            fr=int(d['frames'][k]);period=d['period'][k]
            if fr-last_frame<5*FPS:continue
            a,b=ranges_from_anchor(d,k,P.history_seconds,P.restart_seconds)
            rr=np.arange(a,b+1);rr=rr[d['period'][rr]==period]
            pre=rr[rr<=k];post=rr[rr>=k]
            if len(pre)<2 or len(post)<2:continue
            has_wall=bool(np.count_nonzero(wall[pre])>=2)
            has_setup=bool(np.count_nonzero(formation[pre])>=2)
            has_small_wall=bool(np.count_nonzero(formation_short[pre])>=2)
            has_wait=bool(np.count_nonzero(still[pre]&near[pre])>=2)
            provider=matched_provider(tags,fr,fr,4*FPS)
            if not (has_setup or (has_small_wall and has_wait) or (provider and has_wait and near[k])):
                continue
            # Observed kick after the anchor (can happen in setup snapshot).
            kicks=post[kick[post]]
            restart=int(kicks[0]) if len(kicks) else None
            # No ball: formation candidate, end is unknown. Do not fabricate D end.
            # Beginning: identify preceding collective slowdown and localized setup.
            look=max(1,round(P.max_start_lookback_s*FPS/d['step']))
            before=rr[(rr>=max(0,k-look))&(rr<=k)]
            slow_before=before[slow[before]]
            formation_before=before[formation[before]|formation_short[before]]
            onset=int(formation_before[0]) if len(formation_before) else int(slow_before[0]) if len(slow_before) else int(k)
            # No evidence of foul/whistle in tracking. Beginning remains a broad estimate.
            low_frame=max(int(d['frames'][rr[0]]),int(d['frames'][onset]-P.max_start_lookback_s*FPS))
            if has_setup:subtype='ORGANIZED_WALL'
            elif has_small_wall:subtype='ORGANIZED_SMALL_WALL'
            else:subtype='PROVIDER_SUPPORTED_NO_WALL'
            ev={'event_type':KIND,'subtype':subtype,'period':str(period),'location_id':tag,
                'executing_team_candidate':'Barcelona' if executing==0 else 'Athletic',
                'target_goal_candidate':'LEFT' if goal_x==0 else 'RIGHT',
                'anchor_frame':fr,'anchor_last_frame':fr,'anchor_count':1,
                'd_start_low':low_frame,'d_start_high':int(d['frames'][onset]),
                'd_start_method':'RETROSPECTIVE_FORMATION_ONSET_UNCERTAIN',
                'd_end_low':int(d['frames'][max(0,restart-1)]) if restart is not None else None,
                'd_end_high':int(d['frames'][restart]) if restart is not None else None,
                'd_end_method':'OBSERVED_BALL_DEPARTURE' if restart is not None else 'NOT_LOCALIZED',
                'wall_max_players':int(np.max(wall_count[pre])),
                'formation_independent_snapshots':int(np.count_nonzero(formation[pre])),
                'slow_independent_snapshots':int(np.count_nonzero(slow[pre])),
                'ball_stationary_snapshots':int(np.count_nonzero(still[pre])),
                'provider_events':len(provider),
                'tier':'SEQUENCE_CANDIDATE' if restart is not None and has_setup else 'FORMATION_CANDIDATE' if has_setup else 'POSSIBLE_ONLY',
                'evidence':';'.join(z for z,ok in [('WALL_GEOMETRY',has_wall),('COLLECTIVE_SLOWDOWN',bool(np.any(slow[pre]))),('BALL_STATIONARY',has_wait),('BALL_KICK',restart is not None),('PROVIDER_CONTEXT',bool(provider))] if ok)}
            episodes.append(ev);last_frame=fr
    # Dedupe hypotheses: same formation can fit both goal directions; retain
    # alternatives in evidence rather than silently treating both as separate events.
    episodes.sort(key=lambda z:(z['anchor_frame'],-z['wall_max_players'],z['executing_team_candidate']))
    dedup=[]
    for e in episodes:
        if any(abs(e['anchor_frame']-z['anchor_frame'])<=3*FPS and e['period']==z['period'] for z in dedup):
            continue
        dedup.append(e)
    return dedup,features,{'rule':'TH04_FORMATION_WALL_SLOWDOWN_RESTART',
       'parameter_status':'EXPLORATORY_NOT_CALIBRATED','params':asdict(P),
       'fast_free_kicks':'NOT_AUTONOMOUSLY_CONFIRMED_WITHOUT_PROVIDER_OR_RELIABLE_BALL',
       'no_ball_no_precise_end':True,'player_slots_are_not_identities':True,
       'warning':'Geometría de barrera es hipótesis; falta no observada; D no asignado'}

if __name__=='__main__':cli(detect)
