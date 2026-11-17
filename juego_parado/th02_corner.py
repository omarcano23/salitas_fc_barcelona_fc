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
    aliases={'THROW_IN':('throw-in','throw in'),'CORNER':('corner kick',),'GOAL_KICK':('goal kick',)}
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


KIND='CORNER'

def detect(d,tags):
    """TH02: llegada del ejecutor a esquina + reorganizacion area + ejecucion.
    Córner corto NO requiere que el balón salga lejos de la esquina.
    """
    n=len(d['frames']); vx,vy=observed_ball_motion(d)
    cen,cov=team_centroids(d)
    candidates=[]; features={};last_used={}
    for end in (0,1):
      for lateral in (0,1):
        label=f'{"L" if end==0 else "R"}{"B" if lateral==0 else "T"}'
        cx=0. if end==0 else L;cy=0. if lateral==0 else W
        pt=np.full(n,cx);qt=np.full(n,cy)
        ds=np.hypot(d['x']-cx,d['y']-cy)
        ds[~d['valid']]=np.inf
        closest=np.min(ds,axis=1)
        takers=(ds<=4.0).sum(axis=1)
        near_companions=(ds<=9).sum(axis=1)
        # Area penalty +5m environs. Real penalty box is 16.5m depth, half width20.16m.
        xdist=(d['x'] if end==0 else L-d['x'])
        in_box=d['valid']&(xdist>=0)&(xdist<=21.5)&(np.abs(d['y']-W/2)<=PENALTY_AREA_HALF+5)
        box=team_counts(d,in_box)
        count=box.sum(axis=1)
        growth=lagged(d,count,5)
        area_reorganize=(np.nan_to_num(growth,nan=0)>=2)&(count>=5)
        # No identity assumption: min-distance-to-corner change and dwell.
        approach=lagged(d,closest,2.)
        arrived=(closest<=4)&(np.nan_to_num(approach,nan=0)<-1)
        dwell=(closest<=4)&(rolling_fraction(closest<=4,max(2,int(FPS/d['step'])))>=.5)
        ball_corner=d['obs']&(np.hypot(d['bx']-cx,d['by']-cy)<=3.5)
        # Ball velocity away from corner: long towards box or short near touchline.
        away=(vx*(1 if end==0 else -1)+vy*(1 if lateral==0 else -1))
        prior_corner=np.r_[False,ball_corner[:-1]]
        kick=d['obs']&(ball_corner|prior_corner)&(away>=1.)
        # Goal-line exit from preceding play supported, not assumed from corner shape.
        toward_end=(vx<-.8 if end==0 else vx>.8)
        endzone=d['obs']&(np.abs(d['bx']-cx)<=3)&(np.abs(d['by']-cy)<=18)
        exit_anchor=endzone&toward_end
        features[f'{label}_corner_taker_count']=takers
        features[f'{label}_corner_min_player_dist_m']=np.where(np.isfinite(closest),closest,np.nan)
        features[f'{label}_area_players_home']=box[:,0]
        features[f'{label}_area_players_away']=box[:,1]
        features[f'{label}_area_growth_5s']=growth
        features[f'{label}_taker_arrival']=arrived.astype(int)
        features[f'{label}_taker_dwell']=dwell.astype(int)
        features[f'{label}_ball_near_corner_observed']=ball_corner.astype(int)
        features[f'{label}_ball_exit_toward_end']=exit_anchor.astype(int)
        features[f'{label}_ball_departure']=kick.astype(int)
        # Candidate anchor must include corner-specific taker and at least another phase.
        formation=dwell&((area_reorganize)|(rolling_fraction(box.sum(axis=1)>=6,max(2,int(2*FPS/d['step'])))>=.7))
        seeds=np.flatnonzero(formation|(arrived&area_reorganize))
        for k in seeds:
            fr=int(d['frames'][k]);period=str(d['period'][k]);
            if fr-last_used.get((end,lateral,period),-999999)<FPS*3:continue
            left,right=ranges_from_anchor(d,k,12,35)
            r=np.arange(left,right+1)
            r=r[d['period'][r]==d['period'][k]]
            pre=r[r<=k];post=r[r>=k]
            if len(post)==0:continue
            prior_exit=bool(np.any(exit_anchor[pre]));prior_arrival=bool(np.any(arrived[pre]));
            area_support=bool(np.any(area_reorganize[pre]) or (np.count_nonzero(count[pre]>=6)>=3))
            # kick can be after formation; avoid earlier corner observation reused as execution.
            kickpost=post[kick[post]]
            restart=int(kickpost[0]) if len(kickpost) else None
            # Collective activation after delivery. Strongest with actual ball, weaker without.
            new_area_change=np.nan_to_num(np.abs(lagged(d,count,1)),nan=0)
            response=post[(post>k)&(new_area_change[post]>=2)]
            inferred=int(response[0]) if len(response) else None
            provider=matched_provider(tags,fr,fr,5*FPS)
            specific=bool(prior_exit or prior_arrival or restart is not None or provider)
            if not specific or not area_support:continue
            # Actual confirmed geometry demands corner taker, spatial area, and temporal transition.
            tier=('SEQUENCE_CANDIDATE' if restart is not None and (prior_arrival or prior_exit)
                  else 'FORMATION_CANDIDATE' if prior_arrival and area_support
                  else 'PROVIDER_SUPPORTED' if provider else 'POSSIBLE_ONLY')
            evidence=set(['CORNER_TAKER_DWELL','PENALTY_AREA_STRUCTURE'])
            if prior_arrival:evidence.add('TAKER_APPROACH')
            if prior_exit:evidence.add('ENDLINE_OUTWARD_BALL')
            if restart is not None:evidence.add('CORNER_BALL_DEPARTURE_OBSERVED')
            if inferred is not None:evidence.add('AREA_COLLECTIVE_CHANGE_POST')
            if provider:evidence.add('PROVIDER_CONTEXT')
            begin_idx=pre[np.flatnonzero(exit_anchor[pre])[-1]] if prior_exit else (pre[np.flatnonzero(arrived[pre])[0]] if prior_arrival else pre[0])
            upper=int(d['frames'][max(0,restart)]) if restart is not None else (int(d['frames'][inferred]) if inferred is not None else None)
            # Direction of initial observed ball displacement supports short vs long, not endpoint of D.
            style='UNKNOWN'
            if restart is not None:
                next_idx=min(n-1,restart+max(1,int(FPS/d['step'])))
                moved=np.hypot(d['bx'][next_idx]-cx,d['by'][next_idx]-cy) if d['obs'][next_idx] else np.nan
                if np.isfinite(moved):style='SHORT' if moved<13 else 'LONG'
            candidates.append({'event_type':KIND,'period':period,'location_id':label,
                'anchor_frame':fr,'anchor_last_frame':fr,'anchor_count':1,'evidence':evidence,
                'corner_id':label,'restart_style':style,
                'd_start_low':int(d['frames'][max(0,begin_idx-2)]),'d_start_high':int(d['frames'][begin_idx]),
                'd_start_method':('ENDLINE_DIRECTION_INFERRED' if prior_exit else 'PLAYER_ARRIVAL_INFERRED'),
                'd_end_low':int(d['frames'][max(0,restart-1)]) if restart is not None else (int(d['frames'][max(0,inferred-3)]) if inferred is not None else None),
                'd_end_high':upper,'d_end_method':('OBSERVED_CORNER_BALL_MOTION' if restart is not None else 'COLLECTIVE_INFERRED' if inferred is not None else 'NOT_LOCALIZED'),
                'taker_dwell_snapshots':int(np.count_nonzero(dwell[pre])),
                'area_reorganization_snapshots':int(np.count_nonzero(area_reorganize[pre])),
                'tier':tier,'provider_events':len(provider)})
            last_used[(end,lateral,period)]=fr
    episodes=merge_candidates(candidates,d,merge_s=3)
    for row in episodes:row['evidence']=';'.join(sorted(row['evidence']))
    return episodes,features,{'rule':'TH02_EXIT_ARRIVAL_WAIT_KICK','snapshot_independence':True,
       'ball_missing_not_veto':True,'parameter_status':'EXPLORATORY_NOT_CALIBRATED'}

if __name__=='__main__':cli(detect)
