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


KIND='GOAL_KICK'

def detect(d,tags):
    """TH03: salida por fondo + retirada rival + defensor a área de meta + ejecución.
    Hipótesis tanto por portería como por posible equipo defensor (sin GK ID).
    """
    n=len(d['frames']);vx,vy=observed_ball_motion(d)
    candidates=[];features={};last_used={}
    cent,cov=team_centroids(d)
    for end in (0,1):
      label='LEFT' if end==0 else 'RIGHT';gx=0. if end==0 else L
      distance=(d['x'] if end==0 else L-d['x'])
      in_pa=d['valid']&(distance>=0)&(distance<=16.5)&(np.abs(d['y']-W/2)<=PENALTY_AREA_HALF)
      in_ga=d['valid']&(distance>=0)&(distance<=5.5)&(np.abs(d['y']-W/2)<=GOAL_AREA_HALF)
      counts=team_counts(d,in_pa);ga=team_counts(d,in_ga)
      goal_area_ball=d['obs']&((d['bx']<=5.5) if end==0 else (d['bx']>=L-5.5))&(np.abs(d['by']-W/2)<=GOAL_AREA_HALF)
      toward_end=(vx<-.8) if end==0 else (vx>.8)
      ball_end=d['obs']&(np.abs(d['bx']-gx)<=2.5)&(np.abs(d['by']-W/2)>=GOAL_HALF+1)&toward_end
      features[f'{label}_ball_endline_outward']=ball_end.astype(int)
      features[f'{label}_ball_goal_area_observed']=goal_area_ball.astype(int)
      for defending in (0,1):
        opp=1-defending;tag=f'{label}_{"BAR" if defending==0 else "ATH"}'
        # Opponent withdrawal from own PA: fewer attackers inside and their
        # centroid progressing away from goal over 2-8 s; never require halfway.
        opp_count=counts[:,opp].astype(float)
        opp_delta=lagged(d,opp_count,4.)
        opp_x=cent[:,opp,0]
        progress=lagged(d,opp_x,4)*(1 if end==0 else -1)
        # Distinct evidence of withdrawal, avoiding oclusion as exit.
        observed_ok=(cov[:,opp]>=6)&(lagged(d,cov[:,opp].astype(float),4)>=-1)
        withdraw=(observed_ok&((np.nan_to_num(opp_delta,nan=0)<=-1)&
                   (np.nan_to_num(progress,nan=0)>=2)))
        # Broader supporting retreat when none had been inside the box.
        retreat=observed_ok&(np.nan_to_num(progress,nan=0)>=3)
        # Own team defender in goal area: cannot infer goalkeeper identity.
        kicker=ga[:,defending]>=1
        stayed=rolling_fraction(kicker,max(2,int(2*FPS/d['step'])))>=.6
        spread=counts[:,defending].astype(float)
        structure=(spread>=2)&(cov[:,defending]>=6)
        # Difference of team lateral width (not stable player IDs)
        spread_y=np.full(n,np.nan)
        sl=slice(0,24) if defending==0 else slice(24,46)
        coords=d['y'][:,sl];valid=d['valid'][:,sl]
        for i in np.flatnonzero(cov[:,defending]>=6):
            yy=coords[i,valid[i]]
            spread_y[i]=np.std(yy) if len(yy)>=6 else np.nan
        width_change=lagged(d,spread_y,4.)
        opening=structure&((np.nan_to_num(width_change,nan=0)>=1.0)|
                           (np.nan_to_num(lagged(d,spread,3.),nan=0)>=1))
        # Execution: observed from own goal area, later observed displacement
        # longitudinally away from goal; should not label a normal pass by GK
        # absent prior interruption evidence.
        direction=(vx>1.5) if end==0 else (vx< -1.5)
        prior_ga=np.r_[False,goal_area_ball[:-1]]
        kick=d['obs']&(goal_area_ball|prior_ga)&direction
        ball_still=goal_area_ball&np.isfinite(vx)&np.isfinite(vy)&(np.hypot(vx,vy)<0.7)
        features[f'{tag}_opponents_pa_count']=opp_count
        features[f'{tag}_opponents_pa_delta_4s']=opp_delta
        features[f'{tag}_opponents_retreat_centroid_m_4s']=progress
        features[f'{tag}_opponents_withdrawal']=withdraw.astype(int)
        features[f'{tag}_defenders_goal_area_count']=ga[:,defending]
        features[f'{tag}_defenders_penalty_area_count']=spread
        features[f'{tag}_defender_width_change_4s_m']=width_change
        features[f'{tag}_defender_structure']=structure.astype(int)
        features[f'{tag}_ball_kick_observed']=kick.astype(int)
        setup=stayed&structure&((withdraw)|(retreat&opening))
        seeds=np.flatnonzero(setup)
        for k in seeds:
            fr=int(d['frames'][k]);period=str(d['period'][k])
            if fr-last_used.get((tag,period),-999999)<FPS*3:continue
            left,right=ranges_from_anchor(d,k,12,35)
            r=np.arange(left,right+1);r=r[d['period'][r]==d['period'][k]]
            pre=r[r<=k];post=r[r>=k]
            if len(pre)<2 or len(post)<2:continue
            retreat_support=bool(np.any(withdraw[pre]) or (np.count_nonzero(retreat[pre])>=2))
            shape_support=bool(np.count_nonzero(stayed[pre]&structure[pre])>=2)
            prior_exit=bool(np.any(ball_end[pre]))
            kick_post=post[kick[post]]
            restart=int(kick_post[0]) if len(kick_post) else None
            provider=matched_provider(tags,fr,fr,5*FPS)
            if not retreat_support or not shape_support:continue
            # Do not identify 2 passes from normal open play as goal kicks.
            has_ball_wait=bool(np.count_nonzero(ball_still[pre])>=2)
            has_transition=prior_exit or bool(provider) or (np.any(withdraw[pre]) and has_ball_wait and restart is not None)
            if not has_transition:continue
            tier='SEQUENCE_CANDIDATE' if restart is not None and (prior_exit or bool(provider)) else 'FORMATION_CANDIDATE'
            evidence=set(['OPPONENT_DIRECTIONAL_RETREAT','DEFENDER_GOAL_AREA_DWELL'])
            if prior_exit:evidence.add('ENDLINE_OUTWARD_BALL')
            if restart is not None:evidence.add('GOAL_AREA_BALL_DEPARTURE')
            if provider:evidence.add('PROVIDER_CONTEXT')
            if has_ball_wait:evidence.add('GOAL_AREA_BALL_STATIONARY')
            if np.any(opening[pre]):evidence.add('DEFENDER_BUILDOUT_OPENING')
            start=pre[np.flatnonzero(ball_end[pre])[-1]] if prior_exit else pre[np.flatnonzero(withdraw[pre])[0]] if np.any(withdraw[pre]) else pre[0]
            candidates.append({'event_type':KIND,'period':period,'location_id':tag,
                'anchor_frame':fr,'anchor_last_frame':fr,'anchor_count':1,'evidence':evidence,
                'goal_end_id':label,'defending_team_candidate':('Barcelona' if defending==0 else 'Athletic'),
                'd_start_low':int(d['frames'][max(0,start-2)]),'d_start_high':int(d['frames'][start]),
                'd_start_method':'ENDLINE_DIRECTION_INFERRED' if prior_exit else 'OPPONENT_RETREAT_INFERRED',
                'd_end_low':int(d['frames'][max(0,restart-1)]) if restart is not None else None,
                'd_end_high':int(d['frames'][restart]) if restart is not None else None,
                'd_end_method':'GOAL_AREA_BALL_MOVEMENT_OBSERVED' if restart is not None else 'NOT_LOCALIZED',
                'retreat_independent_snapshots':int(np.count_nonzero(withdraw[pre])),
                'defender_stay_snapshots':int(np.count_nonzero(stayed[pre])),
                'tier':tier,'provider_events':len(provider)})
            last_used[(tag,period)]=fr
    episodes=merge_candidates(candidates,d,merge_s=3)
    for row in episodes:row['evidence']=';'.join(sorted(row['evidence']))
    return episodes,features,{'rule':'TH03_ENDLINE_RETREAT_SETUP_RESTART',
       'goal_area_depth_m':5.5,'penalty_area_depth_m':16.5,
       'team_ids_are_not_player_ids':True,'parameter_status':'EXPLORATORY_NOT_CALIBRATED'}

if __name__=='__main__':cli(detect)
