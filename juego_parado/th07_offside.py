"""TH07 OFFSIDE: detector diagnostico geometricotemporal (no modifica H/A/D).

Instalar junto a config.py y starter/. Requiere numpy, pandas y archivos originales.
Uso: python th07_offside.py --step 5 --out local_outputs/runs
Salida: un JSON con configuracion, hipotesis, evidencias y limites inciertos.

LIMITACION: posiciones son centros de jugadores (sin extremidades), slots ATD no son IDs.
No puede determinar silbato, participacion por bloqueo visual o toque deliberado rival
con certeza. 'HIGH' significa evidencia tracking concordante, no verdad arbitral.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import config

@dataclass(frozen=True)
class Parameters:
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0
    min_team_visible: int = 8
    ball_min_pass_m: float = 2.0
    ball_min_speed_mps: float = 3.0
    ball_max_speed_mps: float = 42.0
    ball_pass_gap_s: float = 0.8
    passer_near_m: float = 3.5
    position_margin_m: float = 1.5
    strong_position_margin_m: float = 3.0
    receiver_near_ball_m: float = 4.0
    receiver_near_origin_m: float = 16.0
    receiver_search_s: float = 4.0
    min_forward_pass_m: float = 1.0
    defensive_direction_margin_m: float = 7.0
    activity_lag_s: float = 1.0
    inactivity_mps: float = 1.0
    inactivity_ratio: float = 0.48
    inactivity_window_s: float = 3.0
    stoppage_search_s: float = 10.0
    restart_search_s: float = 35.0
    stable_ball_radius_m: float = 1.5
    stable_ball_duration_s: float = 0.8
    restart_kick_m: float = 1.0
    restart_kick_speed_mps: float = 2.0
    restart_team_near_m: float = 5.0
    dedup_s: float = 2.0
    provider_tolerance_s: float = 10.0
P = Parameters()
FPS = int(config.FPS)


def col(df,name,fill=np.nan):
    return pd.to_numeric(df[name],errors='coerce').to_numpy(float) if name in df else np.full(len(df),fill)

def valid_period(f):
    return config.get_period(int(f)) in ('FIRST_HALF','SECOND_HALF')

def load_tracking(step):
    if step<1: raise ValueError('step>=1')
    mat=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    n=len(mat)
    if n!=config.N_VIDEO_FRAMES or not np.array_equal(col(mat,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv no alineada con frames de video 0..N-1')
    frames=np.arange(0,n,step,dtype=int)
    frames=frames[np.array([valid_period(f) for f in frames])]
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id': raw=raw.reset_index()
        else: raise ValueError('ATD no contiene frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(col(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD no alinea 1:1 con video')
    xx=[f'{i}_x' for i in range(1,47)]
    yy=[f'{i}_y' for i in range(1,47)]
    if any(c not in raw for c in xx+yy): raise ValueError('Faltan slots ATD 1..46')
    x=raw.loc[frames,xx].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_length_m
    y=raw.loc[frames,yy].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_width_m
    ok=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.pitch_length_m)&(y>=0)&(y<=P.pitch_width_m)
    x[~ok]=np.nan;y[~ok]=np.nan
    bx=col(mat,'ball_x_normalized')[frames]*P.pitch_length_m
    by=col(mat,'ball_y_normalized')[frames]*P.pitch_width_m
    obs=(col(mat,'ball_observed',0)[frames]>.5)&(col(mat,'ball_imputed',0)[frames]<.5)&np.isfinite(bx)&np.isfinite(by)
    periods=np.array([config.get_period(int(f)) for f in frames])
    return dict(frames=frames,x=x,y=y,ok=ok,bx=bx,by=by,obs=obs,periods=periods)


def provider_events():
    try:
        from starter.load_smart_tagging import load
        raw=load(game=config.GAME)
    except (ImportError, OSError, FileNotFoundError, TypeError, KeyError, ValueError):
        return []
    events=[]
    for e in raw:
        if not isinstance(e,dict): continue
        tags=e.get('tags') or {}
        label=' '.join(str(v) for v in [e.get('code',''),tags.get('Type',''),e.get('type','')]).lower()
        if not ('offside' in label or 'off-side' in label or 'fuera de juego' in label): continue
        try:
            a=int(e['start_frame']);b=int(e['end_frame'])
        except (KeyError,TypeError,ValueError): continue
        events.append(dict(start_frame=a,end_exclusive=b,kind='PROVIDER_OFFSIDE_NOT_GT'))
    return events


def team_xy(d,i,team):
    sl=slice(0,24) if team==0 else slice(24,46)
    q=np.column_stack((d['x'][i,sl],d['y'][i,sl]))
    return q[np.isfinite(q).all(axis=1)]


def closest(a,p):
    return float(np.min(np.linalg.norm(a-p,axis=1))) if len(a) else float('inf')


def player_speed_proxy(d,lag_seconds=1):
    """Median nearest-neighbour velocity per team; NOT player identity."""
    n=len(d['frames']);out=np.full((n,2),np.nan)
    for i in range(n):
        frame=d['frames'][i];j=np.searchsorted(d['frames'],frame-int(lag_seconds*FPS))
        if j>=i or d['periods'][j]!=d['periods'][i]: continue
        dt=(frame-d['frames'][j])/FPS
        for t in (0,1):
            a=team_xy(d,i,t);b=team_xy(d,j,t)
            if min(len(a),len(b))<P.min_team_visible or abs(len(a)-len(b))>2: continue
            dist=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
            m=np.min(dist,axis=1)
            use=m<6.0
            if use.sum()>=P.min_team_visible:out[i,t]=float(np.median(m[use])/dt)
    return out


def infer_attacking_direction(d,i,team):
    """Estimate sign of direction from defensive depth: attacking team typically ahead.
    Returns ±1 only if team ordering reasonably clear; else both possible downstream.
    """
    ours=team_xy(d,i,team);theirs=team_xy(d,i,1-team)
    if min(len(ours),len(theirs))<P.min_team_visible:return []
    difference=float(np.median(ours[:,0])-np.median(theirs[:,0]))
    if abs(difference)<P.defensive_direction_margin_m:return []
    return [1 if difference<0 else -1]


def position_candidates(d,i,attacker,sign):
    atk=team_xy(d,i,attacker);defenders=team_xy(d,i,1-attacker)
    if min(len(atk),len(defenders))<P.min_team_visible or not d['obs'][i]:return []
    # 2nd closest defender to attacked goal, regardless of goalkeeper identity
    line=float(np.sort(sign*defenders[:,0])[-2]);ball=float(sign*d['bx'][i]);mid=P.pitch_length_m/2
    threshold=max(line,ball)
    candidates=[]
    for px,py in atk:
        depth=sign*float(px)-threshold
        if sign*(float(px)-mid)>P.position_margin_m and depth>P.position_margin_m:
            candidates.append(dict(x=float(px),y=float(py),margin_m=round(depth,2)))
    return candidates


def observed_pairs(d):
    idx=np.flatnonzero(d['obs'])
    for i,j in zip(idx[:-1],idx[1:]):
        if d['periods'][i]!=d['periods'][j]:continue
        dt=(d['frames'][j]-d['frames'][i])/FPS
        if 0<dt<=P.ball_pass_gap_s:
            vec=np.array([d['bx'][j]-d['bx'][i],d['by'][j]-d['by'][i]])
            dist=float(np.linalg.norm(vec));speed=dist/dt
            yield i,j,vec,dist,speed


def intervention(d,i,j,attacker,sign,players):
    """Look for observed ball arriving near original advanced-player region; cannot
    assert which player touched or interfered without IDs/video. Requires space-time.
    """
    candidates=[];last=min(len(d['frames']),np.searchsorted(d['frames'],d['frames'][i]+round(P.receiver_search_s*FPS),'right'))
    for k in range(j,last):
        if d['periods'][k]!=d['periods'][i] or not d['obs'][k]:continue
        ball=np.array([d['bx'][k],d['by'][k]])
        for p in players:
            start=np.array([p['x'],p['y']])
            # Restrict ball path association to vicinity of advanced player at kick
            if np.linalg.norm(ball-start)>P.receiver_near_origin_m:continue
            team=team_xy(d,k,attacker)
            local=team[np.linalg.norm(team-start,axis=1)<=P.receiver_near_origin_m]
            if not len(local):continue
            near=closest(local,ball)
            if near<=P.receiver_near_ball_m and sign*(ball[0]-d['bx'][i])>=P.min_forward_pass_m:
                candidates.append(dict(frame=int(d['frames'][k]),near_ball_m=round(near,2),
                                       position_margin_m=p['margin_m'],
                                       original_position_m=[round(p['x'],2),round(p['y'],2)]))
    if not candidates:return None
    return min(candidates,key=lambda c:(c['frame'],-c['position_margin_m']))


def quiet_intervals(d,motion):
    """Find low-motion periods using unique snapshots and both teams' median NN movement."""
    n=len(d['frames']);min_points=max(2,int(round(P.inactivity_window_s*FPS/(d['frames'][1]-d['frames'][0] if n>1 else 5))))
    slow=np.all(np.isfinite(motion),axis=1)&(np.mean(motion<P.inactivity_mps,axis=1)>=P.inactivity_ratio)
    # A smooth window is weaker than referee stoppage; mark as evidence only.
    rolling=pd.Series(slow.astype(float)).rolling(min_points,min_periods=min_points).mean().to_numpy()
    return rolling>=0.7


def stoppage_signature(d,motion,quiet,k):
    """Stoppage hypothesis needs transition OR ball stasis, not mere low motion.
    Ball stasis is an observed position repeated >=0.8 s; imputation never counts.
    """
    n=len(d['frames'])
    hi=min(n,np.searchsorted(d['frames'],d['frames'][k]+round(P.stoppage_search_s*FPS),'right'))
    for q in range(k,hi):
        if not quiet[q] or d['periods'][q]!=d['periods'][k]:continue
        # A quiet field that has been quiet throughout is NOT a stoppage onset.
        prior=max(0,np.searchsorted(d['frames'],d['frames'][q]-round(4*FPS)))
        previous=motion[prior:q]
        was_active=bool(np.any(np.isfinite(previous)&(previous>2.2)))
        # Independent observed ball snapshots around q: stationary in same spatial sector.
        start=np.searchsorted(d['frames'],d['frames'][q]-round(1.8*FPS))
        ids=[t for t in range(start,q+1) if d['obs'][t] and d['periods'][t]==d['periods'][q]]
        still=False
        if len(ids)>=2 and d['frames'][ids[-1]]-d['frames'][ids[0]]>=P.stable_ball_duration_s*FPS:
            coords=np.column_stack((d['bx'][ids],d['by'][ids]))
            still=bool(np.max(np.linalg.norm(coords-coords[-1],axis=1))<P.stable_ball_radius_m)
        if was_active or still:return int(d['frames'][q]), ('MOTION_TRANSITION' if was_active else 'OBSERVED_BALL_STASIS')
    return None,None


def restart_evidence(d,start_i,attacker):
    """Look for defender-controlled stationary ball then observed kick after intervention.
    Does not infer a *legal indirect* kick, referee signal, or true touch identity.
    """
    end=min(len(d['frames']),np.searchsorted(d['frames'],d['frames'][start_i]+round(P.restart_search_s*FPS),'right'))
    observed=np.flatnonzero(d['obs'][start_i:end])+start_i
    best=None
    for prev,cur in zip(observed[:-1],observed[1:]):
        if d['periods'][cur]!=d['periods'][start_i]:break
        dt=(d['frames'][cur]-d['frames'][prev])/FPS
        if not(0<dt<=P.ball_pass_gap_s):continue
        ball=np.array([d['bx'][prev],d['by'][prev]])
        move=float(np.hypot(d['bx'][cur]-ball[0],d['by'][cur]-ball[1]))
        if move<P.restart_kick_m or move/dt<P.restart_kick_speed_mps:continue
        defending=team_xy(d,prev,1-attacker)
        if closest(defending,ball)>P.restart_team_near_m:continue
        # Seek multiple observed stationary ball snapshots BEFORE the kick
        stable=[]
        lookback=d['frames'][prev]-round(3*FPS)
        for t in range(np.searchsorted(d['frames'],lookback),prev+1):
            if d['obs'][t] and d['periods'][t]==d['periods'][prev] and np.hypot(d['bx'][t]-ball[0],d['by'][t]-ball[1])<P.stable_ball_radius_m:
                stable.append(t)
        if len(stable)<2 or (d['frames'][prev]-d['frames'][stable[0]])<P.stable_ball_duration_s*FPS:continue
        best=dict(stationary_since=int(d['frames'][stable[0]]),
                  kick_before=int(d['frames'][prev]),kick_after=int(d['frames'][cur]),
                  restart_location_m=[round(float(ball[0]),2),round(float(ball[1]),2)],
                  defender_near_ball_m=round(closest(defending,ball),2))
        break
    return best


def detect(d,providers=()):
    motion=player_speed_proxy(d,P.activity_lag_s)
    quiet=quiet_intervals(d,motion)
    raw=[]
    for i,j,vec,dist,speed in observed_pairs(d):
        if not(P.ball_min_pass_m<=dist and P.ball_min_speed_mps<=speed<=P.ball_max_speed_mps):continue
        ball=np.array([d['bx'][i],d['by'][i]])
        for attacker in (0,1):
            if closest(team_xy(d,i,attacker),ball)>P.passer_near_m:continue
            if closest(team_xy(d,i,1-attacker),ball)+0.7<closest(team_xy(d,i,attacker),ball):continue
            directions=infer_attacking_direction(d,i,attacker)
            # Ambiguous attacking direction cannot be asserted: skip, not pretend.
            for sign in directions:
                if sign*vec[0]<P.min_forward_pass_m:continue
                players=position_candidates(d,i,attacker,sign)
                if not players:continue
                involvement=intervention(d,i,j,attacker,sign,players)
                if involvement is None:continue
                k=np.searchsorted(d['frames'],involvement['frame'])
                stop_frame,stop_basis=stoppage_signature(d,motion,quiet,k)
                restart=restart_evidence(d,k,attacker) if stop_frame is not None else None
                nearprov=[r for r in providers if r['start_frame']<=involvement['frame']+P.provider_tolerance_s*FPS and r['end_exclusive']>=int(d['frames'][i])-P.provider_tolerance_s*FPS]
                ev=dict(type='OFFSIDE_CANDIDATE',stoppage_basis=stop_basis,pass_frame_before=int(d['frames'][i]),
                        pass_frame_after=int(d['frames'][j]),attacking_team_block=int(attacker),
                        attacking_direction='RIGHT' if sign==1 else 'LEFT',
                        ball_pass_m=round(dist,2),ball_pass_speed_mps=round(speed,2),
                        advanced_player_count=len(players),max_position_margin_m=round(max(p['margin_m'] for p in players),2),
                        involvement_proxy=involvement,observed_collective_quiet_start=stop_frame,
                        restart_proxy=restart,provider_offside=bool(nearprov),
                        evidence=['OBSERVED_BALL_PASS','SECOND_LAST_OPPONENT_GEOMETRY','ADVANCED_PLAYER_SPATIOTEMPORAL_PROXIMITY']+
                                 (['LOW_COLLECTIVE_MOTION'] if stop_frame is not None else [])+
                                 (['DEFENDER_RESTART_PROXY'] if restart else [])+
                                 (['PROVIDER_OFFSIDE_TAG_NOT_GT'] if nearprov else []),
                        contradictions=[],limits={},tier='GEOMETRIC_RISK')
                if stop_frame is None:
                    ev['contradictions'].append('NO_OBSERVED_STOPPAGE')
                    ev['limits']=dict(d_start_low=None,d_start_high=None,d_end_low=None,d_end_high=None)
                else:
                    # stop formation is retrospective evidence of prior whistle; no exact instant.
                    ev['limits']['d_start_low']=int(involvement['frame'])
                    ev['limits']['d_start_high']=stop_frame
                    if restart:
                        ev['limits']['d_end_low']=restart['kick_before'];ev['limits']['d_end_high']=restart['kick_after']
                    else:
                        ev['limits']['d_end_low']=None;ev['limits']['d_end_high']=None
                        ev['contradictions'].append('RESTART_NOT_OBSERVED')
                if stop_frame is not None and restart:ev['tier']='RESTART_COMPATIBLE'
                elif stop_frame is not None:ev['tier']='STOPPAGE_COMPATIBLE'
                if nearprov:ev['tier']='PROVIDER_CORROBORATED_'+ev['tier']
                raw.append(ev)
    # Group candidate passes for one possible phase; preserve strongest + chronological
    raw.sort(key=lambda r:(r['attacking_team_block'],r['pass_frame_before']))
    dedup=[]
    score={'GEOMETRIC_RISK':0,'STOPPAGE_COMPATIBLE':1,'RESTART_COMPATIBLE':2}
    for ev in raw:
        if dedup and ev['attacking_team_block']==dedup[-1]['attacking_team_block'] and ev['pass_frame_before']-dedup[-1]['pass_frame_before']<=P.dedup_s*FPS and d['periods'][np.searchsorted(d['frames'],ev['pass_frame_before'])]==d['periods'][np.searchsorted(d['frames'],dedup[-1]['pass_frame_before'])]:
            prev=dedup[-1]
            val=lambda x:score.get(x['tier'].replace('PROVIDER_CORROBORATED_',''),0)+2*int(x['provider_offside'])
            if val(ev)>val(prev):dedup[-1]=ev
        else: dedup.append(ev)
    counts={key:sum(r['tier'].endswith(key) for r in dedup) for key in score}
    return dedup,counts


def run(step=5,out=None,synthetic=False):
    d=synthetic if isinstance(synthetic,dict) else load_tracking(step)
    prov=[] if synthetic else provider_events()
    episodes,counts=detect(d,prov)
    report=dict(detector='TH07_OFFSIDE',status='DIAGNOSTIC_ONLY_NOT_GT',fps=FPS,
                pitch_config_m=[P.pitch_length_m,P.pitch_width_m],
                limitations=['No referee/assistant signal','ATD slots lack persistent IDs',
                             'body part and defender limb positions unavailable',
                             'intentional opponent play vs rebound unknowable',
                             'short movements and absent ball often unobservable',
                             'geometry alone never confirms offside offence'],
                parameters=asdict(P),sample_step=step,provider_offside_tags=len(prov),counts=counts,
                episodes=episodes)
    if out is not None:
        folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
        path=folder/'th07_offside_diagnostics.json'
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        print(f'{path} | candidates={len(episodes)} | by_tier={counts}')
    return report


def selftest():
    # 1) Geometry: second-last defender, own-half and ball line.
    f=np.arange(0,101,5);n=len(f)
    x=np.full((n,46),np.nan,np.float32);y=np.full((n,46),np.nan,np.float32)
    for i in range(n):
        x[i,:11]=np.linspace(40,80,11);y[i,:11]=np.linspace(7,60,11)
        x[i,24:35]=np.array([20,27,30,33,38,42,44,47,53,64,75]);y[i,24:35]=np.linspace(7,60,11)
    x[:,10]=90;y[:,10]=34
    d=dict(frames=f,x=x,y=y,obs=np.ones(n,bool),bx=np.full(n,70.),by=np.full(n,34.),
           periods=np.array(['FIRST_HALF']*n))
    assert len(position_candidates(d,0,0,1))>=1
    d['bx'][:]=92
    assert len(position_candidates(d,0,0,1))==0
    d['bx'][:]=70
    # Ball and passer can be near and defenders behind; no actual ball pass -> no events.
    assert run(step=5,out=None,synthetic=d)['episodes']==[]
    # Synthetic advanced runner, pass trajectory, later observed quiet ball:
    # geometric and temporal chain must be found; not a certified infraction.
    f=np.arange(55)*5;n=len(f)
    sx=np.full((n,46),np.nan,np.float32);sy=sx.copy()
    for h in range(n):
        sx[h,:11]=[30,35,38,40,43,44,45,46,47,50,90]
        sy[h,:11]=[2,8,12,16,20,25,34,40,48,60,34]
        sx[h,24:35]=[53,56,58,60,63,65,67,68,70,75,85]
        sy[h,24:35]=[2,8,12,16,20,25,30,40,48,55,60]
    bx=np.full(n,91.,float);bx[:10]=45
    for h in range(10,20):bx[h]=45+(h-9)*4.5
    z=dict(frames=f,x=sx,y=sy,obs=np.ones(n,bool),bx=bx,
           by=np.full(n,34.),periods=np.array(['FIRST_HALF']*n))
    reports,_=detect(z,[])
    assert reports and reports[0]['tier']=='STOPPAGE_COMPATIBLE'
    assert reports[0]['limits']['d_end_low'] is None
    assert reports[0]['limits']['d_start_high']>reports[0]['limits']['d_start_low']
    print('TH07 self-test OK: lines, geometry-only rejection, pass/involvement/stoppage chain')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=int,default=5)
    parser.add_argument('--out',default=str(config.RUNS_DIR))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:selftest()
    else:run(args.step,args.out)

if __name__=='__main__':main()
