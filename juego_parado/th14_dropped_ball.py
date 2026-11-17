"""TH14 — Balón a tierra, detector diagnóstico independiente.

Uso: python th14_dropped_ball.py --self-test
     python th14_dropped_ball.py --step 5 --out local_outputs/runs

Requiere config.py, starter.load_atd, matrix_v4.csv; numpy, pandas, scipy.
No modifica run.py ni asigna H/A/D. Geometría/umbrales exploratorios.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

@dataclass(frozen=True)
class Parameters:
    fps:int=25
    pitch_length_m:float=105.3
    pitch_width_m:float=68.0
    sample_interval_s:float=0.60
    min_visible_per_team:int=7
    penalty_depth_m:float=16.5
    penalty_width_m:float=40.32
    area_tolerance_m:float=0.6
    distance_rule_m:float=4.0  # IFAB; tolerance applies separately
    spacing_tolerance_m:float=0.8
    min_receiving_neighbors:int=0
    max_receiving_neighbors:int=1
    receiver_near_ball_m:float=2.6
    receiver_hypothesis_min_spacing_m:float=3.1
    receiver_stationarity_m:float=1.7
    keeper_goal_dist_m:float=19.0 # coarse proxy only, not keeper identification
    max_potential_anchors:int=8
    quiet_window_s:float=2.4
    quiet_min_fraction:float=0.60
    min_wait_s:float=1.8
    max_wait_s:float=45.0
    pre_window_s:float=7.0
    min_pre_activity_mps:float=1.6
    quiet_motion_mps:float=1.25
    slowdown_min_mps:float=0.35
    spatial_match_max_m:float=6.0
    min_sample_gap_s:float=0.25
    max_sample_gap_s:float=1.3
    ball_stationarity_m:float=1.5
    ball_absence_weight:float=0.0
    restart_movement_m:float=1.4
    restart_max_pair_gap_s:float=1.3
    continuation_min_s:float=1.2
    allow_ballless_candidates:bool=True
    max_episodes:int=500
    dedup_s:float=5.0
    # Display tiers, NOT calibrated posterior probabilities
    candidate_score:int=5
    strong_score:int=8
P=Parameters()


def numbers(df,name,default=np.nan):
    if name not in df:return np.full(len(df),default,dtype=float)
    return pd.to_numeric(df[name],errors='coerce').to_numpy(dtype=float)


def load_data(step:int):
    import config
    if int(config.FPS)!=P.fps:raise ValueError('FPS incompatible')
    n=int(config.N_VIDEO_FRAMES)
    matrix=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    if len(matrix)!=n or not np.array_equal(numbers(matrix,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv no coincide con frame 0..N-1')
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('ATD sin frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(numbers(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD no se alinea a matrix_v4')
    stride=max(step,int(np.ceil(P.sample_interval_s*P.fps/step))*step)
    frames=np.arange(0,n,stride,dtype=int)
    periods=np.array([config.get_period(int(f)) for f in frames])
    chosen=np.isin(periods,('FIRST_HALF','SECOND_HALF'))
    frames,periods=frames[chosen],periods[chosen]
    xc=[f'{i}_x' for i in range(1,47)];yc=[f'{i}_y' for i in range(1,47)]
    if any(k not in raw.columns for k in xc+yc):raise ValueError('Slots 1..46 faltantes')
    x=raw.loc[frames,xc].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_length_m
    y=raw.loc[frames,yc].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.pitch_width_m
    valid=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.pitch_length_m)&(y>=0)&(y<=P.pitch_width_m)
    x[~valid]=np.nan;y[~valid]=np.nan
    bx=numbers(matrix,'ball_x_normalized')[frames]*P.pitch_length_m
    by=numbers(matrix,'ball_y_normalized')[frames]*P.pitch_width_m
    obs=(numbers(matrix,'ball_observed',0)[frames]>.5)&(numbers(matrix,'ball_imputed',0)[frames]<.5)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=frames,periods=periods,x=x,y=y,bx=bx,by=by,obs=obs)


def team_pts(d,i,t):
    s=slice(0,24) if t==0 else slice(24,46)
    out=np.column_stack((d['x'][i,s],d['y'][i,s]))
    return out[np.isfinite(out).all(axis=1)]

def all_pts(d,i):
    return [team_pts(d,i,t) for t in (0,1)]

def in_penalty(p):
    x,y=p
    deep=(x<=P.penalty_depth_m+P.area_tolerance_m or x>=P.pitch_length_m-P.penalty_depth_m-P.area_tolerance_m)
    return bool(deep and abs(y-P.pitch_width_m/2)<=P.penalty_width_m/2+P.area_tolerance_m)

def goalkeeper_side(p):
    if not in_penalty(p):return None
    return 'LEFT' if p[0]<P.pitch_length_m/2 else 'RIGHT'

def spatial_movement(a,b,dt):
    """Robust approximate displacement from 1:1 spatial matching; never player identity."""
    if len(a)<P.min_visible_per_team or len(b)<P.min_visible_per_team or dt<=0:return np.nan
    dist=cdist(a,b)
    row,col=linear_sum_assignment(dist)
    matched=dist[row,col]
    matched=matched[matched<=P.spatial_match_max_m]
    if len(matched)<min(len(a),len(b))*.55:return np.nan
    return float(np.median(matched)/dt)

def motion(d,i):
    if i<1 or d['periods'][i]!=d['periods'][i-1]:return np.nan
    dt=(d['frames'][i]-d['frames'][i-1])/P.fps
    if not P.min_sample_gap_s<=dt<=P.max_sample_gap_s:return np.nan
    vals=[spatial_movement(team_pts(d,i-1,t),team_pts(d,i,t),dt) for t in (0,1)]
    return float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan

def evaluate_anchor(d,i,p,team,ball_anchor):
    A,B=all_pts(d,i)
    if min(len(A),len(B))<P.min_visible_per_team:return None
    members=[A,B];allp=np.vstack(members)
    dist=cdist(allp,np.asarray(p).reshape(1,2)).ravel()
    nearest=int(np.argmin(dist));receiver_dist=float(dist[nearest]);receiver_team=0 if nearest<len(A) else 1
    # True separation: remove only receiver, check other 21 players, no slot identity.
    other=np.delete(dist,nearest)
    nearest_other=float(np.min(other)) if len(other) else np.inf
    near_other=int(np.count_nonzero(other<P.distance_rule_m-P.spacing_tolerance_m))
    near_other_4=int(np.count_nonzero(other<P.distance_rule_m))
    if receiver_dist>P.receiver_near_ball_m:return None
    if near_other>P.max_receiving_neighbors:return None
    # Observed-ball anchor stronger; without ball, enforce genuine player isolation.
    if not ball_anchor and nearest_other<P.receiver_hypothesis_min_spacing_m:return None
    area=in_penalty(p)
    if area:
        # Without reliable team orientation or persistent goalkeeper ID do not assert keeper.
        # Isolated receiver in penalty is plausible, but goalkeeper-specific claim unverified.
        pass
    return dict(anchor=[float(p[0]),float(p[1])],team=receiver_team,
                area='PENALTY_AREA' if area else 'OPEN_FIELD',goal_side=goalkeeper_side(p),
                observed_ball=bool(ball_anchor),receiver_distance_m=round(receiver_dist,2),
                nearest_other_m=round(nearest_other,2),players_inside_3_2m=near_other,
                players_inside_4m=near_other_4)

def snapshots(d,i):
    A,B=all_pts(d,i)
    if min(len(A),len(B))<P.min_visible_per_team:return []
    if d['obs'][i]:
        a=evaluate_anchor(d,i,(d['bx'][i],d['by'][i]),None,True)
        return [a] if a else []
    if not P.allow_ballless_candidates:return []
    allp=np.vstack([A,B]);dd=cdist(allp,allp)
    np.fill_diagonal(dd,np.inf)
    isolated=np.argsort(np.min(dd,axis=1))[::-1][:P.max_potential_anchors]
    return [a for k in isolated if (a:=evaluate_anchor(d,i,allp[k],None,False))]

def consistent(a,b):
    return a['team']==b['team'] and np.linalg.norm(np.array(a['anchor'])-np.array(b['anchor'])) <= (P.receiver_stationarity_m if not (a['observed_ball'] and b['observed_ball']) else P.ball_stationarity_m)

def ball_motion_end(d,start_idx,stop_idx,anchor):
    """Only infer post-release movement; cannot prove ground contact with 2D tracking."""
    last=None
    for j in range(start_idx,stop_idx+1):
        if not d['obs'][j]:continue
        pp=np.array([d['bx'][j],d['by'][j]])
        if last is not None:
            k,old=last
            dt=(d['frames'][j]-d['frames'][k])/P.fps
            if dt<=P.restart_max_pair_gap_s:
                if np.linalg.norm(pp-old)>=P.restart_movement_m and np.linalg.norm(old-anchor)<=P.receiver_near_ball_m+1.5:
                    return [int(d['frames'][k]),int(d['frames'][j])]
        last=(j,pp)
    return None

def detect(d):
    n=len(d['frames']);ifn=lambda s: int(round(s*P.fps/max(1,np.median(np.diff(d['frames']))))) if n>1 else 0
    window=max(2,ifn(P.quiet_window_s));min_len=ifn(P.min_wait_s)
    max_len=max(window,ifn(P.max_wait_s));prior=ifn(P.pre_window_s)
    mov=np.array([motion(d,i) for i in range(n)])
    cand=[snapshots(d,i) for i in range(n)]
    episodes=[];blocked=-1;i=0
    while i<n:
        if i<=blocked or not cand[i]:i+=1;continue
        for seed in cand[i]:
            run=[(i,seed)];last=i
            for j in range(i+1,min(n,i+max_len)):
                if d['periods'][j]!=d['periods'][i]:break
                candidates=[a for a in cand[j] if consistent(seed,a)]
                if candidates:
                    winner=min(candidates,key=lambda a:np.linalg.norm(np.array(a['anchor'])-np.array(seed['anchor'])))
                    run.append((j,winner));last=j
                elif j-last>1:break
            duration=(d['frames'][last]-d['frames'][i])/P.fps
            if duration<P.min_wait_s or len(run)<3:continue
            # Independence: persistence across distinct sampled observations, not copied frames.
            prior_m=mov[max(1,i-prior):i]
            run_m=mov[i:last+1]
            before=float(np.nanmedian(prior_m)) if np.any(np.isfinite(prior_m)) else np.nan
            during=float(np.nanmedian(run_m)) if np.any(np.isfinite(run_m)) else np.nan
            quiet_frac=float(np.mean(run_m[np.isfinite(run_m)]<=P.quiet_motion_mps)) if np.any(np.isfinite(run_m)) else 0.
            slowdown=bool(np.isfinite(before) and np.isfinite(during) and before-during>=P.slowdown_min_mps and before>=P.min_pre_activity_mps)
            # Never regard a static isolated player as sufficient stoppage evidence.
            if not slowdown or quiet_frac<P.quiet_min_fraction:continue
            observed=sum(x['observed_ball'] for _,x in run)
            best=run[len(run)//2][1]
            strong_spacing=sum(x['players_inside_3_2m']==0 for _,x in run)/len(run)
            if strong_spacing<.75:continue
            evidence=['isolated_receiver','four_metre_exclusion','stable_position','activity_to_wait_transition']
            score=4
            if observed>=2:score+=2;evidence.append('ball_observed_at_anchor')
            if best['area']=='PENALTY_AREA':score+=1;evidence.append('penalty_area_goalkeeper_rule_context_not_identity')
            if duration>=3:score+=1;evidence.append('persistent_wait')
            # Search until next 10 seconds for ball release; D end is earlier ground contact, not measured.
            stop=min(n-1,last+max(2,ifn(10)))
            end=ball_motion_end(d,last,stop,np.array(best['anchor']))
            if end:score+=1;evidence.append('post_release_ball_motion')
            if score<P.candidate_score:continue
            # Estimate onset as uncertainty interval between preceding active motion and first preparation.
            pre_idxs=np.flatnonzero(np.isfinite(mov[max(1,i-prior):i])&(mov[max(1,i-prior):i]>=P.min_pre_activity_mps))
            base=max(1,i-prior)
            onset_lb=int(d['frames'][base+pre_idxs[-1]]) if len(pre_idxs) else None
            onset_ub=int(d['frames'][i])
            episodes.append(dict(type='DROPPED_BALL_COMPATIBLE',confidence_tier='STRONG_PATTERN' if score>=P.strong_score else 'CANDIDATE',score=int(score),
                period=str(d['periods'][i]),first_preparation_frame=int(d['frames'][i]),last_preparation_frame=int(d['frames'][last]),
                d_start_interval=[onset_lb,onset_ub],d_end_interval=None,
                downstream_ball_motion_interval=end,ground_contact_confirmed=False,
                ball_observation_count=int(observed),receiver_team_slot_group=int(best['team']),
                target_location_m=best['anchor'],target_area=best['area'],goal_side=best['goal_side'],
                observed_duration_s=round(duration,2),motion_before_mps=round(before,2),motion_wait_mps=round(during,2),
                quiet_fraction=round(quiet_frac,2),exclusion_fraction=round(strong_spacing,2),evidence=evidence,
                caveat='No árbitro, z ni identidades persistentes: no prueba caída reglamentaria ni portero. D termina al tocar suelo, no al patear.'))
            blocked=last+max(1,ifn(P.dedup_s));break
        i=max(i+1,blocked+1)
        if len(episodes)>=P.max_episodes:break
    return episodes


def synthetic():
    fps=P.fps;step=15;n=42;f=np.arange(n)*step
    x=np.full((n,46),np.nan);y=np.full_like(x,np.nan)
    rng=np.random.default_rng(114)
    # 11 players each, one isolated recipient at 52,34; others >4m.
    aa=np.array([[52.,34.]]+[[12.+j*3.,8.+(j%3)*3.] for j in range(10)])
    bb=np.array([[72.+(j%3)*3.,46.+(j//3)*3.] for j in range(11)])
    for t in range(n):
        jitter=(.65 if t<8 else .02)
        a=aa+rng.normal(0,jitter,aa.shape);b=bb+rng.normal(0,jitter,bb.shape)
        x[t,:11]=a[:,0];y[t,:11]=a[:,1]
        x[t,24:35]=b[:,0];y[t,24:35]=b[:,1]
    bx=np.full(n,np.nan);by=np.full(n,np.nan);obs=np.zeros(n,bool)
    bx[8:30]=52;by[8:30]=34;obs[8:30]=True
    bx[30:33]=[52.0,54.0,57.0];by[30:33]=34;obs[30:33]=True
    return dict(frames=f,periods=np.array(['FIRST_HALF']*n),x=x,y=y,bx=bx,by=by,obs=obs)

def self_test():
    d=synthetic()
    assert in_penalty((10,34)) and not in_penalty((52,34))
    p=evaluate_anchor(d,12,(52,34),None,True)
    assert p is not None and p['players_inside_3_2m']==0
    e=detect(d)
    assert len(e)>=1, 'Debería reconocer preparación aislada temporal'
    assert e[0]['d_end_interval'] is None and not e[0]['ground_contact_confirmed']
    no={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in d.items()};no['obs'][:]=False
    assert any(a for a in snapshots(no,12)),'Sin balón debe conservar hipótesis espaciales'
    # No active-to-wait transition must reject.
    still=synthetic();still['x'][:8]=still['x'][8];still['y'][:8]=still['y'][8]
    assert len(detect(still))==0,'Quietud sin cambio no permite declarar interrupción'
    print('TH14 self-test OK')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',default='local_outputs/runs')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    data=load_data(args.step)
    episodes=detect(data)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dest=out/'th14_dropped_ball_diagnostic.json'
    dest.write_text(json.dumps(dict(detector='TH14_DROPPED_BALL',version='1.0',parameters=asdict(P),sampled_frames=len(data['frames']),
                    episodes=episodes,not_ground_truth=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'TH14: {len(episodes)} candidatos -> {dest}')

if __name__=='__main__':main()
