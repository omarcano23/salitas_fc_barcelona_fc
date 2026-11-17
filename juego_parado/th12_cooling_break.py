"""TH12: desplazamiento colectivo hacia descanso (diagnóstico, NO adjudicación de D).

Uso (junto a config.py, starter/, local_outputs/matrix_v4.csv):
  python th12_cooling_break.py --self-test
  python th12_cooling_break.py --step 5 --out local_outputs/runs
Requiere numpy, pandas. Un archivo: sin imports desde otros TH.
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
    length_m: float = 105.3  # escala del proyecto: validar
    width_m: float = 68.0
    min_team: int = 7
    max_gap_s: float = .9
    evaluation_interval_s: float = 1.0  # observaciones independientes para evidencia
    migration_min_s: float = 5.0
    migration_max_s: float = 30.0
    wait_min_s: float = 8.0
    wait_max_s: float = 200.0
    return_max_s: float = 40.0
    min_net_shift_m: float = 8.0
    min_team_fraction: float = .50
    min_players_moving: int = 5
    preferred_side_band_m: float = 9.0
    target_near_band_m: float = 14.0
    min_band_occupancy_gain: float = .22
    min_direction_fraction: float = .65
    min_cluster_shrink_ratio: float = .78
    max_wait_motion_mps: float = 1.3
    min_wait_quiet_fraction: float = .62
    min_active_motion_mps: float = 1.55
    min_return_spread_gain: float = 2.5
    min_return_band_drop: float = .15
    ball_restart_min_dist_m: float = 1.3
    ball_restart_min_speed_mps: float = 2.4
    max_ball_obs_gap_s: float = 1.1
    episode_exclusion_s: float = 15.0
    max_candidates: int = 250

P = Parameters()


def arr(df, col, default=np.nan):
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)


def load_data(step):
    import config
    if step < 1: raise ValueError('--step debe ser >= 1')
    if int(config.FPS) != P.fps: raise ValueError('FPS de config distinto de TH12')
    n = int(config.N_VIDEO_FRAMES)
    mat = pd.read_csv(Path(config.OUTPUT_DIR) / 'matrix_v4.csv', low_memory=False)
    if len(mat) != n or not np.array_equal(arr(mat, 'frame'), np.arange(n)):
        raise ValueError('matrix_v4.csv no alinea frames 0..N-1')
    f = np.arange(0, n, step, dtype=int)
    per = np.array([config.get_period(int(k)) for k in f])
    valid = np.isin(per, ('FIRST_HALF','SECOND_HALF'))
    f, per = f[valid], per[valid]
    from starter.load_atd import load
    raw = load(game=config.GAME, source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name == 'frame_id': raw = raw.reset_index()
        else: raise ValueError('ATD sin frame_id')
    raw = raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw) != n or not np.array_equal(arr(raw, 'frame_id'), np.arange(1,n+1)):
        raise ValueError('ATD y matrix_v4 no coinciden')
    xx = [f'{i}_x' for i in range(1,47)]
    yy = [f'{i}_y' for i in range(1,47)]
    if not all(k in raw.columns for k in xx+yy): raise ValueError('Faltan slots ATD 1..46')
    x = raw.loc[f, xx].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)*P.length_m
    y = raw.loc[f, yy].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)*P.width_m
    ok = np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.length_m)&(y>=0)&(y<=P.width_m)
    x[~ok]=np.nan; y[~ok]=np.nan
    bx=arr(mat,'ball_x_normalized')[f]*P.length_m
    by=arr(mat,'ball_y_normalized')[f]*P.width_m
    obs=(arr(mat,'ball_observed',0)[f]>.5)&(arr(mat,'ball_imputed',0)[f]<.5)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=f, periods=per, x=x, y=y, bx=bx, by=by, obs=obs)


def team_pts(d,i,t):
    sl=slice(0,24) if t==0 else slice(24,46)
    pts=np.column_stack([d['x'][i,sl],d['y'][i,sl]])
    return pts[np.isfinite(pts).all(axis=1)]


def match_displacements(a,b,max_match_m=30):
    """Correspondencias espaciales 1:1 aproximadas, NO identidades.
    Devuelve desplazamiento a->b, filtrado por emparejamiento mutuo.
    """
    if not len(a) or not len(b): return np.empty((0,2))
    dd=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    aa=dd.argmin(axis=1)
    bb=dd.argmin(axis=0)
    pairs=[(i,int(j)) for i,j in enumerate(aa) if bb[j]==i and dd[i,j]<=max_match_m]
    if len(pairs)<3: return np.empty((0,2))
    return np.array([b[j]-a[i] for i,j in pairs],dtype=float)


def snapshot(d,i):
    out=[]
    for t in (0,1):
        p=team_pts(d,i,t)
        if len(p)<P.min_team: return None
        out.append(p)
    return out


def team_stats(points,side):
    y=points[:,1]; near=y<=P.target_near_band_m if side==0 else y>=P.width_m-P.target_near_band_m
    core=y<=P.preferred_side_band_m if side==0 else y>=P.width_m-P.preferred_side_band_m
    return dict(near=float(near.mean()), core=float(core.mean()),
                spread=float(np.median(np.linalg.norm(points-np.median(points,axis=0),axis=1))),
                center=np.median(points,axis=0))


def evidence(d,i,j,side):
    """Migración simultánea: ambos equipos; geometría + dirección + dispersión.
    No asumir slots como IDs persistentes.
    """
    if d['periods'][i]!=d['periods'][j]: return None
    if (d['frames'][j]-d['frames'][i])/P.fps < P.migration_min_s: return None
    starts=snapshot(d,i); ends=snapshot(d,j)
    if starts is None or ends is None: return None
    teams=[]
    sign=1.0 if side==1 else -1.0
    for a,b in zip(starts,ends):
        sa=team_stats(a,side); sb=team_stats(b,side)
        # desplazamiento de centroide robusto + diferencia marginal de ocupación;
        # no se exige match entre jugadores por 30 s.
        center_shift=sign*(sb['center'][1]-sa['center'][1])
        near_gain=sb['near']-sa['near']; core_gain=sb['core']-sa['core']
        shift_by_quantiles=np.quantile(sign*b[:,1],[.25,.5,.75])-np.quantile(sign*a[:,1],[.25,.5,.75])
        direction=float(np.mean(shift_by_quantiles>2.0))
        team_ok=(center_shift>=P.min_net_shift_m and near_gain>=P.min_band_occupancy_gain
                 and direction>=2/3 and sb['near']>=P.min_team_fraction)
        teams.append(dict(start_band=round(sa['near'],3),end_band=round(sb['near'],3),
                          band_gain=round(near_gain,3),core_gain=round(core_gain,3),
                          center_shift_m=round(float(center_shift),2),
                          quantile_direction=round(direction,2),
                          spread_ratio=round(sb['spread']/max(sa['spread'],.1),3),
                          valid=bool(team_ok)))
    if not all(t['valid'] for t in teams): return None
    return {'from_index':i,'to_index':j,'toward_side': 'y_min' if side==0 else 'y_max',
            'teams':teams}


def eval_indices(d):
    # 1 lectura / segundo, jamás reinterpretar 5 frames muestreados como evidencia nueva.
    f=d['frames']; selected=[]; prev=None
    for i,frame in enumerate(f):
        if prev is None or d['periods'][i]!=d['periods'][prev] or frame-f[prev]>=round(P.evaluation_interval_s*P.fps):
            selected.append(i); prev=i
    return selected


def motion_proxy(d,i,j):
    if d['periods'][i]!=d['periods'][j]: return np.nan
    dt=(d['frames'][j]-d['frames'][i])/P.fps
    if dt<=0:return np.nan
    aa=snapshot(d,i);bb=snapshot(d,j)
    if aa is None or bb is None: return np.nan
    vals=[]
    for a,b in zip(aa,bb):
        dx=match_displacements(a,b,max_match_m=7.0)
        if len(dx)<4:return np.nan
        vals.append(float(np.median(np.linalg.norm(dx,axis=1))/dt))
    return float(np.mean(vals))


def wait_evidence(d,inds,start,side):
    f=d['frames']; lim=f[start]+int(P.wait_max_s*P.fps)
    chosen=[k for k in inds if k>=start and f[k]<=lim and d['periods'][k]==d['periods'][start]]
    if len(chosen)<P.wait_min_s/P.evaluation_interval_s: return None
    near=[];quiet=[];spreads=[];coverage=[]
    # Encuentra la salida de la configuración compacta: corte en primera
    # racha suficientemente larga de dispersión o salida de banda.
    for j,k in enumerate(chosen):
        points=snapshot(d,k)
        if points is None:
            near.append(np.nan);spreads.append(np.nan); coverage.append(False)
        else:
            st=[team_stats(p,side) for p in points]
            near.append(min(z['near'] for z in st))
            spreads.append(float(np.mean([z['spread'] for z in st])))
            coverage.append(True)
        quiet.append(motion_proxy(d,chosen[j-1],k) if j>0 else np.nan)
    near=np.asarray(near); quiet=np.asarray(quiet);spreads=np.asarray(spreads)
    # Estado de espera: ambos equipos mantienen al menos ~40% junto a banda;
    # confirmación de baja movilidad evitando exigir inmovilidad absoluta.
    settled=(near>=.40)&(quiet<=P.max_wait_motion_mps)
    win=max(3,int(np.ceil(P.wait_min_s/P.evaluation_interval_s)))
    for j in range(win,len(chosen)+1):
        seg=slice(j-win,j)
        if np.isfinite(near[seg]).mean()<.75:continue
        if np.mean(settled[seg])>=P.min_wait_quiet_fraction:
            # el primer tramo de quietud puede continuar varios segundos
            k=j
            while k<len(chosen):
                last_start=max(0,k-win+1)
                if np.isfinite(near[k]) and near[k]>=.32 and (not np.isfinite(quiet[k]) or quiet[k]<=P.min_active_motion_mps):
                    k+=1
                else:
                    break
            end=chosen[min(k-1,len(chosen)-1)]
            if (f[end]-f[chosen[j-win]])/P.fps<P.wait_min_s-P.evaluation_interval_s:continue
            return dict(start=chosen[j-win],end=end,quiet_fraction=round(float(np.mean(settled[j-win:k])),3),
                        near_occupancy=round(float(np.nanmedian(near[j-win:k])),3),
                        mean_spread=round(float(np.nanmedian(spreads[j-win:k])),2))
    return None


def find_return(d,inds,wait,side):
    f=d['frames'];start=wait['end']; until=f[start]+int(P.return_max_s*P.fps)
    base=snapshot(d,start)
    if base is None:return None
    base_st=[team_stats(p,side) for p in base]
    for k in inds:
        if k<=start or f[k]>until or d['periods'][k]!=d['periods'][start]:continue
        pt=snapshot(d,k)
        if pt is None:continue
        st=[team_stats(p,side) for p in pt]
        banddrops=[a['near']-b['near'] for a,b in zip(base_st,st)]
        spreadgains=[b['spread']-a['spread'] for a,b in zip(base_st,st)]
        retreats=[(b['center'][1]-a['center'][1])*(1 if side==0 else -1) for a,b in zip(base_st,st)]
        if all(v>=P.min_return_band_drop for v in banddrops) and all(v>=P.min_net_shift_m for v in retreats):
            return dict(frame=int(f[k]), index=int(k), band_drops=[round(v,3) for v in banddrops],
                        spread_gains_m=[round(v,2) for v in spreadgains])
    return None


def ball_restart(d,from_index,to_index):
    f=d['frames']; z=np.flatnonzero(d['obs'] & (f>=f[from_index])&(f<=f[to_index]))
    for i,j in zip(z[:-1],z[1:]):
        dt=(f[j]-f[i])/P.fps
        if dt<=0 or dt>P.max_ball_obs_gap_s or d['periods'][i]!=d['periods'][j]:continue
        dist=float(np.hypot(d['bx'][j]-d['bx'][i],d['by'][j]-d['by'][i]))
        if dist>=P.ball_restart_min_dist_m and dist/dt>=P.ball_restart_min_speed_mps:
            return {'from_frame':int(f[i]),'to_frame':int(f[j]),'observed_displacement_m':round(dist,2)}
    return None


def detect(d):
    f=d['frames'];inds=eval_indices(d); events=[];seen_until=-1
    # Optimización: detección candidata en ventanas 5,10,15,20,30s.
    for pos,j in enumerate(inds):
        if f[j]<=seen_until:continue
        best=None
        for duration in (5,10,15,20,30):
            target=f[j]-duration*P.fps
            z=np.searchsorted(f,target,side='left')
            if z>=j or z<0 or f[j]-f[z] > (duration+P.evaluation_interval_s)*P.fps: continue
            if d['periods'][z]!=d['periods'][j]:continue
            for side in (0,1):
                ev=evidence(d,int(z),j,side)
                if ev and (best is None or sum(t['band_gain'] for t in ev['teams'])>sum(t['band_gain'] for t in best['teams'])):
                    best=ev
        if best is None:continue
        wait=wait_evidence(d,inds,j,0 if best['toward_side']=='y_min' else 1)
        if wait is None:continue
        ret=find_return(d,inds,wait,0 if best['toward_side']=='y_min' else 1)
        # Ser estricto con la vuelta: salida sin regreso = episodio provisional, no confirmado.
        wstart=wait['start']; wend=wait['end']
        # Rango para reanudación: buscar desde después de la espera, no atribuir
        # circulación del balón durante migración como golpeo final.
        stop_at = int(np.searchsorted(f,ret['frame']+10*P.fps,side='right')-1) if ret else wend
        stop_at=max(wend,min(stop_at,len(f)-1))
        restart=ball_restart(d,wend,stop_at) if ret else None
        score=3+2*bool(ret)+int(wait['quiet_fraction']>=.75)+int(bool(restart))
        level='COLLECTIVE_BREAK_STRONG' if ret and score>=6 else 'COLLECTIVE_BREAK_CANDIDATE'
        # No clasificar cooling/drinks por segundos sin observación del contexto.
        migration_start_frame=int(f[best['from_index']]);wait_start_frame=int(f[wstart])
        events.append(dict(type='TH12_COLLECTIVE_BREAK',classification=level,
                           cause='DRINKS_OR_COOLING_UNVERIFIED',score_heuristic=int(score),
                           migration_start_frame=migration_start_frame,
                           arrival_frame=int(f[j]),wait_start_frame=wait_start_frame,
                           wait_end_frame=int(f[wend]),
                           possible_D_start_window=[max(int(f[0]),migration_start_frame-5*P.fps),wait_start_frame],
                           possible_D_end_window=([restart['from_frame'],restart['to_frame']] if restart else None),
                           return_motion=ret, observed_restart=restart,
                           migration=best, wait=wait,
                           warnings=['No hay señal de árbitro ni de hidratación en tracking',
                                     'Las fronteras de D son intervalos, no verdades de terreno']))
        seen_until=(ret['frame'] if ret else int(f[wend]))+round(P.episode_exclusion_s*P.fps)
        if len(events)>=P.max_candidates:break
    return events


def self_test():
    # Dos equipos migran hacia y=0, esperan, luego regresan.
    rng=np.random.default_rng(2);fps=P.fps
    f=np.arange(0,110*fps,fps,dtype=int);n=len(f)
    x=np.full((n,46),np.nan);y=np.full_like(x,np.nan)
    for k in range(n):
        s=k
        if s<12: shift=0
        elif s<27:shift=(s-12)/15
        elif s<70:shift=1
        elif s<90:shift=1-(s-70)/20
        else:shift=0
        for t,(aa,bb) in enumerate(((0,11),(24,35))):
            base=np.linspace(16,89,11)+(t*1.0)
            yy=np.linspace(17,52,11)
            x[k,aa:bb]=base
            y[k,aa:bb]=yy*(1-shift)+3.0*shift
            # una fluctuación ligera del tracking durante espera
            x[k,aa:bb]+=rng.normal(0,.10,11)
            y[k,aa:bb]+=rng.normal(0,.10,11)
    d=dict(frames=f,periods=np.array(['FIRST_HALF']*n),x=x,y=y,
           bx=np.full(n,np.nan),by=np.full(n,np.nan),obs=np.zeros(n,bool))
    result=detect(d)
    assert result, 'Debería detectar migración, espera y regreso'
    assert result[0]['return_motion'] is not None, 'No detectó vuelta'
    normal=dict(d);normal['y']=np.copy(y)
    for k in range(n):
        normal['y'][k,0:11]=np.linspace(17,52,11)
        normal['y'][k,24:35]=np.linspace(17,52,11)
    assert not detect(normal),'Falso positivo de juego sin migración'
    one=dict(d);one['y']=np.copy(y)
    for k in range(n):one['y'][k,24:35]=np.linspace(17,52,11)
    assert not detect(one),'Falso positivo de migración unilateral'
    print('TH12 self-test OK: bilateral, rechazo sin migración y unilateral')


def main():
    ap=argparse.ArgumentParser(description='TH12: pausas colectivas / hidratación compatible')
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',type=Path,default=Path('local_outputs/runs'))
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    d=load_data(args.step)
    ev=detect(d)
    args.out.mkdir(parents=True,exist_ok=True)
    path=args.out/'th12_cooling_break_diagnostic.json'
    payload={'detector':'TH12','parameters':asdict(P),'n_observations':len(d['frames']),
             'events':ev,'note':'NO modifica H/A/D. No prueba hidratación ni silbato.'}
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'TH12: {len(ev)} candidatos; salida: {path}')

if __name__=='__main__':main()
