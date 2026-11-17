"""TH10 - Detección conservadora de PAUSA PROLONGADA / posible revisión VAR.

Independiente: sólo genera diagnóstico JSON; NO modifica H/A/D ni run.py.
Dependencias: config.py, starter.load_atd, matrix_v4.csv, numpy, pandas.
Ejecutar: python th10_var_review.py --self-test
          python th10_var_review.py --step 5 --out local_outputs/runs

NO identifica VAR por quietud: distinción explícita entre PAUSA y VAR corroborado.
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
    pitch_length_m: float = 105.3  # escala proyecto pendiente verificación
    pitch_width_m: float = 68.0
    min_visible_team: int = 7
    lag_s: float = 1.0
    min_lag_factor: float = 0.65
    max_lag_factor: float = 1.55
    match_radius_m: float = 7.0
    motion_quiet_mps: float = 1.10  # proxy, NO velocidad individual
    motion_active_mps: float = 1.75
    baseline_lookback_s: float = 8.0
    min_drop_mps: float = 0.50
    enter_persistence_s: float = 2.0
    min_stop_s: float = 12.0
    prolonged_s: float = 25.0
    long_s: float = 45.0
    quiet_fraction: float = 0.62
    max_sample_gap_s: float = 0.85
    end_confirm_s: float = 2.0
    recovery_motion_mps: float = 1.60
    max_scan_s: float = 180.0
    restart_ball_min_m: float = 1.25
    restart_ball_speed_mps: float = 2.5
    max_ball_observation_gap_s: float = 1.0
    drift_stable_m: float = 10.0
    recent_s: float = 12.0
    candidate_merge_gap_s: float = 5.0
    provider_context_s: float = 40.0
    ball_motion_radius_m: float = 1.25

P = Parameters()

def num(df, col, fill=np.nan):
    if col not in df:
        return np.full(len(df), fill, dtype=float)
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)

def load_data(step: int):
    import config
    if step < 1: raise ValueError('--step debe ser >= 1')
    if int(config.FPS) != P.fps: raise ValueError('FPS distinto: parametrizar P.fps')
    mat = pd.read_csv(Path(config.OUTPUT_DIR) / 'matrix_v4.csv', low_memory=False)
    n = int(config.N_VIDEO_FRAMES)
    if len(mat) != n or not np.array_equal(num(mat, 'frame'), np.arange(n)):
        raise ValueError('matrix_v4.csv no coincide con frames 0..N-1')
    f = np.arange(0, n, step, dtype=int)
    periods = np.asarray([config.get_period(int(k)) for k in f])
    mask = np.isin(periods, ['FIRST_HALF', 'SECOND_HALF'])
    f, periods = f[mask], periods[mask]
    from starter.load_atd import load
    raw = load(game=config.GAME, source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name == 'frame_id': raw = raw.reset_index()
        else: raise ValueError('ATD sin frame_id')
    raw = raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw) != n or not np.array_equal(num(raw, 'frame_id'), np.arange(1,n+1)):
        raise ValueError('ATD no alinea 1..N')
    xc = [f'{i}_x' for i in range(1,47)]
    yc = [f'{i}_y' for i in range(1,47)]
    if any(x not in raw.columns for x in xc + yc): raise ValueError('ATD sin slots 1..46')
    x = raw.loc[f, xc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32) * P.pitch_length_m
    y = raw.loc[f, yc].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32) * P.pitch_width_m
    valid = np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.pitch_length_m)&(y>=0)&(y<=P.pitch_width_m)
    x[~valid] = np.nan; y[~valid] = np.nan
    bx = num(mat, 'ball_x_normalized')[f] * P.pitch_length_m
    by = num(mat, 'ball_y_normalized')[f] * P.pitch_width_m
    observed = (num(mat,'ball_observed',0)[f]>.5)&(num(mat,'ball_imputed',0)[f]<.5)&np.isfinite(bx)&np.isfinite(by)
    return {'frames':f,'periods':periods,'x':x,'y':y,'bx':bx,'by':by,'obs':observed}

def provider_events():
    import config
    try:
        from starter.load_smart_tagging import load
        events = load(game=config.GAME)
    except (ImportError,FileNotFoundError,ValueError,OSError,KeyError,TypeError): return []
    result=[]
    for e in events:
        if not isinstance(e,dict): continue
        tags=e.get('tags') or {}
        s=' '.join(map(str,(tags.get('Type',''),e.get('type',''),e.get('code','')))).lower()
        # Los tipos ordinarios son contexto, NO prueba de VAR.
        relevant = ['var','video review','review','goal','penalty','red card','offside','injury','substitution']
        if not any(w in s for w in relevant):continue
        try: a,b=int(e['start_frame']),int(e['end_frame'])
        except (KeyError,ValueError,TypeError):continue
        if b<=a:continue
        explicit = any(w in s for w in ['var','video review','var check','on-field review'])
        result.append({'start_frame':a,'end_exclusive':b,'type':s,'explicit_var':explicit})
    return result

def team_points(d,i,t):
    sl = slice(0,24) if t==0 else slice(24,46)
    p=np.column_stack((d['x'][i,sl],d['y'][i,sl]))
    return p[np.isfinite(p).all(axis=1)]

def _movement_proxy(a,b,dt):
    if min(len(a),len(b))<P.min_visible_team or dt<=0:return np.nan
    dist=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    q=np.r_[dist.min(axis=0),dist.min(axis=1)]
    valid=q<=P.match_radius_m
    if valid.sum()<2*P.min_visible_team:return np.nan
    return float(np.median(q[valid])/dt)

def features(d):
    f=d['frames'];n=len(f)
    motion=np.full((n,2),np.nan);centroids=np.full((n,2,2),np.nan)
    coverage=np.zeros((n,2),dtype=int)
    for i in range(n):
        pts=[team_points(d,i,t) for t in (0,1)]
        for t in (0,1):
            coverage[i,t]=len(pts[t])
            if len(pts[t])>=P.min_visible_team:centroids[i,t]=np.median(pts[t],axis=0)
        j=int(np.searchsorted(f,f[i]-P.lag_s*P.fps,side='right')-1)
        if j>=0 and j<i and d['periods'][j]==d['periods'][i]:
            dt=(f[i]-f[j])/P.fps
            if P.lag_s*P.min_lag_factor<=dt<=P.lag_s*P.max_lag_factor:
                for t in (0,1):motion[i,t]=_movement_proxy(pts[t],team_points(d,j,t),dt)
    valid=np.all(coverage>=P.min_visible_team,axis=1)&np.isfinite(motion).all(axis=1)
    m=np.full(n,np.nan);m[valid]=motion[valid].mean(axis=1)
    return {'motion':m,'coverage':coverage,'centroid':centroids,'valid':valid}

def _ball_motion(d,lo,hi):
    f=d['frames'];idx=np.flatnonzero(d['obs']&(f>=lo)&(f<=hi));ans=[]
    for a,b in zip(idx[:-1],idx[1:]):
        if d['periods'][a]!=d['periods'][b]:continue
        dt=(f[b]-f[a])/P.fps
        if dt<=0 or dt>P.max_ball_observation_gap_s:continue
        ds=float(np.hypot(d['bx'][b]-d['bx'][a],d['by'][b]-d['by'][a]))
        if ds>=P.restart_ball_min_m and ds/dt>=P.restart_ball_speed_mps:
            ans.append({'from_frame':int(f[a]),'to_frame':int(f[b]),'meters':round(ds,2)})
    return ans

def _sustained(d,ft,idx,seconds,mode='quiet'):
    if len(idx)<3:return False
    f=d['frames'];duration=(f[idx[-1]]-f[idx[0]])/P.fps
    if duration < seconds:return False
    if np.max(np.diff(f[idx]))/P.fps>P.max_sample_gap_s:return False
    good=ft['valid'][idx]
    if good.mean()<.75:return False
    if mode=='quiet': q=(ft['motion'][idx]<=P.motion_quiet_mps)
    else:q=(ft['motion'][idx]>=P.recovery_motion_mps)
    return bool(np.mean(q&good)>=P.quiet_fraction)

def _context(providers,a,b):
    near=[]
    for e in providers:
        if e['start_frame']<=b+P.provider_context_s*P.fps and e['end_exclusive']>=a-P.provider_context_s*P.fps:
            near.append(e)
    return near

def detect(d, providers=None):
    """Conservador: episodios candidatos; no etiqueta VAR sin evidencia explícita."""
    providers=[] if providers is None else providers
    f=d['frames'];n=len(f)
    if n<5:return []
    ft=features(d)
    quiet=ft['valid']&(ft['motion']<=P.motion_quiet_mps)
    episodes=[];i=1
    while i<n:
        if not quiet[i]:i+=1;continue
        if quiet[i-1] and d['periods'][i]==d['periods'][i-1]:i+=1;continue
        a=i;per=d['periods'][i]
        # Demostrar caída respecto a actividad: no basta estar quieto desde el comienzo.
        prev=np.flatnonzero((f>=f[i]-P.baseline_lookback_s*P.fps)&(f<f[i])&(d['periods']==per)&ft['valid'])
        pre=float(np.nanquantile(ft['motion'][prev],.75)) if len(prev)>=3 else np.nan
        if not np.isfinite(pre) or pre<P.motion_active_mps or pre-ft['motion'][i]<P.min_drop_mps:
            i+=1;continue
        # Construir intervalo de espera, admitiendo movimiento temporal y sin salvar grandes huecos.
        j=i;maxj=int(np.searchsorted(f,f[i]+P.max_scan_s*P.fps,side='right'))
        while j+1<min(n,maxj) and d['periods'][j+1]==per:
            if (f[j+1]-f[j])/P.fps>P.max_sample_gap_s:break
            k=j+1
            if not ft['valid'][k]:break
            recent=np.flatnonzero((f>=f[k]-P.recent_s*P.fps)&(f<=f[k])&(d['periods']==per))
            if (f[k]-f[i])/P.fps>=P.min_stop_s and len(recent)>=3:
                good=np.mean(quiet[recent])
                if good<P.quiet_fraction and np.nanmedian(ft['motion'][recent])>=P.recovery_motion_mps:break
            j=k
        idx=np.arange(a,j+1)
        duration=(f[j]-f[a])/P.fps
        if duration<P.min_stop_s or not _sustained(d,ft,idx,P.enter_persistence_s):
            i=max(i+1,j+1);continue
        # Centroide: cuenta cuánto se desplazó la distribución durante la pausa; no es árbitro.
        movement=[]
        for team in (0,1):
            c=ft['centroid'][idx,team,:]
            c=c[np.isfinite(c).all(axis=1)]
            if len(c)>=2:movement.append(float(np.linalg.norm(c[-1]-c[0])))
        spatial_stability=bool(len(movement)==2 and max(movement)<=P.drift_stable_m)
        # En reposo pueden moverse jugadores a un lado: la estabilidad no es requisito.
        # Fin potencial: primera recuperación sostenida posterior; NO atribuirlo a golpeo automáticamente.
        end=None
        search_to=int(np.searchsorted(f,f[j]+20*P.fps,side='right'))
        for k in range(j+1,min(search_to,n)):
            if d['periods'][k]!=per:break
            r=np.flatnonzero((f>=f[k])&(f<=f[k]+P.end_confirm_s*P.fps)&(d['periods']==per))
            if _sustained(d,ft,r,P.end_confirm_s*.8,'active'):
                end={'movement_recovery_lower_frame':int(f[j]),'movement_recovery_upper_frame':int(f[k]),'not_proven_kick':True};break
        associated=_context(providers,int(f[a]),int(f[j]))
        explicit=any(e['explicit_var'] for e in associated)
        context_kinds=sorted({e['type'] for e in associated if not e['explicit_var']})
        # Detectar desplazamientos del balón cerca del final, nunca confundir con restart legal.
        ball_moves=_ball_motion(d,int(f[max(a,j-5)]),int(f[min(n-1,j+max(1,int(P.fps/5)))]))
        if duration>=P.long_s: band='VERY_LONG'
        elif duration>=P.prolonged_s: band='PROLONGED'
        else:band='EXTENDED'
        episodes.append({
            'kind':'VAR_CONTEXT_SUPPORTED' if explicit else 'PROLONGED_STOPPAGE_UNSPECIFIED',
            'var_confirmed':False, 'provider_explicit_var':explicit,
            'length_band':band,'quiet_start_frame':int(f[a]),'quiet_end_frame':int(f[j]),
            'quiet_duration_s':round(float(duration),2),
            'dead_start_bracket':{'lower_frame':int(f[prev[-1]]) if len(prev) else None,'upper_frame':int(f[a]),'is_estimate':True},
            'dead_end_bracket':end,
            'ball_motion_near_transition':ball_moves[:8],
            'teams_centroid_drift_m':[round(v,2) for v in movement],
            'spatially_stable':spatial_stability,
            'prior_activity_proxy_mps':round(pre,3),
            'provider_context_types':context_kinds[:12],
            'evidence':{'activity_to_quiet':True,'sustained_quiet':True,'explicit_var_tag':explicit,
                        'duration_s':round(float(duration),2)},
            'limitations':['No referee or VAR signal in tracking','May be injury, cooling break, substitution, goal celebration or administrative delay',
                           'Neither boundaries nor VAR status is adjudicated']
        })
        i=max(i+1,j+1)
    # No fusionar episodios si hay recuperación sostenida entre ellos.
    return episodes

def self_test():
    # Pruebas internas de los elementos matemáticos y de no-confirmación sin proveedor.
    a=np.array([[0.,0.],[10.,10.],[20.,20.],[30.,30.],[40.,40.],[50.,50.],[60.,60.]])
    assert _movement_proxy(a,a,1.)==0.
    f=np.arange(0,1500,5);n=len(f)
    motion=np.r_[np.full(35,2.25),np.full(180,.30),np.full(n-215,2.2)]
    # Test de detección puramente temporal reemplazando la extracción espacial;
    # ejercita agrupación, fronteras y no confirmación de VAR.
    dd={'frames':f,'periods':np.array(['FIRST_HALF']*n),'x':np.full((n,46),np.nan),
        'y':np.full((n,46),np.nan),'bx':np.full(n,np.nan),'by':np.full(n,np.nan),'obs':np.zeros(n,bool)}
    old=globals()['features']
    try:
        globals()['features']=lambda d:{'motion':motion,'valid':np.ones(n,bool),
                         'centroid':np.zeros((n,2,2)),'coverage':np.full((n,2),11)}
        result=detect(dd,[])
        assert result and not result[0]['var_confirmed'] and result[0]['kind']=='PROLONGED_STOPPAGE_UNSPECIFIED',result
        assert result[0]['quiet_duration_s']>=P.min_stop_s
        globals()['features']=lambda d:{'motion':np.full(n,.3),'valid':np.ones(n,bool),
                         'centroid':np.zeros((n,2,2)),'coverage':np.full((n,2),11)}
        assert not detect(dd,[]), 'Inmovilidad sin transición no debe activar'
    finally:
        globals()['features']=old
    print('TH10 self-test OK (synthetic; no full-match validation)')

def main():
    parser=argparse.ArgumentParser(description='TH10 pausa prolongada / VAR no confirmado')
    parser.add_argument('--step',type=int,default=5)
    parser.add_argument('--out',default='local_outputs/runs')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    d=load_data(args.step)
    events=detect(d,provider_events())
    result={'detector':'TH10_PROLONGED_REVIEW','parameters':asdict(P),'input_step':args.step,
            'episodes':events,'count':len(events),
            'WARNING':'No se confirma VAR sin evidencia externa. Ningún resultado asigna H/A/D.'}
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    dest=out/f'th10_var_review_step{args.step}.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{dest} | candidatos={len(events)} | VAR no confirmado')

if __name__=='__main__':main()
