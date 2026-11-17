"""TH08 GOAL. Detector temporal independiente; NO modifica run.py ni H/A/D.

Uso: python th08_goal.py --step 5 --out local_outputs/runs
     python th08_goal.py --self-test
Requiere config.py, starter/, matrix_v4.csv; numpy y pandas.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import config

FPS=int(config.FPS)

@dataclass(frozen=True)
class Parameters:
    length_m:float=105.3
    width_m:float=68.0
    goal_width_m:float=7.32
    # El radio es aproximación: tracking 2D puede ser centro de balón, no superficie.
    ball_radius_m:float=0.11
    goal_lateral_tolerance_m:float=0.60
    goal_line_tolerance_m:float=0.65
    ball_cross_max_gap_s:float=0.80
    ball_cross_min_advance_m:float=0.35
    approach_zone_m:float=23.0
    near_goal_y_tolerance_m:float=6.0
    minimum_team_visible:int=8
    center_half_tolerance_m:float=1.6
    own_half_ratio:float=.75
    center_taker_radius_m:float=3.8
    circle_radius_m:float=9.15
    circle_tolerance_m:float=1.3
    near_center_ball_m:float=2.2
    formation_stable_s:float=2.0
    formation_min_samples:int=3
    formation_min_ratio:float=.60
    center_lookahead_s:float=120.0
    ball_pre_goal_lookback_s:float=8.0
    celebration_motion_s:float=5.0
    goal_formation_min_delay_s:float=4.0
    goal_formation_max_delay_s:float=95.0
    kickoff_kick_search_s:float=18.0
    kickoff_ball_min_move_m:float=.7
    kickoff_ball_min_speed_mps:float=1.8
    dedup_s:float=18.0
    provider_match_s:float=12.0
    team_slot_split:int=24
P=Parameters()

def numeric(df,key,default=np.nan):
    return pd.to_numeric(df[key],errors='coerce').to_numpy(dtype=float) if key in df else np.full(len(df),default)

def period(f):return config.get_period(int(f))

def load_data(step):
    if step<1:raise ValueError('step debe ser >=1')
    mat=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    n=len(mat)
    if n!=int(config.N_VIDEO_FRAMES) or not np.array_equal(numeric(mat,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv desalineada con vídeo')
    frames=np.arange(0,n,step,dtype=int)
    frames=frames[np.array([period(f) in ('FIRST_HALF','SECOND_HALF') for f in frames])]
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('Tracking carece de frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(numeric(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD no alineado con vídeo')
    xx=[f'{i}_x' for i in range(1,47)]; yy=[f'{i}_y' for i in range(1,47)]
    if any(k not in raw for k in xx+yy):raise ValueError('Faltan slots de jugadores 1..46')
    x=raw.loc[frames,xx].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.length_m
    y=raw.loc[frames,yy].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*P.width_m
    ok=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=P.length_m)&(y>=0)&(y<=P.width_m)
    x[~ok]=np.nan;y[~ok]=np.nan
    bx=numeric(mat,'ball_x_normalized')[frames]*P.length_m
    by=numeric(mat,'ball_y_normalized')[frames]*P.width_m
    observed=(numeric(mat,'ball_observed',0)[frames]>.5)&(numeric(mat,'ball_imputed',0)[frames]<.5)&np.isfinite(bx)&np.isfinite(by)
    # Coordenadas 2D recortadas no prueban paso completo de la línea.
    return dict(frames=frames,x=x,y=y,ok=ok,bx=bx,by=by,obs=observed,
                periods=np.array([period(f) for f in frames]))

def provider_goals():
    try:
        from starter.load_smart_tagging import load
        events=load(game=config.GAME)
    except (ImportError,OSError,KeyError,ValueError,TypeError,FileNotFoundError):return []
    out=[]
    for e in events:
        if not isinstance(e,dict):continue
        tags=e.get('tags') or {}
        label=str(tags.get('Type',e.get('type',''))).strip().lower()
        code=str(e.get('code','')).strip().lower()
        # Excluir goal kick: 'goal' solo es demasiado amplio.
        if label not in ('goal','goals','own goal','own_goal') and code not in ('goal','goals','own goal','own_goal'):
            continue
        try:a=int(e['start_frame']);b=int(e['end_frame'])
        except (KeyError,ValueError,TypeError):continue
        out.append(dict(start_frame=a,end_exclusive=b,team=str(tags.get('Team','')),
                        label=label,source='SMART_TAGGING_AUTO_NOT_GT'))
    return sorted(out,key=lambda v:v['start_frame'])

def goal_mouth(y,tolerance=0):
    return bool(abs(y-P.width_m/2)<=P.goal_width_m/2+tolerance)

def ball_goal_evidence(d):
    """Encuentra tránsito hacia la zona de boca y eventual cruce de x=0/L.
    Un centro de balón observado más allá de x=0/L puede no existir por clipping.
    Ningún punto 2D demuestra altura bajo travesaño o validez arbitral.
    """
    out=[]; idx=np.flatnonzero(d['obs']); frames=d['frames']
    for a,b in zip(idx[:-1],idx[1:]):
        if d['periods'][a]!=d['periods'][b]:continue
        dt=(frames[b]-frames[a])/FPS
        if dt<=0 or dt>P.ball_cross_max_gap_s:continue
        xa,xb=float(d['bx'][a]),float(d['bx'][b]);ya,yb=float(d['by'][a]),float(d['by'][b])
        for side,line,sign in (('LEFT',0.,-1),('RIGHT',P.length_m,1)):
            progress=sign*(xb-xa)
            if progress<P.ball_cross_min_advance_m:continue
            if min(abs(xa-line),abs(xb-line))>P.approach_zone_m:continue
            # interpolate projection to goal line, clamp to observed trajectory span.
            r=np.clip((line-xa)/(xb-xa),0,1) if abs(xb-xa)>1e-7 else 0.
            yi=ya+r*(yb-ya)
            if not goal_mouth(yi,P.goal_lateral_tolerance_m):continue
            contact=abs(xb-line)<=P.goal_line_tolerance_m
            real_cross=(sign*(xa-line)<=-P.ball_radius_m and sign*(xb-line)>=P.ball_radius_m)
            if contact or real_cross:
                out.append(dict(frame_before=int(frames[a]),frame_after=int(frames[b]),
                   side=side,interpolated_y_m=round(float(yi),3),
                   observed_full_center_cross=bool(real_cross),
                   clipped_or_boundary=not bool(real_cross),
                   advance_m=round(float(progress),3),
                   evidence='2D_GOAL_LINE_COMPATIBLE_NOT_PROVEN'))
    return out

def center_formation(d):
    """Configuración de saque de centro; prueba colectiva, nunca gol por sí sola."""
    x,y,ok=d['x'],d['y'],d['ok'];cx,cy=P.length_m/2,P.width_m/2
    n=len(d['frames']);visible=np.zeros((n,2),int);frac=np.zeros((n,2,2))
    inside_circle=np.hypot(x-cx,y-cy)<=P.circle_radius_m+P.circle_tolerance_m
    near_center=np.hypot(x-cx,y-cy)<=P.center_taker_radius_m
    for t,sl in enumerate((slice(0,24),slice(24,46))):
        visible[:,t]=ok[:,sl].sum(axis=1)
    for h in (0,1):
        for t,sl in enumerate((slice(0,24),slice(24,46))):
            left=(t==0)==(h==0)
            own=(x[:,sl]<=cx+P.center_half_tolerance_m) if left else (x[:,sl]>=cx-P.center_half_tolerance_m)
            frac[:,h,t]=(own&ok[:,sl]).sum(axis=1)/np.maximum(visible[:,t],1)
    present=np.all(visible>=P.minimum_team_visible,axis=1)
    own=np.max(np.min(frac,axis=2),axis=1)>=P.own_half_ratio
    # Por lo general quedan al menos 1-2 jugadores junto al centro y el resto fuera del círculo.
    middle=(near_center&ok).sum(axis=1)
    others=(inside_circle&ok).sum(axis=1)
    ball_center=d['obs']&(np.hypot(d['bx']-cx,d['by']-cy)<=P.near_center_ball_m)
    formation=present&own&(middle>=1)&(others<=5)
    # El balón puede faltar, pero la formación debe ser más restrictiva si no se observa.
    formation=formation&(ball_center | ((middle<=3)&(others<=3)))
    return dict(formation=formation,ball_center=ball_center,own_fraction=frac,
                visible=visible,near_center=middle,inside_circle=others)

def formation_runs(d,ft):
    frames=d['frames'];f=ft['formation'];out=[];s=None
    step_est=int(np.median(np.diff(frames))) if len(frames)>1 else 1
    max_gap=max(FPS,5*step_est)
    for i in range(len(f)+1):
        split=i==len(f) or (i>0 and d['periods'][i]!=d['periods'][i-1]) or (i>0 and frames[i]-frames[i-1]>max_gap)
        if i<len(f) and f[i] and s is None:s=i
        if s is not None and (i==len(f) or not f[i] or split):
            e=i if i==len(f) else i-1
            # Keep only real time spans with repeated independent observations.
            if e>=s and (frames[e]-frames[s])/FPS>=P.formation_stable_s and e-s+1>=P.formation_min_samples:
                out.append(dict(first=int(frames[s]),last=int(frames[e]),first_idx=s,last_idx=e,
                                ball_center_seen=bool(np.any(ft['ball_center'][s:e+1]))))
            s=i if i<len(f) and f[i] else None
    return out

def collective_return(d,form,ft):
    """Compara ocupación de mitades antes de la formación sin identificar slots.
    Valor desconocido si no hay cobertura temporal. La retirada NO es requisito.
    """
    frame=d['frames'];i=int(form['first_idx']);target=frame[i]-round(8*FPS)
    j=int(np.searchsorted(frame,target))
    if j>=i or d['periods'][j]!=d['periods'][i]:
        return dict(known=False,reason='NO_SAME_PERIOD_HISTORY')
    if np.any(ft['visible'][j]<P.minimum_team_visible):
        return dict(known=False,reason='INSUFFICIENT_PREVIOUS_COVERAGE')
    orientation=int(np.argmax(np.min(ft['own_fraction'][i],axis=1)))
    before=ft['own_fraction'][j,orientation];after=ft['own_fraction'][i,orientation]
    delta=after-before
    return dict(known=True,observed_seconds=round(float((frame[i]-frame[j])/FPS),2),
        orientation_hypothesis='TEAM0_LEFT' if orientation==0 else 'TEAM0_RIGHT',
        own_half_before=np.round(before,3).tolist(),own_half_after=np.round(after,3).tolist(),
        own_half_gain=np.round(delta,3).tolist(),
        both_teams_returned=bool(np.all(delta>=.08)),
        note='Ocupaciones por equipo; no son trayectorias individuales')

def kickoff_motion(d,form):
    """Acota kick-off entre dos muestras observadas (no finge golpeo exacto)."""
    frame=d['frames'];start=np.searchsorted(frame,form['first']);end=np.searchsorted(frame,form['last']+int(P.kickoff_kick_search_s*FPS),side='right')
    indexes=np.flatnonzero(d['obs'][start:end])+start;cx,cy=P.length_m/2,P.width_m/2
    for a,b in zip(indexes[:-1],indexes[1:]):
        if d['periods'][a]!=d['periods'][b]:continue
        dt=(frame[b]-frame[a])/FPS
        if not 0<dt<=P.ball_cross_max_gap_s:continue
        r0=np.hypot(d['bx'][a]-cx,d['by'][a]-cy)
        delta=np.hypot(d['bx'][b]-d['bx'][a],d['by'][b]-d['by'][a])
        if r0<=P.near_center_ball_m and delta>=P.kickoff_ball_min_move_m and delta/dt>=P.kickoff_ball_min_speed_mps:
            return dict(before=int(frame[a]),after=int(frame[b]),source='OBSERVED_BALL_BRACKET')
    return None

def recent_goalward_ball(d,goal_frame,side):
    frames=d['frames'];i=np.searchsorted(frames,goal_frame,side='right')-1
    if i<0:return False
    j=np.searchsorted(frames,goal_frame-round(P.ball_pre_goal_lookback_s*FPS))
    line=0. if side=='LEFT' else P.length_m
    sel=d['obs'][j:i+1]& (np.abs(d['bx'][j:i+1]-line)<P.approach_zone_m)&(np.abs(d['by'][j:i+1]-P.width_m/2)<=P.goal_width_m/2+P.near_goal_y_tolerance_m)
    return bool(np.any(sel))

def discover(d,providers):
    """Anclas independientes: proveedor, balón cercano al cruce, transición al saque de centro.
    La última exige balón en zona de peligro previa, para excluir los inicios de periodos.
    """
    crossings=ball_goal_evidence(d);ft=center_formation(d);forms=formation_runs(d,ft);anchors=[]
    for c in crossings:anchors.append(dict(frame=c['frame_after'],origin='BALL',side=c['side'],cross=c))
    for e in providers:
        if period(e['start_frame']) in ('FIRST_HALF','SECOND_HALF'):
            anchors.append(dict(frame=e['start_frame'],origin='PROVIDER',side=None,provider=e))
    # Formación sin proveedor ni cruce: solo candidato de menor confianza.
    for form in forms:
        if anchors and min(abs(form['first']-int(x['frame'])) for x in anchors) < P.dedup_s*FPS:
            continue
        period_start=(config.FIRST_HALF_START if period(form['first'])=='FIRST_HALF' else config.SECOND_HALF_START)
        if form['first']-period_start<round(40*FPS):continue
        for side in ('LEFT','RIGHT'):
            lo=form['first']-round(P.goal_formation_max_delay_s*FPS)
            hi=form['first']-round(P.goal_formation_min_delay_s*FPS)
            idx=np.searchsorted(d['frames'],hi,side='right')-1
            if idx<0:continue
            previous=np.flatnonzero(d['obs']& (d['frames']>=lo)&(d['frames']<=hi)
                     &(np.abs(d['bx']-(0 if side=='LEFT' else P.length_m))<P.approach_zone_m)
                     &(np.abs(d['by']-P.width_m/2)<P.goal_width_m/2+P.near_goal_y_tolerance_m))
            if not len(previous):continue
            anchors.append(dict(frame=int(d['frames'][previous[-1]]),origin='FORMATION_BACKWARD',side=side,form_hint=form))
    # Deduplicar sin concatenar goles potencialmente separados: agrupar proximidad temporal corta.
    priority={'PROVIDER':3,'BALL':2,'FORMATION_BACKWARD':1}
    anchors.sort(key=lambda a:(a['frame'],-priority[a['origin']]))
    groups=[]
    for a in anchors:
        if not groups or a['frame']-groups[-1][-1]['frame']>round(P.dedup_s*FPS):groups.append([a])
        else:groups[-1].append(a)
    results=[]
    for g in groups:
        best=max(g,key=lambda a:priority[a['origin']]);frame=int(best['frame']);
        matching_forms=[f for f in forms if frame+P.goal_formation_min_delay_s*FPS<=f['first']<=frame+P.center_lookahead_s*FPS and period(f['first'])==period(frame)]
        form=matching_forms[0] if matching_forms else None
        crosses=[a['cross'] for a in g if a['origin']=='BALL']
        tags=[a['provider'] for a in g if a['origin']=='PROVIDER']
        nearby_provider=[e for e in providers if abs(e['start_frame']-frame)<=P.provider_match_s*FPS]
        if nearby_provider and not tags:tags=nearby_provider
        side=next((x['side'] for x in g if x.get('side')),None)
        kick=kickoff_motion(d,form) if form else None
        collective=collective_return(d,form,ft) if form else {'known':False,'reason':'NO_FORMATION'}
        # Sin señal independiente de gol (cruce, tag), formación sola no prueba gol.
        confidence='HIGH_CONTEXTUAL' if (crosses or tags) and form and kick else ('MEDIUM' if (crosses or tags) and form else ('LOW_CANDIDATE' if crosses or tags else 'UNVERIFIED_HYPOTHESIS'))
        if tags and not crosses and not form:confidence='PROVIDER_ONLY_UNVERIFIED'
        start_bracket={'lower_frame':min(c['frame_before'] for c in crosses),'upper_frame':max(c['frame_after'] for c in crosses),'basis':'OBSERVED_2D_GOAL_ZONE'} if crosses else None
        # Si no hay cruce observado, el proveedor da contexto, NO instante exacto de la señal arbitral.
        restart_bracket=kick
        results.append(dict(type='GOAL',anchor_frame=frame,period=period(frame),
            confidence=confidence,side=side,provider_goals=tags,
            ball_goal_zone=crosses,formation=form,collective_return=collective,
            kickoff_bracket=restart_bracket,
            dead_start_bracket=start_bracket,dead_end_bracket=restart_bracket,
            inferred_dead_start_frame=None if start_bracket is None else start_bracket['upper_frame'],
            validated_goal=False, # imposible certificar 3D y decisión arbitral desde tracking 2D
            reasons=[z for z,flag in [('goal_line_2d',bool(crosses)),('provider_goal',bool(tags)),('kickoff_formation',bool(form)),('kickoff_ball',bool(kick)),('collective_return',bool(collective.get('both_teams_returned')))] if flag],
            caveats=['2D no comprueba altura ni totalidad del balón','Smart Tagging no es GT','sin señal arbitral no se confirma gol legal']))
    return results,dict(ball_crossings=len(crossings),kickoff_formations=len(forms),providers=len(providers))

def synthetic_test():
    f=np.arange(0,500,5);n=len(f)
    x=np.tile(np.r_[np.full(24,20.),np.full(22,85.)],(n,1)).astype(float)
    y=np.tile(np.linspace(10,58,46),(n,1)).astype(float)
    d=dict(frames=f,x=x,y=y,ok=np.ones((n,46),bool),bx=np.full(n,55.),by=np.full(n,34.),obs=np.zeros(n,bool),periods=np.array(['FIRST_HALF']*n))
    assert not ball_goal_evidence(d)
    assert not discover(d,[])[0]
    d['bx'][9]=2.;d['bx'][10]=.15;d['obs'][9:11]=True
    assert len(ball_goal_evidence(d))>=1
    assert not any(v['validated_goal'] for v in discover(d,[])[0])
    # Balón fuera de boca de gol nunca produce cruce compatible.
    d['by'][9:11]=55.
    assert not ball_goal_evidence(d)
    print('TH08 synthetic tests: OK')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',default='local_outputs/runs');ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:synthetic_test();return
    data=load_data(args.step);events=provider_goals();detections,stats=discover(data,events)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    path=out/f'th08_goal_step{args.step}.json'
    payload=dict(detector='TH08_GOAL_V1',config=asdict(P),fps=FPS,
                 warning='DIAGNOSTICO, NO CLASIFICACION H/A/D NI CONFIRMACION ARBITRAL',
                 stats=stats,candidates=detections)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
    print(f'TH08: {len(detections)} candidatos -> {path}')
    print('Señales:',stats)

if __name__=='__main__':main()
