"""TH06 KICK_OFF — detector diagnóstico geométrico-temporal, SIN escribir H/A/D.

Instalar en la raíz del proyecto (junto a config.py y starter/).
python th06_kick_off.py --step 5 --out local_outputs/runs

Exige seguimiento de ambos equipos (slots ATD, NO identidades persistentes).
El balón imputado jamás demuestra golpeo. Smart Tagging NO es GT.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import datetime
import argparse,json
import numpy as np
import pandas as pd
import config

FPS=int(config.FPS)
L,W=105.3,68.0  # CONFIGURACION DE PROYECTO, NO CERTIFICACION DEL CAMPO
CX,CY=L/2,W/2

@dataclass(frozen=True)
class Params:
    half_tolerance_m:float=1.5
    circle_radius_m:float=9.15
    circle_tolerance_m:float=1.3
    centre_taker_radius_m:float=3.8
    centre_ball_radius_m:float=2.0
    min_team_visible:int=8
    own_half_fraction:float=.75
    outside_circle_fraction:float=.80
    opponent_inside_circle_max:int=2
    low_motion_mps:float=2.0
    stable_window_s:float=3.0
    min_stable_snapshots:int=4
    min_stable_ratio:float=.55
    transition_window_s:float=9.0
    formation_lookback_s:float=25.0
    goal_lookback_s:float=90.0
    kick_search_s:float=13.0
    kick_gap_max_s:float=.8
    kick_min_move_m:float=.7
    kick_speed_min_mps:float=1.8
    post_restart_window_s:float=3.0
    post_activation_min_mps:float=1.6
    merge_anchor_s:float=15.0
P=Params()

def _series(df,k,default=np.nan):
    return pd.to_numeric(df[k],errors='coerce').to_numpy(float) if k in df else np.full(len(df),default)

def period(f):return config.get_period(int(f))

def load_data(step:int):
    if step<1:raise ValueError('step debe ser >=1')
    v4=pd.read_csv(Path(config.OUTPUT_DIR)/'matrix_v4.csv',low_memory=False)
    n=len(v4)
    if n!=config.N_VIDEO_FRAMES or not np.array_equal(_series(v4,'frame'),np.arange(n)):
        raise ValueError('matrix_v4.csv no está alineado con frames 0..N-1')
    frames=np.arange(0,n,step,dtype=int)
    # Incluir explícitamente los bordes de periodo aunque step no los muestree.
    frames=np.unique(np.r_[frames,config.FIRST_HALF_START,config.SECOND_HALF_START])
    frames=frames[np.array([period(f) in ('FIRST_HALF','SECOND_HALF') for f in frames])]
    from starter.load_atd import load
    raw=load(game=config.GAME,source='tactical').to_df()
    if 'frame_id' not in raw:
        if raw.index.name=='frame_id':raw=raw.reset_index()
        else:raise ValueError('ATD sin frame_id')
    raw=raw.sort_values('frame_id').reset_index(drop=True)
    if len(raw)!=n or not np.array_equal(_series(raw,'frame_id'),np.arange(1,n+1)):
        raise ValueError('ATD sin alineación unívoca con vídeo')
    xx=[f'{j}_x' for j in range(1,47)]; yy=[f'{j}_y' for j in range(1,47)]
    if any(c not in raw for c in xx+yy):raise ValueError('ATD sin 46 pares x/y esperados')
    x=raw.loc[frames,xx].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*L
    y=raw.loc[frames,yy].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)*W
    ok=np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<=L)&(y>=0)&(y<=W)
    x[~ok]=np.nan;y[~ok]=np.nan
    bx=_series(v4,'ball_x_normalized')[frames]*L
    by=_series(v4,'ball_y_normalized')[frames]*W
    imputed=_series(v4,'ball_imputed',0)[frames]>.5
    observed=(_series(v4,'ball_observed',0)[frames]>.5)&(~imputed)&np.isfinite(bx)&np.isfinite(by)
    return dict(frames=frames,x=x,y=y,valid=ok,bx=bx,by=by,obs=observed,imp=imputed,
                period=np.array([period(f) for f in frames]),step=step)

def provider_context():
    """Etiquetas automáticas: GOALS y SET PIECES/Kick Off; no convierten un intervalo en D."""
    try:
        from starter.load_smart_tagging import load
        events=load(game=config.GAME)
    except (ImportError,OSError,KeyError,FileNotFoundError,ValueError,TypeError) as e:
        return [],'unavailable: '+str(e)
    out=[]
    for event in events:
        if not isinstance(event,dict):continue
        tags=event.get('tags') or {}
        label=str(tags.get('Type',event.get('type',''))).lower()
        code=str(event.get('code','')).lower()
        if 'kick off' in label or 'kick-off' in label or 'kickoff' in label:kind='PROVIDER_KICKOFF'
        elif 'goal' in code and 'kick' not in label:kind='PROVIDER_GOAL'
        else:continue
        try:
            a=int(event['start_frame']);b=int(event['end_frame'])
        except (ValueError,TypeError,KeyError):continue
        out.append(dict(kind=kind,start_frame=a,end_exclusive=b,team=str(tags.get('Team','')),
                        type=str(tags.get('Type','')),source='AUTO_SMART_TAGGING_NOT_GROUND_TRUTH'))
    return sorted(out,key=lambda z:z['start_frame']),'available'

def _past_snapshot(data,i,seconds):
    target=data['frames'][i]-round(seconds*FPS)
    j=np.searchsorted(data['frames'],target)
    if j>=i or data['period'][j]!=data['period'][i]:return None
    return int(j)

def occupancy_speed(d,seconds=1.0,cutoff_m=8.0):
    """Velocidad proxy: distancia a vecino anterior por equipo (NO ID de jugador)."""
    n=len(d['frames']);motion=np.full((n,2),np.nan)
    for i in range(n):
        j=_past_snapshot(d,i,seconds)
        if j is None:continue
        dt=(d['frames'][i]-d['frames'][j])/FPS
        for team,sl in enumerate((slice(0,24),slice(24,46))):
            a=np.column_stack((d['x'][i,sl],d['y'][i,sl]))
            b=np.column_stack((d['x'][j,sl],d['y'][j,sl]))
            a=a[np.isfinite(a).all(axis=1)];b=b[np.isfinite(b).all(axis=1)]
            if min(len(a),len(b))<P.min_team_visible or abs(len(a)-len(b))>2:continue
            distances=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
            nearest=np.min(distances,axis=1)
            accepted=nearest<cutoff_m
            if accepted.sum()>=P.min_team_visible:
                motion[i,team]=float(np.median(nearest[accepted])/dt)
    return motion

def geometric_features(d):
    """Evalúa ambos sentidos de ataque: equipo 0 a izquierda o a derecha.
    La orientación no se deduce de un solo x; se infiere en ventanas estables.
    """
    x,y,valid=d['x'],d['y'],d['valid'];n=len(x)
    inside=(np.hypot(x-CX,y-CY)<P.circle_radius_m+P.circle_tolerance_m)&valid
    center=np.hypot(x-CX,y-CY)<=P.centre_taker_radius_m
    visible=np.stack((valid[:,:24].sum(axis=1),valid[:,24:].sum(axis=1)),axis=1)
    in_circle=np.stack((inside[:,:24].sum(axis=1),inside[:,24:].sum(axis=1)),axis=1)
    near_center=np.stack(((center&valid)[:,:24].sum(axis=1),(center&valid)[:,24:].sum(axis=1)),axis=1)
    # 2 configuraciones posibles de lado propio: first team LEFT / first team RIGHT
    hypotheses=[]
    for team0_left in (True,False):
        own0=(x[:,:24]<=CX+P.half_tolerance_m) if team0_left else (x[:,:24]>=CX-P.half_tolerance_m)
        own1=(x[:,24:]>=CX-P.half_tolerance_m) if team0_left else (x[:,24:]<=CX+P.half_tolerance_m)
        proper=np.stack(((own0&valid[:,:24]).sum(axis=1),
                         (own1&valid[:,24:]).sum(axis=1)),axis=1)
        fraction=proper/np.maximum(visible,1)
        hypotheses.append(fraction)
    return dict(visible=visible,in_circle=in_circle,near_center=near_center,
                own_fraction=np.stack(hypotheses,axis=1))

def collective_transition(d,g,seconds=3.0):
    """Evolución de los 22 jugadores visibles sin suponer identidad en slots.
    Centroides, cambio de ocupación de mitades, retirada del círculo central.
    Desconocido si cobertura insuficiente o hay cambio de periodo.
    """
    n=len(d['frames'])
    centroid=np.full((n,2,2),np.nan)
    count=g['visible']
    for team,sl in enumerate((slice(0,24),slice(24,46))):
        good=count[:,team]>=P.min_team_visible
        centroid[good,team,0]=np.nansum(d['x'][good,sl],axis=1)/count[good,team]
        centroid[good,team,1]=np.nansum(d['y'][good,sl],axis=1)/count[good,team]
    centroid_delta=np.full_like(centroid,np.nan)
    half_gain=np.full((n,2,2),np.nan) # [obs,sideHypothesis,team]
    circle_exit=np.full((n,2),np.nan)
    for i in range(n):
        j=_past_snapshot(d,i,seconds)
        if j is None:continue
        if np.any(count[i]<P.min_team_visible) or np.any(count[j]<P.min_team_visible):continue
        if np.max(np.abs(count[i]-count[j]))>2:continue
        centroid_delta[i]=centroid[i]-centroid[j]
        half_gain[i]=g['own_fraction'][i]-g['own_fraction'][j]
        circle_exit[i]=g['in_circle'][j]-g['in_circle'][i]
    return centroid,centroid_delta,half_gain,circle_exit

def observed_kicks(d):
    """Golpeo al centro SOLO desde balón observado hasta balón observado.
    Devuelve par de frames [anterior, posterior] que acota el golpeo, no frame falso exacto.
    """
    events=[];indexes=np.flatnonzero(d['obs'])
    for j,i in zip(indexes[:-1],indexes[1:]):
        if d['period'][j]!=d['period'][i]:continue
        dt=(d['frames'][i]-d['frames'][j])/FPS
        if dt<=0 or dt>P.kick_gap_max_s:continue
        before=np.hypot(d['bx'][j]-CX,d['by'][j]-CY)
        after=np.hypot(d['bx'][i]-CX,d['by'][i]-CY)
        delta=np.hypot(d['bx'][i]-d['bx'][j],d['by'][i]-d['by'][j])
        if before<=P.centre_ball_radius_m and delta>=P.kick_min_move_m and delta/dt>=P.kick_speed_min_mps:
            events.append(dict(before=int(d['frames'][j]),after=int(d['frames'][i]),
                               distance_m=round(float(delta),3),speed_mps=round(float(delta/dt),3),
                               ball_to_center_m=round(float(before),3),post_distance_m=round(float(after),3)))
    return events

def _first_after(d,frame):return int(np.searchsorted(d['frames'],frame,side='left'))

def _support(events,frame,kind,lookback,lookahead):
    return [e for e in events if e['kind']==kind and e['start_frame']<=frame+lookahead*FPS
            and e['end_exclusive']>=frame-lookback*FPS]

def detect(d,providers=(),start_frames=None):
    """Candidatos de TH06; siempre devolver evidencias, incertidumbres y contradicciones.
    Configuraciones de centro NO identifican por sí mismas al equipo ejecutante.
    """
    n=len(d['frames']);g=geometric_features(d);motion=occupancy_speed(d)
    centroid,centroid_delta,half_gain,circle_exit=collective_transition(d,g)
    kickoff_ball=d['obs']&(np.hypot(d['bx']-CX,d['by']-CY)<=P.centre_ball_radius_m)
    motion_low=np.all(np.isfinite(motion),axis=1)&np.all(motion<=P.low_motion_mps,axis=1)
    stable=np.zeros((n,2,2),bool) # [snapshot, sideHypothesis, takerTeam]
    geometry=np.zeros_like(stable)
    outside=np.zeros_like(stable,float)
    for side in range(2):
        frac=g['own_fraction'][:,side,:]
        own=np.all((g['visible']>=P.min_team_visible)&(frac>=P.own_half_fraction),axis=1)
        for team in (0,1):
            other=1-team
            # 1-2 ejecutores (tolerar un tercero por oclusión/ruido); rivales fuera del círculo
            taker=(g['near_center'][:,team]>=1)&(g['near_center'][:,team]<=3)
            far=g['in_circle'][:,other]<=P.opponent_inside_circle_max
            outside[:,side,team]=1-g['in_circle'][:,other]/np.maximum(g['visible'][:,other],1)
            far &= outside[:,side,team]>=P.outside_circle_fraction
            geometry[:,side,team]=own&taker&far
            stable[:,side,team]=geometry[:,side,team]&motion_low
    # Persistencia real: snapshots independientes, no replicar 5 frames en 5 observaciones.
    form=np.zeros_like(stable)
    window=round(P.stable_window_s*FPS)
    for i,fr in enumerate(d['frames']):
        j=np.searchsorted(d['frames'],fr-window,side='left')
        for side in range(2):
            for team in (0,1):
                samples=stable[j:i+1,side,team]
                same=d['period'][j:i+1]==d['period'][i]
                denom=int(same.sum())
                form[i,side,team]=(int(np.sum(samples&same))>=P.min_stable_snapshots and
                                  np.sum(samples&same)/max(denom,1)>=P.min_stable_ratio)
    if start_frames is None:start_frames=[int(config.FIRST_HALF_START),int(config.SECOND_HALF_START)]
    anchors=sorted(set(int(z) for z in start_frames))
    kicks=observed_kicks(d)
    # Cada ejecución observada desde centro se evalúa respecto a formación ANTERIOR,
    # no se considera saque de centro sólo por la pelota.
    candidates=[]
    for kick in kicks:
        end=kick['after'];stop=_first_after(d,kick['before'])
        begin=_first_after(d,kick['before']-P.formation_lookback_s*FPS)
        preceding=np.flatnonzero(np.any(form[begin:stop+1],axis=(1,2)))+begin
        if len(preceding)==0:continue
        center_idx=int(preceding[-1]);side,team=np.argwhere(form[center_idx])[0]
        candidates.append(dict(anchor_idx=center_idx,side=int(side),team=int(team),kick=kick,origin='OBSERVED_KICK'))
    # Formaciones sin balón observado, como candidatas inciertas;
    # requieren ancla de inicio/descanso o GOAL previo o tag de saque contextual.
    for i in range(n):
        ss=np.argwhere(form[i]);
        if len(ss)==0:continue
        if i>0 and d['period'][i]==d['period'][i-1] and np.any(form[i-1]):continue
        fr=int(d['frames'][i]);period_anchor=any(0<=fr-a<=P.formation_lookback_s*FPS for a in anchors)
        previous_goal=bool(_support(providers,fr,'PROVIDER_GOAL',P.goal_lookback_s,0))
        provider_kick=bool(_support(providers,fr,'PROVIDER_KICKOFF',P.goal_lookback_s,12))
        if not (period_anchor or previous_goal or provider_kick):continue
        side,team=ss[0]
        candidates.append(dict(anchor_idx=i,side=int(side),team=int(team),kick=None,origin='GEOMETRY_CONTEXT_ONLY'))
    # Desduplicar dentro del mismo periodo: una fase estable + un golpeo = una sola reanudación.
    candidates.sort(key=lambda c:(d['period'][c['anchor_idx']],d['frames'][c['anchor_idx']],
                                  0 if c['kick'] else 1))
    groups=[]
    for c in candidates:
        anchor=int(d['frames'][c['anchor_idx']]);per=d['period'][c['anchor_idx']]
        existing=None
        for group in reversed(groups):
            if group['period']==per and abs(anchor-group['anchor_frame'])<=P.merge_anchor_s*FPS:
                existing=group;break
        if existing is not None:
            if c['kick'] and existing['_kick'] is None:
                existing['_kick']=c['kick'];existing['origin']='OBSERVED_KICK'
            continue
        groups.append(dict(period=per,anchor_frame=anchor,anchor_idx=c['anchor_idx'],
                           side=c['side'],team=c['team'],_kick=c['kick'],origin=c['origin']))
    episodes=[]
    for group in groups:
        idx=group['anchor_idx'];f=int(d['frames'][idx]);per=group['period']
        prior_goals=_support(providers,f,'PROVIDER_GOAL',P.goal_lookback_s,0)
        prior_kicks=_support(providers,f,'PROVIDER_KICKOFF',P.goal_lookback_s,12)
        first_anchor=int(config.FIRST_HALF_START) if per=='FIRST_HALF' else int(config.SECOND_HALF_START)
        start_type='PERIOD_START' if abs(f-first_anchor)<=P.formation_lookback_s*FPS else ('AFTER_GOAL' if prior_goals else 'UNDETERMINED')
        kick=group['_kick']
        # retroceso: desde la configuración hasta primera ruptura de juego, nunca restar N como verdad.
        prev=_first_after(d,f-P.transition_window_s*FPS)
        # como fragmento inicial plausible de la formación, con cobertura temporal propia
        valid_form=np.any(geometry[prev:idx+1],axis=(1,2))
        onset_idx=prev+int(np.flatnonzero(valid_form)[0]) if np.any(valid_form) else idx
        uncertain_start_low=max(first_anchor,int(d['frames'][onset_idx])-round(P.transition_window_s*FPS))
        uncertain_start_high=int(d['frames'][onset_idx])
        # Period start: D puede preceder primer frame puntuable; no declarar inicio de interrupción inexistente.
        if start_type=='PERIOD_START': uncertain_start_low=uncertain_start_high=None
        # Buscar reanudación posterior que rompa formación, SIN inventar golpeo.
        ai=_first_after(d,f+P.post_restart_window_s*FPS)
        ai=min(ai,n-1)
        post=motion[idx:ai+1]
        activated=bool(np.any(np.isfinite(post)&(post>=P.post_activation_min_mps)))
        end_low=kick['before'] if kick else None
        end_high=kick['after'] if kick else None
        # Que el movimiento posgolpeo no sea medible no invalida golpeo observado.
        contradictions=[]
        if g['visible'][idx].min()<P.min_team_visible:contradictions.append('LOW_TRACKING_COVERAGE')
        if not activated:contradictions.append('POST_RESTART_ACTIVITY_UNCONFIRMED')
        if not kick:contradictions.append('KICK_TIME_NOT_OBSERVED')
        if start_type=='UNDETERMINED' and not prior_kicks:contradictions.append('NO_GOAL_OR_PERIOD_ANCHOR')
        first_goal=max((ev['end_exclusive'] for ev in prior_goals if ev['end_exclusive']<=f),default=None)
        # Dos estados de evidencias no equivalentes: forma no implica reanudación observada.
        evidence=['BOTH_TEAMS_OWN_HALF','OPPONENTS_OUTSIDE_CENTRE_CIRCLE',
                  'CENTRE_TAKER_POSITION','COLLECTIVE_STILLNESS','PERSISTENT_FORMATION']
        if kick:evidence+=['OBSERVED_BALL_LEAVING_CENTRE','KICK_BRACKETED_BY_OBSERVATIONS']
        # Movimiento anterior sólo corrobora la película, no determina por sí solo el saque.
        start_window=max(0,_first_after(d,f-P.transition_window_s*FPS))
        has_half_return=bool(np.any(half_gain[start_window:idx+1,group['side'],:]>0.12))
        has_circle_vacating=bool(np.any(circle_exit[start_window:idx+1,:]>1))
        if has_half_return:evidence.append('DYNAMIC_RETURN_TO_OWN_HALVES')
        if has_circle_vacating:evidence.append('DYNAMIC_CENTRE_CIRCLE_VACATING')
        if prior_goals:evidence.append('GOAL_CONTEXT_AUTOMATIC')
        if prior_kicks:evidence.append('KICKOFF_CONTEXT_AUTOMATIC')
        if start_type=='PERIOD_START':evidence.append('PERIOD_ANCHOR_CONFIG')
        episodes.append(dict(type='KICK_OFF',period=per,subtype=start_type,
          anchor_frame=f,team_candidate=('BARCELONA' if group['team']==0 else 'ATHLETIC'),
          team_assignment='SPATIAL_HYPOTHESIS_NOT_PLAYER_ID',
          team0_own_side=('LEFT' if group['side']==0 else 'RIGHT'),
          center_x_m=CX,center_y_m=CY,goal_prior_end_frame=first_goal,
          d_start_low=uncertain_start_low,d_start_high=uncertain_start_high,
          d_start_method=('NOT_APPLICABLE_PERIOD_BEGINNING' if start_type=='PERIOD_START' else 'RETROSPECTIVE_GEOMETRIC_CHANGE_UNCERTAIN'),
          d_end_low=end_low,d_end_high=end_high,
          d_end_method='OBSERVED_BALL_MOVEMENT' if kick else 'UNOBSERVED_UNRESOLVED',
          evidence=';'.join(evidence),contradictions=';'.join(contradictions),
          tier='OBSERVED_RESTART_AND_FORMATION' if kick else 'FORMATION_PLUS_CONTEXT_NO_EXACT_END',
          initial_setup_frame=int(d['frames'][onset_idx]),post_collective_activation=activated,
          dynamic_half_return=has_half_return,dynamic_circle_vacating=has_circle_vacating,
          observed_ball_at_centre=bool(np.any(kickoff_ball[max(0,idx-15):idx+1])),
          source='TH06_GEOMETRY_TEMPORAL',score_is_calibrated=False))
    features=dict(visible_team0=g['visible'][:,0],visible_team1=g['visible'][:,1],
       inside_circle_team0=g['in_circle'][:,0],inside_circle_team1=g['in_circle'][:,1],
       center_taker_team0=g['near_center'][:,0],center_taker_team1=g['near_center'][:,1],
       own_half_left_team0=g['own_fraction'][:,0,0],own_half_left_team1=g['own_fraction'][:,0,1],
       own_half_right_team0=g['own_fraction'][:,1,0],own_half_right_team1=g['own_fraction'][:,1,1],
       motion_team0=motion[:,0],motion_team1=motion[:,1],still=motion_low.astype(int),
       centroid_x_team0=centroid[:,0,0],centroid_x_team1=centroid[:,1,0],
       centroid_x_delta_team0=centroid_delta[:,0,0],centroid_x_delta_team1=centroid_delta[:,1,0],
       half_gain_left_team0=half_gain[:,0,0],half_gain_left_team1=half_gain[:,0,1],
       half_gain_right_team0=half_gain[:,1,0],half_gain_right_team1=half_gain[:,1,1],
       circle_exit_team0=circle_exit[:,0],circle_exit_team1=circle_exit[:,1],
       center_ball_observed=kickoff_ball.astype(int),geometry_any=np.any(geometry,axis=(1,2)).astype(int),
       persistent_formation=np.any(form,axis=(1,2)).astype(int))
    return episodes,features,dict(params=asdict(P),anchor_frames=anchors,provider_not_gt=True,
        pitch_assumed_m=[L,W],not_HAD_output=True,slot_ids_not_persistent=True,
        goal_signal_not_mandatory_for_observed_restart=True,observed_kick_count=len(kicks))

def save(d,episodes,features,providers,meta,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');prefix=outdir/('th06_kick_off_'+stamp)
    paths={s:Path(str(prefix)+'_'+s+'.'+('json' if s=='summary' else 'csv'))
           for s in ('features','episodes','provider','summary')}
    df=pd.DataFrame({'frame':d['frames'],'period':d['period'],'ball_observed':d['obs'].astype(int),
                     'ball_imputed':d['imp'].astype(int)})
    for k,v in features.items():
        if len(v)!=len(df):raise ValueError(k+' shape error')
        df[k]=v
    df.to_csv(paths['features'],index=False)
    pd.DataFrame(episodes).to_csv(paths['episodes'],index=False)
    pd.DataFrame(providers).to_csv(paths['provider'],index=False)
    report={**meta,'episodes':len(episodes),'frames_sampled':len(df),'outputs':{k:str(v) for k,v in paths.items()},
            'warning':'DIAGNOSTICO, NO modifica H/A/D ni run.py; no precision validada'}
    paths['summary'].write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--step',type=int,default=5)
    cli.add_argument('--out',default=str(config.RUNS_DIR))
    opts=cli.parse_args()
    d=load_data(opts.step);events,status=provider_context()
    episodes,features,meta=detect(d,events)
    save(d,episodes,features,events,{**meta,'provider_status':status},opts.out)

if __name__=='__main__':main()
