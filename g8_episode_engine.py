"""G8: seguidor causal de episodios D; SOLO evidencias V4/S27/TH/S31, nunca etiquetas.

Modulo interno de run.py, no es un postprocesador de submissions.
Los scores S31 son compatibilidades heuristicas, NO probabilidades calibradas.
La continuidad permite breves ausencias de evidencia, pero salida confirmada
siempre tiene precedencia. Todas las ventanas se reinician por mitad.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def intervals(mask):
    z = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(z == 1), np.flatnonzero(z == -1)-1))


def rolling_segment(values, relevant, frames, *, center=False):
    out=np.zeros(len(relevant),dtype=np.float32)
    for a,b in intervals(relevant):
        x=np.asarray(values[a:b+1],dtype=float)
        out[a:b+1]=(pd.Series(x).rolling(max(1,frames),center=center,min_periods=1)
                    .mean().fillna(0).to_numpy(dtype=np.float32))
    return out


def verified_restart(evidence, relevant, frames, threshold=.75):
    """La reanudacion exige circulacion observada + dinamica de equipos.

    No usa exclusivamente la velocidad del balon ni el fin esperado de TH.
    """
    n=len(relevant)
    obs=np.asarray(evidence['observed'],bool)
    flow=np.asarray(evidence['observed_flow'],bool)
    live=np.asarray(evidence['collective_live'],bool)
    physical=(obs & flow & live & np.asarray(relevant,bool))
    return rolling_segment(physical,relevant,frames)>=threshold


def episode_evidence(event_frames, evidence, relevant, th_masks=None, *,
                     types=('FREE_KICK','GOAL','COOLING_BREAK','UNKNOWN_INTERRUPTION'),
                     threshold=.58, seed_seconds=1.2, support=.40,
                     patience_seconds=8., max_episode_seconds=120.,
                     restart_seconds=.7, fps=25., gmm_probability=None,
                     gmm_weight=0., s31_enabled=True, th_enabled=True,
                     hold_factor=.70, entry_fraction=.68, physical_window_seconds=.8,
                     restart_threshold=.75, th_near_seconds=8., minimum_seed_seconds=.6):
    """Genera evidencia/episodios D ANTES del RAM de posesion.

    Una nueva D necesita simultaneamente causa contextual y comportamiento
    colectivo/no-circulacion. El episodio se mantiene durante breves huecos
    hasta reanudacion fisica corroborada o agotamiento de soporte.
    """
    relevant=np.asarray(relevant,bool)
    n=len(relevant)
    if not (len(event_frames)==n and len(evidence['dead_confirmed'])==n):
        raise ValueError('S31/TH/V4: indices de frame incompatibles')
    frames=event_frames
    if not np.array_equal(np.asarray(frames['frame']),np.arange(n)):
        raise ValueError('S31: frames no alineados')
    prev_dead=np.asarray(evidence['dead_confirmed'],bool)
    obs=np.asarray(evidence['observed'],bool)
    idle=np.clip(pd.to_numeric(frames['event_team_idle'],errors='coerce').fillna(0).to_numpy(float),0,1)
    regroup=np.clip(pd.to_numeric(frames['event_regrouping'],errors='coerce').fillna(0).to_numpy(float),0,1)
    slowdown=np.clip(pd.to_numeric(frames['event_team_slowdown'],errors='coerce').fillna(0).to_numpy(float),0,1)
    physical=.50*idle+.30*regroup+.20*slowdown
    if gmm_probability is not None and gmm_weight:
        gmm=np.nan_to_num(np.asarray(gmm_probability,dtype=float),nan=0.)
        physical=(1.-gmm_weight)*physical+gmm_weight*np.clip(gmm,0,1)
    phys_smooth=rolling_segment(physical,relevant,max(1,round(physical_window_seconds*fps)),center=True)
    shapes=(np.asarray(evidence['spatial_setup'],bool)|np.asarray(evidence['zone_reorganization'],bool)
            |np.asarray(evidence['collective_stop'],bool))
    ball_idle=obs & np.asarray(evidence['low_ball'],bool)
    # Las familias de evidencia deben ser distintas: regimen colectivo y
    # configuracion espacial / balon observado / TH. La falta de tracking
    # no se transforma en prueba de detencion.
    physical_independent=shapes | ball_idle
    th_any=np.zeros(n,dtype=bool)
    th_type_map={'THROW_IN':'TH01','CORNER':'TH02','GOAL_KICK':'TH03',
        'FREE_KICK':'TH04','PENALTY':'TH05','KICK_OFF':'TH06',
        'OFFSIDE':'TH07','GOAL':'TH08','CARD_INTERVENTION':'TH09',
        'VAR_REVIEW':'TH10','INJURY':'TH11','COOLING_BREAK':'TH12',
        'SUBSTITUTION':'TH13','DROPPED_BALL':'TH14',
        'UNKNOWN_INTERRUPTION':'TH15'}
    if th_enabled and th_masks:
        for typ in types:
            th=th_masks.get(th_type_map.get(typ,''))
            if th is not None:th_any |= np.asarray(th,bool)
    th_near=rolling_segment(th_any.astype(float),relevant,max(1,round(th_near_seconds*fps)),center=True)>0
    context_score=np.zeros(n,dtype=float)
    event_types=np.asarray(frames['event_primary_type']).astype(str)
    for t in types:
        col='event_score_'+t.lower()
        if col in frames.columns:
            context_score=np.maximum(context_score,
                    np.nan_to_num(pd.to_numeric(frames[col],errors='coerce').to_numpy(float),nan=0.))
    n_families=pd.to_numeric(frames['event_distinct_evidence_families'],errors='coerce').fillna(0).to_numpy()
    # S31 debe estar realmente habilitado para abrir episodio sin TH primario.
    s31_context=s31_enabled & (context_score>=threshold)
    corroborated=(phys_smooth>=support) & (physical_independent | th_near) & (
        (s31_context & (n_families>=2)) | (th_enabled & th_near & (not s31_enabled or context_score>=threshold*.85)))
    # Veto robusto de reinicio, no un frame aislado de juego rapido.
    restart=verified_restart(evidence,relevant,max(1,round(restart_seconds*fps)),threshold=restart_threshold)
    corroborated &= relevant & ~restart
    # Una vez abierta una pausa, la continuidad es una regla DISTINTA
    # del inicio: permite oscilaciones de S31/TH si sigue habiendo
    # actividad colectiva reducida y no se ha confirmado reanudacion.
    sustain=(phys_smooth>=max(.23,support*hold_factor)) & relevant & ~restart
    # En cambio, con flujo observado sostenido NO se puede mantener D.
    sustain &= ~(np.asarray(evidence['observed_flow'],bool)
                  & np.asarray(evidence['collective_live'],bool))
    # En una pausa larga, una evidencia positiva reciente permite sobrevivir
    # hasta patience_seconds aun cuando la estimacion oscile.
    seed_frames=max(1,round(seed_seconds*fps))
    seed_roll=rolling_segment(corroborated.astype(float),relevant,seed_frames)
    seed=corroborated & (seed_roll>=entry_fraction)
    # No crear episodios cortos sin trayectoria sostenida.
    min_frames=min(seed_frames,max(1,int(round(minimum_seed_seconds*fps))))
    patience=max(1,int(round(patience_seconds*fps)))
    max_len=max(1,int(round(max_episode_seconds*fps)))
    accepted=np.zeros(n,bool)
    ep_type=np.full(n,'NONE',dtype=object)
    reasons=np.full(n,'NONE',dtype=object)
    total_episodes=0;blocked_by_restart=0
    for a,b in intervals(relevant):
        i=a
        while i<=b:
            if not seed[i] or prev_dead[i]:
                i+=1;continue
            # Require corroborated evidence for >= 0.6 s within seed region.
            start=i
            look=min(b+1,i+seed_frames)
            if int(np.sum(corroborated[i:look]))<min_frames:
                i+=1;continue
            last_support=i
            j=i;end=i-1
            while j<=b and j-start<max_len:
                if restart[j]:
                    blocked_by_restart+=1;break
                if sustain[j] or corroborated[j]:last_support=j
                if j-last_support>patience:break
                end=j;j+=1
            if end>=start and end-start+1>=min_frames:
                # No volver a escribir D anterior; solo incorporar lo nuevo.
                new = ~prev_dead[start:end+1]
                accepted[start:end+1]=new
                if new.any():
                    tscore={k:float(np.nanmean(pd.to_numeric(frames['event_score_'+k.lower()].iloc[start:end+1],errors='coerce').fillna(0)))
                            for k in types if 'event_score_'+k.lower() in frames.columns}
                    primary=max(tscore,key=tscore.get) if tscore else 'UNKNOWN_INTERRUPTION'
                    ep_type[start:end+1]=primary
                    reasons[start:end+1]='G8_EPISODE_TH_S31_PHYSICS'
                    total_episodes+=1
                i=end+1
            else:i+=1
    result=prev_dead|accepted
    result[~relevant]=False
    added=accepted&~prev_dead&relevant
    episode_state=np.full(n,'NONE',dtype=object)
    episode_state[corroborated]='STOP_PENDING'
    episode_state[added]='STOP_CONFIRMED'
    episode_state[restart]='LIVE_CONFIRMED'
    stats={'enabled':True,'new_D_frames':int(added.sum()),'new_episodes':total_episodes,
           'S31_context_frames':int((s31_context&relevant).sum()),
           'physics_context_frames':int((corroborated&relevant).sum()),
           'seed_frames':int((seed&relevant).sum()),
           'restart_verified_frames':int((restart&relevant).sum()),
           'sustain_frames':int((sustain&relevant).sum()),
           'restart_stops':blocked_by_restart,
           'new_frames_per_type':{k:int(np.sum(added&(ep_type==k))) for k in types},
           'mean_added_episode_frames':round(float(added.sum())/max(1,total_episodes),2)}
    return result, {'added':added,'seed':seed,'restart':restart,
                    'score':phys_smooth,'context_score':context_score,
                    'event_type':ep_type,'reason':reasons,'corroborated':corroborated,
                    'episode_state':episode_state},stats
