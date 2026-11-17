"""G9 FAST: prolongación y enlace de episodios D corroborados, antes de H/A.

Intervención OPCIONAL sobre la máscara de inactividad del RAM, NUNCA sobre el
CSV final. Nunca usa datos GT. El movimiento lento por sí solo no crea D.
Toda decisión necesita vecindad de una parada existente, contexto S31 compatible,
evidencia física suficiente y ausencia de circulación observada confirmada.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from g8_episode_engine import intervals, rolling_segment, verified_restart


def refine_dead(dead, relevant, frames, evidence, *, types=('COOLING_BREAK',),
                context_threshold=.44, physics_threshold=.35,
                max_edge_seconds=4., max_bridge_seconds=8.,
                patience_seconds=.6, restart_seconds=.6,
                fps=25.):
    dead=np.asarray(dead,bool).copy()
    rel=np.asarray(relevant,bool)
    n=len(dead)
    if not (len(frames)==len(rel)==len(evidence['observed'])):
        raise ValueError('G9: desalineación V4 / S31 / TH')
    start=dead.copy(); added=np.zeros(n,bool); edge=np.zeros(n,bool); bridge=np.zeros(n,bool)
    score=np.zeros(n,dtype=float)
    for typ in types:
        col='event_score_'+typ.lower()
        if col in frames:
            score=np.maximum(score,np.nan_to_num(pd.to_numeric(frames[col],errors='coerce').to_numpy(float),nan=0))
    idle=np.nan_to_num(pd.to_numeric(frames['event_team_idle'],errors='coerce').to_numpy(float),nan=0)
    regroup=np.nan_to_num(pd.to_numeric(frames['event_regrouping'],errors='coerce').to_numpy(float),nan=0)
    slowdown=np.nan_to_num(pd.to_numeric(frames['event_team_slowdown'],errors='coerce').to_numpy(float),nan=0)
    phys=rolling_segment(.50*idle+.30*regroup+.20*slowdown,rel,max(1,round(.8*fps)),center=True)
    independent=(np.asarray(evidence['spatial_setup'],bool)|np.asarray(evidence['zone_reorganization'],bool)
                 |np.asarray(evidence['collective_stop'],bool)
                 |(np.asarray(evidence['observed'],bool)&np.asarray(evidence['low_ball'],bool)))
    flow=(np.asarray(evidence['observed_flow'],bool)&np.asarray(evidence['collective_live'],bool))
    restart=verified_restart(evidence,rel,max(1,round(restart_seconds*fps)))
    eligible=rel & (score>=context_threshold) & (phys>=physics_threshold) & independent & ~flow & ~restart
    # Tolerar ausencias cortas de contexto, pero no un restart confirmado.
    pad=max(0,int(round(patience_seconds*fps)))
    max_edge=max(0,int(round(max_edge_seconds*fps)))
    max_bridge=max(0,int(round(max_bridge_seconds*fps)))
    for a,b in intervals(rel):
        old=[(x+a,y+a) for x,y in intervals(start[a:b+1])]
        # Prolongación de límites de D, deteniéndose ante reinicio físico.
        if max_edge:
            for left,right in old:
                for direction,at in ((-1,left-1),(1,right+1)):
                    if at<a or at>b:continue
                    misses=0;changes=[];j=at
                    while a<=j<=b and len(changes)<max_edge and not start[j]:
                        if restart[j] or flow[j]:break
                        if eligible[j]:misses=0
                        else:
                            misses+=1
                            if misses>pad:break
                        changes.append(j);j+=direction
                    # Solo aceptar si el tramo contiene evidencia positiva real.
                    if changes and any(eligible[k] for k in changes):
                        edge[changes]=True
        # Puentes entre dos D separados por intervalo corto con corroboración.
        if max_bridge:
            for (_,end),(begin,_) in zip(old,old[1:]):
                left=end+1;right=begin-1
                if right<left or right-left+1>max_bridge:continue
                part=slice(left,right+1)
                if np.any(restart[part]|flow[part]):continue
                if np.mean(eligible[part])>=.55 and np.max(score[part])>=context_threshold:
                    bridge[part]=True
    added=(edge|bridge)&~start&rel
    out=start|added
    out[~rel]=False
    return out, {'added':added,'edge':edge&added,'bridge':bridge&added,'restart':restart,
                 'context_score':score,'physical_score':phys}, {
                'enabled':True,'new_D_frames':int(added.sum()),
                'edge_frames':int(np.sum(edge&added)),
                'bridged_frames':int(np.sum(bridge&added)),
                'new_D_regions':len(intervals(added)),
                'verified_restart_frames':int(np.sum(restart&rel))}
