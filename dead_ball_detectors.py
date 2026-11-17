"""S31 — Hipótesis de 15 situaciones de interrupción, sin GT.

NO predice D directamente: reconoce el tipo candidato y los indicadores
que pueden alimentar el motor de actividad. Scores heurísticos [0,1],
NO probabilidades calibradas. Las clases sin evidencia identificadora
quedan INSUFFICIENT_EVIDENCE y no se falsifica una causa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

CATEGORIES = (
    "THROW_IN", "CORNER", "GOAL_KICK", "FREE_KICK", "PENALTY",
    "KICK_OFF", "OFFSIDE", "GOAL", "CARD_INTERVENTION",
    "VAR_REVIEW", "INJURY", "COOLING_BREAK", "SUBSTITUTION",
    "DROPPED_BALL", "UNKNOWN_INTERRUPTION",
)
FPS = 25


def num(df, name, fill=np.nan):
    if name not in df.columns:
        return np.full(len(df), fill, dtype=np.float64)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64, copy=True)


def clip01(x):
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0, neginf=0), 0, 1)


def rolling(x, width, minp=None):
    return (pd.Series(np.asarray(x, dtype=float))
            .rolling(width, min_periods=(minp if minp is not None else max(1,width//2)))
            .mean().fillna(0).to_numpy(dtype=float, copy=True))


def centred(x, width):
    return (pd.Series(np.asarray(x, dtype=float))
            .rolling(width, min_periods=max(1,width//3), center=True)
            .mean().fillna(0).to_numpy(dtype=float, copy=True))


def intervals(mask):
    a = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(a == 1), np.flatnonzero(a == -1)-1))


def bridge_gaps(mask, gap=8):
    out = np.asarray(mask, bool).copy()
    rr = intervals(out)
    for (_, e), (s2, _) in zip(rr, rr[1:]):
        if 0 < s2 - e - 1 <= gap:
            out[e+1:s2] = True
    return out


def _maybe_smart_events(game):
    """El V4 trae SET PIECES agregado, pero NO Type. Recuperar XML fuente."""
    try:
        from starter.load_smart_tagging import load
        events = load(game=game)
        if not isinstance(events, (list, tuple)):
            raise TypeError(f"Smart Tagging no es lista: {type(events)}")
        return list(events), "raw_type_available", ""
    except (ImportError, FileNotFoundError, OSError, ValueError, TypeError, KeyError) as exc:
        return [], "raw_type_unavailable", f"{type(exc).__name__}: {exc}"


def _raw_subtype_masks(n, game):
    events, status, detail = _maybe_smart_events(game)
    tags = {k: np.zeros(n, dtype=np.float32) for k in CATEGORIES}
    tag_records = []
    mapping = {
        "throw-in": "THROW_IN", "throw in": "THROW_IN",
        "corner kick": "CORNER", "goal kick": "GOAL_KICK",
        "free kick": "FREE_KICK", "kick off (after goal)": "KICK_OFF",
        "kick off (start)": "KICK_OFF", "kick-off": "KICK_OFF",
    }
    for evt in events:
        code = str(evt.get("code", "")).strip().upper()
        meta = evt.get("tags") or {}
        typ = str(meta.get("Type", "")).strip().lower()
        kind = mapping.get(typ)
        if code == "GOALS":
            kind = "GOAL"
        if kind is None:
            continue
        try:
            s = max(0, int(evt["start_frame"]))
            e = min(n, int(evt["end_frame"]))
        except (TypeError, ValueError, KeyError):
            continue
        if e <= s:
            continue
        # No interpretamos TODO el intervalo SmartTag como juego muerto.
        # El marcador confirma contexto de tipo, no la frontera de D.
        tags[kind][s:e] = 1.
        tag_records.append({"type": kind, "start_frame": s, "end_exclusive": e,
                            "side": meta.get("Side", ""), "team": meta.get("Team", ""),
                            "source": "smart_tagging_auto", "tag_type": meta.get("Type", "")})
    return tags, tag_records, status, detail


@dataclass
class DetectorOutput:
    frames: pd.DataFrame
    episodes: pd.DataFrame
    type_summary: pd.DataFrame
    sources: pd.DataFrame
    tag_records: pd.DataFrame
    status: dict


def detect(v4: pd.DataFrame, s27: pd.DataFrame, relevant: np.ndarray,
           *, game="BAR-ATH", fps=25, external_events=None) -> DetectorOutput:
    """Ejecuta las 15 hipótesis usando SOLO señales del proveedor/derivadas.

    Retorno: una fila/frame, una fila/episodio, conteos por categoría,
    inventario de señales y eventos Smart Tagging originales disponibles.
    NO se consultan GT ni salidas H/A/D para asignar contexto.
    """
    n = len(v4)
    relevant = np.asarray(relevant, dtype=bool)
    if len(s27) != n or len(relevant) != n:
        raise ValueError("V4/S27/máscara deben tener la misma cantidad de frames")
    if external_events is None:
        tag, records, source_status, source_detail = _raw_subtype_masks(n, game)
    else:
        tag = {k: np.zeros(n, dtype=np.float32) for k in CATEGORIES}
        records = []
        source_status, source_detail = "external_events", ""
        mapping = {"THROW-IN":"THROW_IN", "CORNER KICK":"CORNER", "GOAL KICK":"GOAL_KICK",
                   "FREE KICK":"FREE_KICK", "KICK OFF (AFTER GOAL)":"KICK_OFF",
                   "KICK OFF (START)":"KICK_OFF", "GOALS":"GOAL"}
        for ev in external_events:
            k = mapping.get(str(ev.get("type", ev.get("code", ""))).upper())
            if k is None: continue
            a = max(0, int(ev["start_frame"])); b = min(n, int(ev["end_frame"]))
            tag[k][a:b] = 1
            records.append({"type": k, "start_frame": a, "end_exclusive": b,
                            "side": ev.get("side", ""), "team": ev.get("team", ""),
                            "source":"injected_auto", "tag_type":ev.get("type", "")})
    obs = num(v4, "ball_observed",0) > .5
    imp = num(v4, "ball_imputed",0) > .5
    gap = num(v4, "ball_gap_length",0)
    q_ball = np.where(obs,1., np.where(imp,.7*np.exp(-np.maximum(gap,0)/100.),0.))
    q_ball = clip01(q_ball)
    x = num(v4, "ball_x_normalized"); y = num(v4, "ball_y_normalized")
    geom = np.isfinite(x) & np.isfinite(y) & (q_ball>.15)
    # Ratios normalizados 0..1; posible clipping en upstream: NO inferir cruce físico.
    lateral = geom & ((y<.045)|(y>.955))
    endline = geom & ((x<.045)|(x>.955))
    corner = lateral & endline
    interior = geom & ~lateral & ~endline
    near_center = geom & (abs(x-.5)<.075) & (abs(y-.5)<.16)
    penalty_spot = geom & (((abs(x-.11)<.05)|(abs(x-.89)<.05))&(abs(y-.5)<.095))
    in_box = geom & (((x<.19)|(x>.81)) & (abs(y-.5)<.28))
    edge_zone = geom & (lateral|endline)
    speed = num(v4, "ball_speed_mps")
    ball_speed_valid = np.isfinite(speed) & (speed>=0) & (speed<45) & (q_ball>.4)
    ball_slow = ball_speed_valid & (speed < 1.2)
    ball_fast = ball_speed_valid & (speed > 3.5)
    ball_motion = num(v4, "ball_displacement_m")
    moved = np.isfinite(ball_motion)&(ball_motion>.2)&(q_ball>.5)
    hd = num(s27,"home_distance_m"); ad=num(s27,"away_distance_m")
    comparable = num(s27,"geometry_comparable",0)>.5
    control_near = comparable & (np.minimum(hd,ad)<2.5)

    hs=num(v4,"Barcelona_speed_median"); aas=num(v4,"Atletic_Club_speed_median")
    hcov=num(v4,"Barcelona_speed_coverage",0); acov=num(v4,"Atletic_Club_speed_coverage",0)
    valid_team=np.isfinite(hs)&np.isfinite(aas)&(hcov>=.30)&(acov>=.30)
    low_both=valid_team&(hs<1.55)&(aas<1.55)
    active_teams=valid_team&((hs>1.8)|(aas>1.8))
    very_low=valid_team&(hs<1.05)&(aas<1.05)
    speed_avg = .5*(np.nan_to_num(hs,nan=0)+np.nan_to_num(aas,nan=0))
    slowed=valid_team & ((pd.Series(speed_avg).shift(25).rolling(25,min_periods=8).mean().to_numpy() - speed_avg)>.55)
    slowed=np.nan_to_num(slowed).astype(bool)
    idle=clip01(.55*rolling(low_both,25)+.45*rolling(very_low,50))
    slow_change=rolling(slowed,25)
    # Spatial team signals. A geometric zone != permanent player identity.
    cxh=num(v4,"Barcelona_centroid_x_m"); cxa=num(v4,"Atletic_Club_centroid_x_m")
    cyh=num(v4,"Barcelona_centroid_y_m"); cya=num(v4,"Atletic_Club_centroid_y_m")
    pcount_h=num(v4,"Barcelona_valid_position_count",0)
    pcount_a=num(v4,"Atletic_Club_valid_position_count",0)
    coverage = (pcount_h>=6)&(pcount_a>=6)&valid_team
    # Occupancy goal-side is approximation, not full 16.5m box.
    area_counts=np.zeros(n)
    lateral_counts=np.zeros(n)
    middle_counts=np.zeros(n)
    if all(f"{t}_zone_X1_Y2_count" in v4.columns for t in ("Barcelona","Atletic_Club")):
        for t in ("Barcelona","Atletic_Club"):
            for xx in (1,3):
                for yy in (1,2,3):
                    area_counts+=num(v4,f"{t}_zone_X{xx}_Y{yy}_count",0)
            for xx in (1,2,3):
                for yy in (1,3):
                    lateral_counts+=num(v4,f"{t}_zone_X{xx}_Y{yy}_count",0)
            for xx in (1,2,3):
                middle_counts+=num(v4,f"{t}_zone_X{xx}_Y2_count",0)
    nvisible=np.maximum(1,pcount_h+pcount_a)
    goal_concentration=clip01((area_counts/nvisible-.33)/.38)*coverage
    touchline_concentration=clip01((lateral_counts/nvisible-.40)/.40)*coverage
    central_concentration=clip01((middle_counts/nvisible-.20)/.55)*coverage
    # Team centroids near perimeter -> weak if only partial tracking.
    near_touchline_team = coverage & ((np.minimum(cyh,cya)<16)|(np.maximum(cyh,cya)>52))
    regroup=clip01(.6*rolling(slowed,50)+.4*rolling(low_both,50)) * coverage
    # Dynamic tactical change based on 3 longitudinal thirds.
    team_zone_change=np.zeros(n)
    zone_available=all(f"{t}_zone_X{i}_Y{j}_count" in v4.columns
                        for t in ("Barcelona","Atletic_Club") for i in (1,2,3) for j in (1,2,3))
    if zone_available:
        for t, count in (("Barcelona",pcount_h),("Atletic_Club",pcount_a)):
            for i in (1,2,3):
                z=sum(num(v4,f"{t}_zone_X{i}_Y{j}_count",0) for j in (1,2,3)) / np.maximum(count,6)
                old=pd.Series(z).shift(50).to_numpy(dtype=float)
                team_zone_change += np.nan_to_num(abs(z-old),nan=0)/4.
    reorganization=clip01(rolling(team_zone_change,25)/.20)*coverage
    # Causal condition of play, NOT a default ACTIVE label.
    flow = clip01(.55*rolling(ball_fast&control_near&obs,10)
                  + .45*rolling(active_teams&((moved&obs)|(q_ball<.3)),25))
    low_flow = 1.-flow
    tag_general=num(v4,"smart_tagging_code_set_pieces_count",0)>0
    tag_goal_general=num(v4,"smart_tagging_code_goals_count",0)>0
    tactical_build=num(v4,"smart_tagging_code_build_up_count",0)>0
    # Fundamental: smart phases can overlap ACTIVE; no automatic dead assignment.
    scores: Dict[str, np.ndarray] = {}
    family_count: Dict[str, np.ndarray] = {}
    def make(kind, geom_score, group_score, motion_score,
             *, tag_weight=.28, base_weights=(.32,.25,.15), max_cap=1.0):
        a,b,c=base_weights
        specific=clip01(tag[kind])
        gg=clip01(geom_score); pp=clip01(group_score); mm=clip01(motion_score)
        # Debias positive evidence when observed fluid play contradicts pause.
        raw=(tag_weight*specific + a*gg + b*pp + c*mm)
        raw -= .18*flow * (1.-specific*.5)
        scores[kind]=np.minimum(max_cap,clip01(raw)) * relevant
        family_count[kind]=(specific>.5).astype(np.int8)+(gg>.50).astype(np.int8)+(pp>.50).astype(np.int8)+(mm>.50).astype(np.int8)

    make("THROW_IN", lateral & ~endline, .62*touchline_concentration+.38*near_touchline_team,
         .55*idle+.45*regroup)
    make("CORNER", corner, goal_concentration, .55*idle+.45*regroup)
    make("GOAL_KICK", endline & ~corner, .60*goal_concentration+.4*(1.-touchline_concentration)*coverage,
         .5*idle+.5*regroup)
    make("FREE_KICK", interior | (geom&~corner), .65*regroup+.35*slow_change,
         .6*idle+.4*slow_change)
    make("PENALTY", penalty_spot, .6*goal_concentration+.4*central_concentration,
         .65*idle+.35*regroup, tag_weight=0.0,base_weights=(.42,.37,.21),max_cap=.85)
    make("KICK_OFF", near_center, .6*central_concentration+.4*regroup,
         .5*idle+.5*regroup)
    # Cause-level categories w/o whistle/referee: deliberately capped.
    # Higher score means compatibility, never a verified cause.
    make("OFFSIDE", (geom&((x<.25)|(x>.75))), .5*regroup+.5*slow_change,
         .7*idle+.3*slow_change, tag_weight=0.0,base_weights=(.20,.42,.38),max_cap=.64)
    make("GOAL", (geom&(((x<.03)|(x>.97))&(abs(y-.5)<.15))), .65*regroup+.35*idle,
         rolling(tag_goal_general,25),tag_weight=.27,base_weights=(.22,.23,.28),max_cap=.88)
    make("CARD_INTERVENTION", (interior|edge_zone)*.3,regroup,
         .55*idle+.45*slow_change,tag_weight=0.0,base_weights=(.10,.52,.38),max_cap=.64)
    make("VAR_REVIEW", np.zeros(n),rolling(very_low,250),
         .5*rolling(low_both,250)+.5*rolling(regroup,125),
         tag_weight=0.,base_weights=(0.,.58,.42),max_cap=.58)
    make("INJURY", np.zeros(n),.5*rolling(very_low,125)+.5*regroup,
         .6*idle+.4*slow_change,tag_weight=0.,base_weights=(0.,.52,.48),max_cap=.58)
    make("COOLING_BREAK", np.zeros(n), .5*rolling(very_low,250)+.5*touchline_concentration,
         .5*rolling(regroup,125)+.5*rolling(low_both,125),
         tag_weight=0.,base_weights=(0.,.57,.43),max_cap=.64)
    make("SUBSTITUTION", np.zeros(n), .5*touchline_concentration+.5*regroup,
         .5*idle+.5*slow_change,tag_weight=0.,base_weights=(0.,.60,.40),max_cap=.55)
    make("DROPPED_BALL", (interior&ball_slow)*.65, .5*regroup+.5*idle,
         .5*slow_change+.5*idle, tag_weight=0.,base_weights=(.27,.38,.35),max_cap=.64)
    # Unknown INTERRUPTION is a detection class, not a claimed cause.
    # Needs corroboration from team organization and activity loss.
    unknown=clip01(.42*idle+.32*regroup+.26*slow_change-.22*flow)*relevant
    scores["UNKNOWN_INTERRUPTION"]=unknown
    family_count["UNKNOWN_INTERRUPTION"]=(idle>.6).astype(np.int8)+(regroup>.5).astype(np.int8)+(slow_change>.45).astype(np.int8)
    types=list(CATEGORIES)
    stacked=np.stack([scores[k] for k in types],axis=1)
    idx=np.argmax(stacked,axis=1)
    top=stacked[np.arange(n),idx]
    ranked = np.sort(stacked,axis=1)
    runner=ranked[:,-2]
    # A scarce, visually ambiguous cause must stay tentative.
    evidence=np.stack([family_count[k] for k in types],axis=1)
    distinct=evidence[np.arange(n),idx]
    stage=np.full(n,"NO_CANDIDATE",dtype=object)
    stage[relevant & (top>=.40)]="POSSIBLE"
    corroborated=relevant&(top>=.73)&(distinct>=2)&((top-runner)>=.08)
    stage[corroborated]="CORROBORATED_CONTEXT"
    primary=np.array(types,dtype=object)[idx]
    primary[stage=="NO_CANDIDATE"]="NONE"
    # Confidence is a heuristic score, NOT P(event).
    candidate_context = relevant & (top>=.40)
    # Episode groups formed on strong candidates, with short-gap tolerance.
    # Never cross excluded game sections/half-time.
    epis=[]
    episode_id=np.full(n,-1,dtype=np.int32)
    episode_type=np.full(n,"NONE",dtype=object)
    active_streaks=intervals(relevant)
    for rs,re in active_streaks:
        mask=bridge_gaps(candidate_context[rs:re+1],gap=8)
        for local_s,local_e in intervals(mask):
            a,b=rs+local_s,rs+local_e
            if b-a+1<15: continue
            # Dominant type aggregated across episode, avoids per-frame flicker.
            vals=np.mean(stacked[a:b+1],axis=0)
            q=int(np.argmax(vals)); kind=types[q]
            mean=float(vals[q]); gap_to_next=float(vals[q]-np.partition(vals,-2)[-2])
            corroborated_frames=int(np.count_nonzero(stage[a:b+1]=="CORROBORATED_CONTEXT"))
            if corroborated_frames>=10 and mean>=.48: tier="CORROBORATED_CONTEXT"
            elif mean>=.40: tier="POSSIBLE"
            else: tier="AMBIGUOUS"
            eid=len(epis)+1
            episode_id[a:b+1]=eid
            episode_type[a:b+1]=kind
            epis.append({"episode_id":eid,"start_frame":a,"end_frame":b,
                "duration_s":round((b-a+1)/fps,2),"type":kind,"tier":tier,
                "type_score_mean":round(mean,4),"score_margin_mean":round(gap_to_next,4),
                "supported_frames":corroborated_frames,
                "observed_ball_fraction":round(float(np.mean(obs[a:b+1])),4),
                "imputed_ball_fraction":round(float(np.mean(imp[a:b+1])),4),
                "team_coverage_fraction":round(float(np.mean(coverage[a:b+1])),4),
                "observed_flow_mean":round(float(np.mean(flow[a:b+1])),4),
                "smart_tag_subtype_fraction":round(float(np.mean(tag[kind][a:b+1])),4),
                "proposed_stop_start": "NOT_DETERMINED",
                "proposed_restart_frame": "NOT_DETERMINED"})
    out=pd.DataFrame({"frame":np.arange(n,dtype=np.int32),
        "relevant":relevant.astype(np.int8),"event_primary_type":primary,
        "event_primary_score":top.round(4),"event_runnerup_score":runner.round(4),
        "event_distinct_evidence_families":distinct.astype(np.int8),
        "event_confidence_tier":stage,"event_episode_id":episode_id,
        "event_episode_type":episode_type,"event_raw_tag_subtype_any":
            np.maximum.reduce([tag[k] for k in types]).astype(np.int8),
        "event_observed_ball":obs.astype(np.int8),
        "event_ball_quality":np.round(q_ball,3),
        "event_collective_valid":coverage.astype(np.int8),
        "event_team_idle":np.round(idle,3),
        "event_regrouping":np.round(regroup,3),
        "event_team_slowdown":np.round(slow_change,3),
        "event_observed_flow":np.round(flow,3),
        "event_lateral_zone":lateral.astype(np.int8),
        "event_corner_zone":corner.astype(np.int8),
        "event_goal_area_proxy":np.round(goal_concentration,3),
        "event_lateral_concentration":np.round(touchline_concentration,3),
        "event_reorganization":np.round(reorganization,3)})
    for k in types:
        out["event_score_"+k.lower()]=scores[k].astype(np.float32)
    episodes=pd.DataFrame(epis,columns=["episode_id","start_frame","end_frame","duration_s",
       "type","tier","type_score_mean","score_margin_mean","supported_frames",
       "observed_ball_fraction","imputed_ball_fraction","team_coverage_fraction",
       "observed_flow_mean","smart_tag_subtype_fraction","proposed_stop_start","proposed_restart_frame"])
    summary=[]
    for k in types:
        ss=scores[k]
        summary.append({"type":k,"score_frames_ge_0_40":int(np.count_nonzero((ss>=.4)&relevant)),
            "score_frames_ge_0_73":int(np.count_nonzero((ss>=.73)&relevant)),
            "dominant_frames":int(np.count_nonzero((primary==k)&relevant)),
            "episode_count":int((episodes["type"]==k).sum()),
            "episode_seconds":round(float(episodes.loc[episodes["type"]==k,"duration_s"].sum()),2),
            "raw_smart_event_count":sum(1 for r in records if r["type"]==k),
            "identification_limit":("context_only_cannot_confirm_cause" if k in
                {"OFFSIDE","CARD_INTERVENTION","VAR_REVIEW","INJURY","COOLING_BREAK","SUBSTITUTION","DROPPED_BALL"}
                else "requires_restarting_boundary_confirmation")})
    summary=pd.DataFrame(summary)
    fields=["ball_observed","ball_imputed","ball_gap_length","ball_x_normalized","ball_y_normalized",
      "ball_speed_mps","ball_displacement_m","Barcelona_speed_median","Atletic_Club_speed_median",
      "Barcelona_speed_coverage","Atletic_Club_speed_coverage",
      "Barcelona_valid_position_count","Atletic_Club_valid_position_count",
      "smart_tagging_code_set_pieces_count", "smart_tagging_code_goals_count"]
    fields += [f"{t}_zone_X{x}_Y{y}_count" for t in ("Barcelona","Atletic_Club")
      for x in (1,2,3) for y in (1,2,3)]
    fields += ["home_distance_m","away_distance_m","geometry_comparable"]
    source_rows=[]
    for name in fields:
        df=s27 if name in ("home_distance_m","away_distance_m","geometry_comparable") else v4
        present=name in df.columns
        if present:
            v=num(df,name); valid=np.isfinite(v)&relevant
        else: valid=np.zeros(n,dtype=bool)
        source_rows.append({"signal":name,"available":int(present),
             "valid_game_frames":int(valid.sum()),"game_frames":int(relevant.sum())})
    source_rows.append({"signal":"RAW_SMART_TAGGING_Type", "available":int(source_status != "raw_type_unavailable"),
         "valid_game_frames":len(records),"game_frames":int(relevant.sum())})
    return DetectorOutput(out,episodes,summary,pd.DataFrame(source_rows),
                          pd.DataFrame(records,columns=["type","start_frame","end_exclusive","side","team","source","tag_type"]),
                          {"raw_smart_tagging_status":source_status,"raw_smart_tagging_detail":source_detail,
                           "n_episodes":len(epis),"n_raw_typed_events":len(records),
                           "uses_ground_truth":False,
                           "scores_are_probabilities":False,
                           "labels_changed":False})
