#!/usr/bin/env python3
"""Orquestador exclusivo de TH01–TH15. No ejecuta run.py ni altera H/A/D.

python run_con_th.py --project C:\\ruta\\proyecto --step 5
python run_con_th.py --project . --skip-run --step 5
python run_con_th.py --self-test

Las salidas TH heterogéneas se convierten a una tabla normalizada de hipótesis;
NINGUNA frontera incierta se transforma automáticamente en frames D.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import subprocess
import sys

NAMES = {
 'TH01':'th01_throw_in.py','TH02':'th02_corner.py','TH03':'th03_goal_kick.py',
 'TH04':'th04_free_kick.py','TH05':'th05_penalty.py','TH06':'th06_kick_off.py',
 'TH07':'th07_offside.py','TH08':'th08_goal.py','TH09':'th09_card_intervention.py',
 'TH10':'th10_var_review.py','TH11':'th11_injury.py',
 'TH12':'th12_cooling_break.py','TH13':'th13_substitution.py',
 'TH14':'th14_dropped_ball.py','TH15':'th15_unknown_interruption.py'
}
CSV_STEMS = {'TH01':'throw_in','TH02':'corner','TH03':'goal_kick',
             'TH04':'free_kick','TH05':'penalty','TH06':'th06_kick_off'}
FAMILY = {
 'TH01':'RESTART','TH02':'RESTART','TH03':'RESTART','TH04':'RESTART',
 'TH05':'RESTART','TH06':'RESTART','TH07':'CAUSE','TH08':'CAUSE',
 'TH09':'CONTEXT','TH10':'DELAY','TH11':'CONTEXT','TH12':'DELAY',
 'TH13':'CONTEXT','TH14':'RESTART','TH15':'RESIDUAL'
}
KEYS=('detector','family','source','event_type','tier','anchor_frame','start_low','start_high',
      'end_low','end_high','event_first','event_last','evidence','confidence_is_calibrated',
      'boundaries_are_confirmed','note')


def integer(v):
 if v is None or isinstance(v,bool): return None
 try:
  q=float(v)
  return int(q) if math.isfinite(q) else None
 except (TypeError,ValueError): return None


def bracket(obj):
 if isinstance(obj, dict):
  return integer(obj.get('lower_frame')), integer(obj.get('upper_frame'))
 if isinstance(obj, (tuple,list)) and len(obj)>=2:
  return integer(obj[0]),integer(obj[1])
 return None,None


def normalize_event(code, event, source):
 if not isinstance(event,dict): return None
 sl,sh=integer(event.get('d_start_low')),integer(event.get('d_start_high'))
 el,eh=integer(event.get('d_end_low')),integer(event.get('d_end_high'))
 if sl is None and sh is None:
  sl,sh=bracket(event.get('dead_start_bracket'))
 if el is None and eh is None:
  el,eh=bracket(event.get('dead_end_bracket'))
 if sl is None and sh is None:
  sl,sh=bracket(event.get('d_start_interval') or event.get('possible_D_start_window') or event.get('dead_start_interval'))
 if el is None and eh is None:
  el,eh=bracket(event.get('d_end_interval') or event.get('possible_D_end_window') or event.get('dead_end_interval'))
 if code=='TH08':
  sl,sh=bracket(event.get('dead_start_bracket'))
  el,eh=bracket(event.get('dead_end_bracket'))
 anchor=next((integer(event.get(k)) for k in ('anchor_frame','pass_frame_before','quiet_start_frame',
      'migration_start_frame','first_preparation_frame','exit_frame','start_frame') if integer(event.get(k)) is not None),None)
 first=next((integer(event.get(k)) for k in ('quiet_start_frame','migration_start_frame','first_preparation_frame','start_frame','exit_frame') if integer(event.get(k)) is not None),None)
 last=next((integer(event.get(k)) for k in ('quiet_end_frame','wait_end_frame','last_preparation_frame','end_frame','entry_frame') if integer(event.get(k)) is not None),None)
 if anchor is None:anchor=first if first is not None else sl if sl is not None else sh
 if first is None:first=sl if sl is not None else sh if sh is not None else anchor
 if last is None:last=eh if eh is not None else el if el is not None else anchor
 evtype=str(event.get('event_type') or event.get('type') or event.get('kind') or code)
 tier=str(event.get('tier') or event.get('confidence_tier') or event.get('candidate_strength') or event.get('classification') or event.get('confidence') or 'UNRANKED')
 evid=event.get('evidence','')
 if isinstance(evid,(dict,list,set)): evid=json.dumps(list(evid) if isinstance(evid,set) else evid,ensure_ascii=False,default=str)
 return dict(detector=code,family=FAMILY[code],source=str(source),event_type=evtype,tier=tier,
   anchor_frame=anchor,start_low=sl,start_high=sh,end_low=el,end_high=eh,
   event_first=first,event_last=last,evidence=str(evid),
   confidence_is_calibrated=False,boundaries_are_confirmed=False,
   note='EVIDENCIA_DIAGNOSTICA_NO_ETIQUETA_D')


def read_detector_outputs(code, out):
    """Normaliza resultados únicamente dentro de la carpeta aislada de la corrida."""
    if code in CSV_STEMS:
        files = sorted(out.glob(f'{CSV_STEMS[code]}*_episodes.csv'), key=lambda p:p.stat().st_mtime)
        if not files: return [], 'No se generó CSV de episodios'
        source = files[-1]
        with source.open(encoding='utf-8-sig',newline='') as fh:
            rows = [r for e in csv.DictReader(fh) if (r:=normalize_event(code,e,source)) is not None]
        return rows,None
    files = sorted(out.glob(f'{code.lower()}*.json'),key=lambda p:p.stat().st_mtime)
    if not files:return [],'No se generó JSON de diagnóstico'
    source=files[-1]
    obj=json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(obj,dict):return [],'El JSON no es un objeto'
    items=next((obj[k] for k in ('episodes','candidates','events') if isinstance(obj.get(k),list)),None)
    if items is None:return [],'Falta lista episodes/candidates/events en JSON'
    return [r for e in items if (r:=normalize_event(code,e,source)) is not None],None

def conflicts(rows,window=75):
 # Vecindad temporal de hipótesis, NO arbitraje automático por puntuaciones incompatibles.
 ordered=sorted((r for r in rows if r['anchor_frame'] is not None),key=lambda r:r['anchor_frame'])
 nearby=[]
 for i,a in enumerate(ordered):
  for b in ordered[i+1:]:
   delta=b['anchor_frame']-a['anchor_frame']
   if delta>window:break
   if a['detector']!=b['detector']:
    nearby.append({'frame_a':a['anchor_frame'],'frame_b':b['anchor_frame'],
      'detector_a':a['detector'],'detector_b':b['detector'],
      'family_a':a['family'],'family_b':b['family'],
      'resolution':'NOT_ARBITRATED'})
 return nearby


def write_csv(path,rows,keys):
 with path.open('w',encoding='utf-8',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');wr.writeheader();wr.writerows(rows)


def self_test():
    a=normalize_event('TH09',{'quiet_start_frame':130,'dead_start_bracket':{'lower_frame':120,'upper_frame':130},'kind':'INTERVENTION'},'test.json')
    assert (a['start_low'],a['anchor_frame'],a['family'])==(120,130,'CONTEXT')
    b=normalize_event('TH01',{'anchor_frame':150,'d_start_low':135,'d_end_high':180},'test.csv')
    assert len(conflicts([a,b],75))==1
    assert len(NAMES)==15
    print('SELF TEST OK: 15 detectores, TH09, normalización y conflictos')

def execute(argv,cwd,log):
 with log.open('w',encoding='utf-8') as out:
  env=os.environ.copy()
  env['PYTHONPATH']=str(cwd)+os.pathsep+env.get('PYTHONPATH','')
  r=subprocess.run(argv,cwd=str(cwd),env=env,stdout=out,stderr=subprocess.STDOUT,check=False)
 return r.returncode


def discover(project, explicit):
    options = ([explicit] if explicit else []) + [project/'juego_parado',project/'detectores',project/'TH',project]
    options=[Path(p).resolve() for p in options if p is not None]
    for p in options:
        if p.is_dir() and all((p/name).is_file() for name in NAMES.values()):return p
    found=[]
    for p in options:
        if p.is_dir():
            n=sum((p/name).is_file() for name in NAMES.values())
            found.append(f'{p}: {n}/15')
    raise FileNotFoundError('No se encontraron los 15 scripts en una carpeta. Revisadas: '+', '.join(found))


def main():
    ap=argparse.ArgumentParser(description='TH01–TH15 solamente; NO ejecuta ni modifica run.py ni H/A/D')
    ap.add_argument('--project',type=Path,default=Path('.'),help='Carpeta con config.py y los datos del proyecto')
    ap.add_argument('--th-dir',type=Path,default=None,help='Carpeta que contiene los 15 scripts (opcional si está en juego_parado)')
    ap.add_argument('--step',type=int,default=5)
    ap.add_argument('--out',type=Path,default=None,help='Carpeta raíz de resultados, por defecto local_outputs/runs')
    ap.add_argument('--only-normalize',type=Path,default=None,help='No ejecutar detectores; normalizar una carpeta de salida TH existente')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--fail-fast',action='store_true')
    args=ap.parse_args()
    if args.self_test:self_test();return
    if args.step<1:ap.error('--step debe ser mayor que cero')
    project=args.project.resolve()
    if not (project/'config.py').is_file():ap.error('No existe config.py dentro de --project')
    if args.only_normalize:
        evidence=args.only_normalize.resolve()
        if not evidence.is_dir():ap.error('No existe --only-normalize')
        thdir=None
    else:
        try:thdir=discover(project,args.th_dir)
        except FileNotFoundError as e:ap.error(str(e))
        timestamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        root=(args.out if args.out is not None else project/'local_outputs'/'runs').resolve()
        evidence=root/f'th_only_{timestamp}'
        evidence.mkdir(parents=True,exist_ok=False)
    status={}
    if thdir is not None:
        print(f'Proyecto: {project}\nScripts TH: {thdir}\nResultados: {evidence}',flush=True)
        # TH15 siempre al final, tras tener evidencia de los anteriores.
        for code,filename in NAMES.items():
            command=[sys.executable,str(thdir/filename),'--step',str(args.step),'--out',str(evidence)]
            if code=='TH15': command+=['--evidence-dir',str(evidence)]
            log=evidence/f'log_{code.lower()}.txt'
            rc=execute(command,project,log)
            status[code]={'status':'OK' if rc==0 else 'FAILED','returncode':rc,'log':str(log),'command':command}
            print(f'{code}: '+('OK' if rc==0 else f'ERROR {rc} -> {log}'),flush=True)
            if rc and args.fail_fast:break
    records=[];issues=[];counts={}
    for code in NAMES:
        if code in status and status[code]['returncode']!=0:
            counts[code]=0
            issues.append({'detector':code,'issue':'Falló la ejecución; consultar su log'})
            continue
        try:
            rows,err=read_detector_outputs(code,evidence)
            counts[code]=len(rows)
            if err:issues.append({'detector':code,'issue':err})
            records.extend(rows)
        except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:
            counts[code]=0
            issues.append({'detector':code,'issue':f'{type(e).__name__}: {e}'})
    write_csv(evidence/'evidencia_th_normalizada.csv',records,KEYS)
    write_csv(evidence/'solapamientos_th.csv',conflicts(records),
              ('frame_a','frame_b','detector_a','detector_b','family_a','family_b','resolution'))
    manifest={
      'mode':'TH_ONLY_NO_RUN_NO_HAD_CHANGES',
      'timestamp':dt.datetime.now().isoformat(),
      'project':str(project),'th_dir':str(thdir) if thdir else None,'step':args.step,
      'total_detectors':15,'detector_status':status,'episode_counts':counts,'total_episodes':len(records),
      'issues':issues,
      'outputs':{'normalized':str(evidence/'evidencia_th_normalizada.csv'),
                 'overlaps':str(evidence/'solapamientos_th.csv')},
      'interpretation':'Hipótesis diagnósticas, no etiquetas D ni certeza arbitral.',
      'integration_notes':['TH09 es un complemento disciplinario, no una reanudación.',
       'TH15 compara solo intervalos explícitos disponibles; no implica arbitraje completo.',
       'Puntuaciones entre TH no son directamente comparables.',
       'No se llama a run.py; ninguna submission ni H/A/D se modifica.',
       'El motor de arbitraje y la conexión con run.py serán un paso separado.']}
    (evidence/'resumen_th.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'carpeta':str(evidence),'episodios':len(records),'conteos':counts,'incidencias':issues},ensure_ascii=False,indent=2))
    if issues:sys.exit(2)

if __name__=='__main__':main()
