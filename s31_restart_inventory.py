"""S31-v3: diagnostico temporal de 15 eventos con memoria de jugadores.

No escribe submission ni altera run.py. Requiere config.py, matrix_v4.csv, s27.
Si existe starter.load_atd.load con datos tacticos, usa 46 slots de jugadores
por frame para configuraciones reglamentarias. Si no, comunica la ausencia.
"""
from datetime import datetime
import json
import argparse
import numpy as np
import pandas as pd
import config
from dead_ball_detectors import detect,load_individual_tracking


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--sin-tracking-individual',action='store_true',help='solo auditar fuentes, no fabricar configuraciones')
    args=pa.parse_args()
    rid=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print('='*65+'\nS31-v3 — CONFIGURACIONES Y SECUENCIAS TEMPORALES (NO GENERA D)\n'+'='*65)
    paths=[config.OUTPUT_DIR/'matrix_v4.csv',config.OUTPUT_DIR/'s27_football_signals.csv']
    v4=pd.read_csv(paths[0],low_memory=False)
    s27=pd.read_csv(paths[1],low_memory=False)
    n=config.N_VIDEO_FRAMES
    for name,df in [('V4',v4),('S27',s27)]:
        if len(df)!=n or 'frame' not in df or not np.array_equal(df['frame'].to_numpy(),np.arange(n)):
            raise ValueError(f'{name} no está alineado con 0..{n-1}')
    relevant=np.array([config.get_period(i) in ('FIRST_HALF','SECOND_HALF') for i in range(n)])
    print('[1] Datos V4/S27 alineados; frames de juego:',int(relevant.sum()))
    tracking=None;tracking_status='omitido_por_usuario'
    if not args.sin_tracking_individual:
        print('[2] Cargando tracking individual original mediante Kloppy...')
        tracking,tracking_status=load_individual_tracking(n,config.GAME)
    print('Tracking:',tracking_status)
    print('[3] Calculando configuraciones y episodios...')
    result=detect(v4,s27,relevant,game=config.GAME,fps=config.FPS,tracking=tracking)
    config.ensure_output_dirs()
    out=config.RUNS_DIR
    files={
      'event_context':out/f'event_context_temporal_{rid}.csv',
      'event_episodes':out/f'event_episodes_temporal_{rid}.csv',
      'event_type_summary':out/f'event_type_summary_temporal_{rid}.csv',
      'event_signal_coverage':out/f'event_signal_coverage_temporal_{rid}.csv',
      'event_raw_smart_tags':out/f'event_raw_smart_tags_temporal_{rid}.csv',
      'event_summary':out/f'event_summary_temporal_{rid}.json'}
    result.frames.to_csv(files['event_context'],index=False)
    result.episodes.to_csv(files['event_episodes'],index=False)
    result.type_summary.to_csv(files['event_type_summary'],index=False)
    result.sources.to_csv(files['event_signal_coverage'],index=False)
    result.tag_records.to_csv(files['event_raw_smart_tags'],index=False)
    status={**result.status,'tracking_status':tracking_status,'run_id':rid,
            'outputs':{k:str(v) for k,v in files.items()}}
    files['event_summary'].write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8')
    print('\n=== S31-v3 COMPLETADO ===')
    print(result.type_summary[['type','raw_smart_event_count','candidate_episodes',
           'corroborated_episodes','provider_context_episodes']].to_string(index=False))
    print('Total CORROBORATED_CONFIGURATION:',status['n_corrob_episodes'])
    print('Smart Tagging tipificados:',status['n_raw_typed_events'])
    print('Detalle de fuentes:',files['event_signal_coverage'])
    print('Episodios:',files['event_episodes'])
    print('No se han alterado H/A/D ni el archivo run.py.')

if __name__=='__main__':main()
