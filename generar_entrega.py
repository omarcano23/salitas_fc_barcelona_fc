#!/usr/bin/env python3
"""Construye las señales, ejecuta el motor de posesión y entrega el CSV final."""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
# Huella de etiquetas de la salida final del equipo: se usa únicamente para comprobar reproducción.
EXPECTED_LABEL_SHA256='c3e487aeb34d8e12f2a5081b4b3e16a46d53669715ae28b15949c05c7ac242bd'
STAGES=('s14_matrix_v3.py','s23_matrix_v4.py','s25_possession_evidence_audit.py','s27_football_signal_audit.py')


def execute(command, env):
    print('Ejecutando:', ' '.join(str(x) for x in command),flush=True)
    subprocess.run([str(x) for x in command],cwd=ROOT,env=env,check=True)


def validate_and_digest(path):
    label_hash=hashlib.sha256()
    counts={k:0 for k in 'HADX'}
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=csv.reader(f)
        head=next(rows,None)
        if head!=['frame','label']:raise ValueError(f'Cabecera incorrecta: {head}')
        total=0
        for index,row in enumerate(rows):
            if len(row)!=2 or row[0]!=str(index) or row[1] not in counts:
                raise ValueError(f'Formato incorrecto en frame {index}: {row}')
            label_hash.update(row[1].encode('ascii'))
            counts[row[1]]+=1
            total+=1
    if total!=199319:raise ValueError(f'Filas: {total}, esperadas: 199319')
    return counts,label_hash.hexdigest()


def main():
    ap=argparse.ArgumentParser(description='Generación de etiquetas del partido táctico')
    ap.add_argument('--solo-motor',action='store_true',help='Conservar tablas intermedias existentes')
    ap.add_argument('--evidencia-th',type=Path,help='CSV de 15 detectores ya generado; para --solo-motor')
    ap.add_argument('--sin-comparar',action='store_true',help='Emitir CSV aunque no coincida la huella prevista')
    args=ap.parse_args()
    for path in (ROOT/'games.py',ROOT/'starter/load_atd.py',ROOT/'starter/load_homography.py',ROOT/'starter/load_smart_tagging.py'):
        if not path.is_file():ap.error(f'Falta componente del repositorio oficial: {path.relative_to(ROOT)}')
    env=os.environ.copy()
    # Eliminar sobreescrituras externas accidentales; run.py contiene los valores del motor.
    for key in list(env):
        if key.startswith(('RAM_','G4_S17_')):env.pop(key)
    env['RAM_G9_FAST_IO']='0'
    if not args.solo_motor:
        for script in STAGES:execute((sys.executable,ROOT/script),env)
        folder=ROOT/'local_outputs/runs'
        before={p.resolve() for p in folder.glob('th_only_*') if p.is_dir()} if folder.exists() else set()
        execute((sys.executable,ROOT/'integrar_solo_th.py','--project',ROOT,'--th-dir',ROOT/'juego_parado','--step','5','--fail-fast'),env)
        after={p.resolve() for p in folder.glob('th_only_*') if p.is_dir()}
        created=list(after-before)
        if len(created)!=1:raise RuntimeError('No se detectó una carpeta única con evidencia TH nueva')
        evidence=created[0]/'evidencia_th_normalizada.csv'
    else:
        evidence=args.evidencia_th
        if evidence is None:
            folders=sorted((ROOT/'local_outputs/runs').glob('th_only_*/evidencia_th_normalizada.csv'))
            if not folders:ap.error('Falta --evidencia-th o generación de TH')
            evidence=folders[-1]
    if not evidence.is_file():ap.error(f'No existe evidencia TH: {evidence}')
    runs=ROOT/'local_outputs/runs'
    before={p.resolve() for p in runs.glob('submission_*.csv')}
    execute((sys.executable,ROOT/'run.py','--th-evidence',evidence.resolve()),env)
    added={p.resolve() for p in runs.glob('submission_*.csv')}-before
    if len(added)!=1:raise RuntimeError('No se encontró un CSV único recién generado')
    output=added.pop()
    counts,digest=validate_and_digest(output)
    print('CSV:',output)
    print('Distribución:',counts)
    print('SHA256 de etiquetas:',digest)
    final=ROOT/'local_outputs/entrega/posesion_fc_barcelona.csv'
    final.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(output,final)
    print('ARCHIVO FINAL:',final)
    if digest!=EXPECTED_LABEL_SHA256:
        msg='ADVERTENCIA: las etiquetas generadas no coinciden con la huella prevista para esta versión del motor.'
        print(msg,file=sys.stderr)
        if not args.sin_comparar:raise SystemExit(3)
    else:print('OK: etiquetas idénticas a la salida prevista.')

if __name__=='__main__':main()
