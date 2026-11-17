# Solución de posesión del equipo — FC Barcelona Hackathon 2027

Este directorio contiene **únicamente nuestro código** de segmentación de posesión. Se incorpora a una copia del [repositorio oficial](https://github.com/metrica-sports/more-than-a-hack-2027), cuyos archivos se mantienen intactos.

## Qué hace

1. **Lee y estandariza** el tracking, la homografía y Smart Tagging proporcionados por la organización.
2. **Construye familias de señales** geométricas, temporales y de movimiento, y reconstruye las observaciones faltantes del balón.
3. **Analiza juego interrumpido** con 15 detectores futbolísticos (`juego_parado/th01_*.py` a `th15_*.py`).
4. **Resuelve juego activo e inactivo** mediante condiciones futbolísticas y continuidad temporal. Asigna **H** (Barcelona), **A** (Athletic Club), **D** (balón fuera de juego) o **X** (sin respuesta).
5. **Genera el archivo de entrega** con una fila por frame del vídeo táctico (199.319 filas).

Las condiciones y umbrales de la solución fueron revisados por el equipo mediante criterio futbolístico y experiencia en analítica deportiva.

## Instrucciones: un solo comando

**Requisitos:** utilizar la copia base del concurso, sin modificar `games.py`, `starter/`, `evaluation/`, ni las dependencias oficiales. Los datos oficiales deben estar ubicados como indica [DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) (por defecto bajo `data/tactical/`, o usando `MTAH_DATA_DIR`).

**Paso 1.** Extraer este ZIP **en la raíz de ese repositorio**, junto a `games.py`. Los archivos del concurso no vienen incluidos ni se sustituyen.

**Paso 2.** Desde esa misma raíz, instalar los requisitos si es la primera ejecución:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-equipo.txt
```

**Paso 3.** Ejecutar **solo este comando**:

```bash
python generar_entrega.py
```

El programa llama internamente a las etapas necesarias, incluidos los 15 detectores y el motor final. **No es necesario ejecutar otros scripts ni ningún evaluador.** Al terminar, el CSV queda en:

```text
local_outputs/entrega/posesion_fc_barcelona.csv
```

No se escribe en la raíz del repositorio. Contiene las columnas `frame,label` y los frames `0` a `199318` en orden. El programa realiza una comprobación local del formato y de la huella de etiquetas de la configuración seleccionada; esto **no es una evaluación contra ground truth**. Si avisa de una huella diferente, el archivo generado no se puede considerar idéntico a la salida de referencia.

El **CSV** se sube al formulario oficial, no al repositorio público. El nombre para el formulario debe ajustarse al nombre del equipo solicitado por la organización; renombrar una copia si hace falta.

## Qué se incluye en el commit

Los scripts Python de este paquete, la carpeta `juego_parado/`, `requirements-equipo.txt`, `README_SOLUCION.md`, `DOCUMENTO_TECNICO.md` y el archivo `local_outputs/.gitignore`. **No incluir** `data/`, `local_outputs/` generado, reportes `ETAPA_*.md`, videos, claves ni CSV de tracking.

**Commit selectivo desde la raíz del repositorio:**

```bash
git add README_SOLUCION.md DOCUMENTO_TECNICO.md requirements-equipo.txt config.py generar_entrega.py run.py dead_ball_detectors.py g8_episode_engine.py g9_episode_refinement.py integrar_solo_th.py s*.py juego_parado/
git add -f local_outputs/.gitignore
git diff --cached --name-only
git commit -m "Add rule-based possession pipeline"
```

Revisar lo que muestra `git diff --cached --name-only` **antes de confirmar el commit**. El repositorio oficial suministra `games.py`, los cargadores y la documentación propia del concurso.

## Limitaciones

El balón puede faltar en oclusiones y duelos; el tracker de jugadores tiene fragmentación de identidades y Smart Tagging es automático. Las recepciones breves, los rebotes y los momentos exactos de reanudación pueden ser ambiguos. La inferencia emplea reglas y continuidad temporal, sin depender de etiquetas humanas del partido al ejecutarse.

Documentación oficial: [Datos](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) · [Requisitos de entrega](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md).
