# Segmentación de posesión de balón — documento técnico

## Objetivo

Etiquetar el vídeo táctico FC Barcelona–Athletic Club a 25 fps con los estados **H**, **A**, **D** y **X**. La salida consiste en una fila por fotograma del vídeo táctico. La unidad de evaluación principal es el tramo continuo de posesión, no una decisión aislada de cada frame.

## Datos y preparación

El sistema emplea el tracking ATD de jugadores y balón, la homografía del vídeo táctico y las anotaciones automáticas de Smart Tagging que distribuye la organización. Los registros se alinean con el índice del vídeo (frame 0–199318); se comprueban el número de filas, el orden y los periodos de juego. Los archivos de origen se conservan sin modificación.

La extracción se ordena en familias de identidad/tiempo, geometría colectiva, movimiento, ocupación espacial, distancia al balón, aproximación y cinemática. Se reconstruye la posición del balón en observaciones ausentes mediante el código de `s17_ball_reconstruction.py`, manteniendo información sobre fiabilidad y origen de las observaciones.

## Lógica de decisión

El sistema es **algorítmico y basado en reglas**, no un clasificador entrenado para inferencia. Se organiza en dos decisiones enlazadas:

- **Estado inactivo (`D`)**: quince detectores proporcionan hipótesis sobre saques, faltas, fuera de juego, goles, interrupciones disciplinarias, revisiones, lesiones, pausas colectivas, sustituciones y bote neutral. El motor exige coherencia temporal y corroboración física: una anotación táctica por sí sola no determina `D`.
- **Estado activo (`H` o `A`)**: se combinan la distancia al balón, su fiabilidad, el margen de control geométrico, el movimiento y la continuidad de la posesión. Las condiciones de transición, permanencia y suavizado temporal buscan evitar saltos por rebotes o toques sin control. Los estados inactivos quedan separados de los tramos de posesión.

La configuración final incorpora umbrales de distancia y tiempo, prioridades lógicas y ventanas de decisión. El equipo ha revisado la coherencia futbolística de estas condiciones con su experiencia en analítica deportiva, interpretación de jugadas y medidas de campo.

## Ejecución

Partiendo de una copia original del repositorio oficial, colocar los archivos propios de este paquete en su raíz; conservar `games.py`, `starter/` y `evaluation/` sin modificarlos. Instalar las dependencias del concurso y `requirements-equipo.txt`, colocar los activos de datos bajo `data/tactical/` y ejecutar:

```bash
python generar_entrega.py
```

El proceso encadena la matriz base, la reconstrucción del balón, las señales futbolísticas, los 15 detectores, la inferencia H/A/D/X y la escritura del CSV final. El archivo para enviar se guarda en `local_outputs/entrega/posesion_fc_barcelona.csv`; se renombra para el envío con el nombre del equipo. El código no necesita etiquetas humanas del partido durante la inferencia.

## Limitaciones

Las observaciones faltantes del balón y los cambios de identidad del tracker disminuyen la certeza de control. Las deflexiones, los duelos aéreos, las jugadas de transición rápida y la exactitud del reinicio son especialmente difíciles. Las anotaciones automáticas pueden contener errores y el sistema emplea coherencia temporal en vez de aceptar cada evento como certeza. Una implementación de reglas puede producir fronteras aproximadas incluso cuando identifica correctamente al equipo predominante.

## Información oficial

[Desafío y datos](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md) · [Evaluación](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md) · [Reglas](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) · [Envío](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)
