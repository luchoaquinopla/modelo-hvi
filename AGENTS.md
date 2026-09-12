# AGENTS.md — modelo-hvi

Guía para cualquier agente (humano o IA) que trabaje en este repositorio.

## Qué es este proyecto

Modelo de IA del proyecto final: **evaluar si un modelo puede detectar hipertrofia
ventricular izquierda (HVI), confirmada por ecocardiograma, a partir del ECG de 12
derivaciones**. Es el consumidor de los datos que produce el pipeline de anonimización
(`D:\proyectos\anonimizacion`), que es un repo separado.

- MVP (diciembre 2026): entrenamiento con el dataset público EchoNext (PhysioNet).
- Proyecto de 12 meses: el mismo modelo, ajustado con datos del Instituto de Cardiología
  de Corrientes que salen del pipeline.

## Datos (no negociable)

- EchoNext está bajo la **PhysioNet Restricted Health Data License 1.5.0**: acceso
  individual, solo investigación, sin uso comercial. Los datos viven en
  `D:\datasets\echonext\` (configurable con la variable `ECHONEXT_DIR`), **nunca** en el
  repo ni en el vault de Obsidian.
- **Nunca** imprimir, loguear ni guardar filas individuales de pacientes. Los scripts
  solo emiten resultados **agregados** (cantidades, porcentajes, métricas).
- Nunca enviar datos a servicios online de terceros (incluidos asistentes de IA): solo
  resultados agregados.

## Convenciones de código

- Identificadores, módulos y tests en **español**, como en el repo del anonimizador.
- Comentarios cortos, solo el *por qué* cuando no es obvio.
- Entorno con `uv`: `uv run python <script>`.

## Estructura

- `exploracion/`: scripts de análisis de datos; escriben agregados en `resultados/`.
- `herramientas/`: utilidades de verificación (por ejemplo, inspección de PDFs de ECG).
- `resultados/`: salidas agregadas versionadas, que son la evidencia de los experimentos.

## Documentación del proyecto final

Este repo es parte del proyecto final. Todo lo que se documenta en Obsidian se rige por la
nota `02 - Documentación/Guía de documentación del proyecto.md` del vault: decisiones,
experimentos e hitos, no el paso a paso. Las notas de este repo viven en
`04 - Desarrollo/desarrollo modelo IA/`. Al cerrar cada sesión de trabajo, proponer las
entradas correspondientes y cargarlas solo con aprobación de un integrante. Nunca copiar
datos de pacientes al vault.
