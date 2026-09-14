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

- `modelo_hvi/`: formato único de entrada (ECG + máscara) y adaptadores por fuente
  (patrón Strategy, uno por origen de datos). En Colab se usa una copia en
  `MyDrive/modelo-hvi/codigo/` con el commit de origen en `codigo/COMMIT`.
- `pruebas/`: tests unitarios de `modelo_hvi/`, todos con datos sintéticos. `uv run pytest`.
- `exploracion/`: scripts de análisis de datos; escriben agregados en `resultados/`.
- `herramientas/`: utilidades de verificación (por ejemplo, inspección de PDFs de ECG).
- `notebooks/`: un notebook de Colab por experimento, `E-XXX_<tema>.ipynb`, **sin resultados
  guardados**. La primera celda dice para qué sirve, qué datos usa, qué produce y qué reglas sigue.
- `resultados/`: salidas agregadas versionadas, que son la evidencia de los experimentos
  (`E-XXX_<tema>.json` para los que corren en Colab).

## Cómputo en Colab (decisión D-009)

El entrenamiento y la evaluación corren en Google Colab, no en la PC local. Copia de trabajo
en Drive:

- `MyDrive/modelo-hvi/datos/<fuente>/`: datos restringidos. Nunca se comparten.
- `MyDrive/modelo-hvi/notebooks/E-XXX_<tema>.ipynb`: el mismo número que el experimento en Obsidian.
- `MyDrive/modelo-hvi/resultados/E-XXX_<tema>.json`: solo agregados.

Al cerrar un experimento, se copia el notebook al repo sin resultados y se copia el JSON.
Nunca poner nombres de personas en rutas. Credenciales solo en Colab Secrets.

## Documentación del proyecto final

Este repo es parte del proyecto final. Todo lo que se documenta en Obsidian se rige por la
nota `02 - Documentación/Guía de documentación del proyecto.md` del vault: decisiones,
experimentos e hitos, no el paso a paso. Las notas de este repo viven en
`04 - Desarrollo/desarrollo modelo IA/`. Al cerrar cada sesión de trabajo, proponer las
entradas correspondientes y cargarlas solo con aprobación de un integrante. Nunca copiar
datos de pacientes al vault.
