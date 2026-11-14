# Proyecto de IAX — Guía de desarrollo y reparto del trabajo

**Proyecto:** comparación de las explicaciones de ResNet50 y EfficientNet-B0 mediante Grad-CAM e Integrated Gradients  
**Equipo:** Geer Wang Liu, Zhenhua Ye Xiang y Linda Fei Payeras Olano  
**Documento de referencia:** `PropuestaDeProyecto-4.pdf`  
**Objetivo de esta guía:** convertir la propuesta en un plan ejecutable, con tareas paralelizables, entregables verificables y criterios comunes.

---

# 1. Resumen ejecutivo: qué vamos a construir

## 1.1. Pregunta de investigación

La propuesta plantea estudiar hasta qué punto las regiones destacadas por las explicaciones de dos modelos son similares y cuánto depende esa similitud del método de explicabilidad utilizado.

Se compararán:

- **ResNet50**, con pesos `IMAGENET1K_V2`.
- **EfficientNet-B0**, con pesos `IMAGENET1K_V1`.
- **Grad-CAM**, implementado con `pytorch-grad-cam`.
- **Integrated Gradients (IG)**, implementado con Captum.
- **Imagenette v2 (320 px)**: 300 imágenes de validación, 30 por categoría.

No se entrenarán ni ajustarán los modelos. Se evaluarán similitud, fidelidad, estabilidad, sensibilidad a la aleatorización de pesos e impacto computacional.

## 1.2. Qué resultados debe producir el proyecto

Al terminar deberíamos tener:

1. Un conjunto fijo y reproducible de 300 imágenes.
2. Predicciones de ambos modelos para las mismas imágenes.
3. Cuatro mapas de atribución por imagen:
   - ResNet50 + Grad-CAM.
   - EfficientNet-B0 + Grad-CAM.
   - ResNet50 + Integrated Gradients.
   - EfficientNet-B0 + Integrated Gradients.
4. Métricas de similitud entre mapas.
5. Curvas y AUC de deletion e insertion.
6. Resultados de estabilidad ante transformaciones leves.
7. Resultados de pruebas de aleatorización progresiva de pesos.
8. Mediciones de tiempo, energía y emisiones estimadas.
9. Figuras, tablas, análisis, memoria y presentación reproducibles.

**Idea clave:** generar mapas es solo una parte del proyecto. La contribución está en diseñar comparaciones justas y evaluar si las explicaciones son similares, fieles y estables.

## 1.3. Matriz experimental

| Modelo | Grad-CAM | Integrated Gradients |
|---|---|---|
| ResNet50 | Mapa A | Mapa B |
| EfficientNet-B0 | Mapa C | Mapa D |

Comparaciones que interesa realizar:

| Comparación | Pregunta |
|---|---|
| ResNet50 Grad-CAM vs EfficientNet Grad-CAM | ¿Coinciden los modelos cuando usamos Grad-CAM? |
| ResNet50 IG vs EfficientNet IG | ¿Coinciden los modelos cuando usamos IG? |
| ResNet50 Grad-CAM vs ResNet50 IG | ¿Cuánto depende la explicación del método en ResNet50? |
| EfficientNet Grad-CAM vs EfficientNet IG | ¿Cuánto depende la explicación del método en EfficientNet? |

Las dos primeras comparaciones responden directamente a la pregunta principal. Las dos últimas ayudan a estudiar el efecto del método.

---

# 2. Decisiones metodológicas que hay que cerrar antes de programar

Estas decisiones son importantes porque, si cada integrante toma una decisión distinta, después no podremos combinar los resultados de forma válida.

## 2.1. Mismas imágenes y mismos identificadores

- Seleccionar 30 imágenes de cada una de las 10 categorías de validación.
- Fijar una semilla.
- Guardar la selección en un CSV.
- No volver a muestrear imágenes en cada notebook.
- Los tres integrantes deben utilizar exactamente los mismos identificadores.

## 2.2. Preprocesamiento específico de cada modelo

Los pesos de Torchvision incluyen sus transformaciones recomendadas. Se deben utilizar las transformaciones correspondientes a cada conjunto de pesos:

- ResNet50: `ResNet50_Weights.IMAGENET1K_V2.transforms()`.
- EfficientNet-B0: `EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()`.

No debemos forzar un único tensor de entrada si las recetas de preprocesamiento de los modelos son diferentes. Sí debemos conservar la imagen original y documentar las transformaciones.

**Punto de control:** los recortes pueden hacer que dos mapas se refieran a contenido visual ligeramente diferente. Hay que comprobar y documentar el tamaño, recorte y región visible de cada entrada. Si se comparan mapas sobre la imagen original, deben llevarse a un marco espacial común de forma justificada; no basta con redimensionarlos sin comprobar la alineación.

## 2.3. Clase objetivo: comparación justa

Esta es una de las decisiones más importantes.

Para comparar las explicaciones de dos modelos, ambos deben explicar **la misma clase objetivo** en una imagen. Si ResNet explica su clase predicha y EfficientNet explica otra clase distinta, las diferencias del mapa podrían deberse a que se está explicando una pregunta diferente, no necesariamente a la arquitectura o al método.

Propuesta:

- **Análisis principal:** utilizar la etiqueta real de la imagen como clase objetivo común para ambos modelos. Así ambos explican la misma clase.
- **Análisis complementario:** explicar la clase predicha por cada modelo para estudiar cómo justifican sus propias decisiones. Interpretar esta comparación separadamente.
- Registrar siempre la etiqueta real, la predicción de cada modelo y la clase objetivo usada para generar cada mapa.

Si se decide utilizar otra estrategia como análisis principal, debe acordarse antes y aplicarse a los cuatro experimentos.

## 2.4. Normalización de mapas

Acordar una única convención:

- Un mapa 2D por imagen y método.
- Misma resolución espacial para las comparaciones.
- Normalización documentada, por ejemplo a `[0, 1]`.
- Guardar, si es posible, el mapa bruto además del normalizado.
- No cambiar la normalización entre métricas sin registrarlo.

Grad-CAM e IG no tienen exactamente la misma naturaleza. Por eso, las comparaciones entre métodos deben interpretarse con cautela y no presentarse como si ambas técnicas midieran exactamente lo mismo.

## 2.5. Baseline de Integrated Gradients y perturbaciones

Acordar y documentar:

- Baseline inicial de IG (por ejemplo, tensor cero en el espacio de entrada normalizado).
- Número de pasos de integración.
- Criterio de convergencia de Captum, si se registra.
- Estrategia de sustitución/ocultación para deletion e insertion.
- Número de niveles de perturbación.
- Transformaciones leves que se utilizarán para estabilidad.

El baseline de IG y el de deletion/insertion son decisiones relacionadas, pero no tienen por qué ser el mismo. Deben quedar especificados por separado.

---

# 3. Plan de trabajo detallado por fases

## Fase 0 — Organización, repositorio y entorno

**Objetivo:** crear una base común que permita trabajar en paralelo sin conflictos.

### Tareas

1. Crear el repositorio GitHub.
2. Crear y probar el entorno virtual.
3. Fijar versiones de Python y dependencias.
4. Crear la estructura de carpetas.
5. Definir una configuración común (`config.yaml`).
6. Acordar nombres de funciones, formatos de datos y convenciones.
7. Crear un README con instrucciones de instalación y ejecución.

### Estructura propuesta

```text
proyecto-iax/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
│
├── data/
│   └── imagenette/                 # no subir el dataset al repositorio
│
├── metadata/
│   ├── selected_images.csv
│   └── class_mapping.json
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── classifiers.py
│   ├── explainability/
│   │   ├── gradcam.py
│   │   └── integrated_gradients.py
│   ├── evaluation/
│   │   ├── similarity.py
│   │   ├── fidelity.py
│   │   ├── stability.py
│   │   └── randomization.py
│   ├── utils/
│   │   ├── visualization.py
│   │   ├── reproducibility.py
│   │   └── io.py
│   └── pipeline.py
│
├── notebooks/
│   ├── 01_dataset_and_predictions.ipynb
│   ├── 02_explanations.ipynb
│   ├── 03_similarity_and_fidelity.ipynb
│   ├── 04_stability_and_randomization.ipynb
│   └── 05_results_and_analysis.ipynb
│
├── results/
│   ├── predictions/
│   ├── explanations/
│   ├── metrics/
│   ├── figures/
│   └── impact/
│
└── docs/
    ├── methodology.md
    └── decisions.md
```

No es necesario que toda la lógica viva en notebooks. Es preferible que las funciones importantes estén en `src/` y que los notebooks las utilicen para ejecutar experimentos, visualizar resultados y explicar el procedimiento.

### Entregable

Repositorio que los tres integrantes pueden clonar, instalar y ejecutar con instrucciones comunes.

---

## Fase 1 — Dataset y selección reproducible

**Responsable principal:** Persona 1.  
**Pueden avanzar en paralelo:** Personas 2 y 3 pueden preparar sus módulos con imágenes/tensores de prueba mientras se termina esta fase.

### Tareas

1. Descargar Imagenette v2 (320 px).
2. Localizar las imágenes de validación y sus etiquetas.
3. Identificar las diez categorías.
4. Seleccionar 30 imágenes por categoría con una semilla fija.
5. Guardar rutas, identificadores y etiquetas en `metadata/selected_images.csv`.
6. Crear una función para cargar una imagen original.
7. Crear funciones para aplicar el preprocesamiento específico de cada modelo.
8. Comprobar que las 300 imágenes se cargan correctamente.
9. Crear una cuadrícula visual con ejemplos de todas las categorías.

### Comprobaciones

- 300 identificadores únicos.
- 30 imágenes por categoría.
- Ninguna ruta rota.
- Etiquetas coherentes con las carpetas/clases.
- Semilla y selección guardadas.
- Imagen original disponible para visualizar los mapas.

### Entregables

- `selected_images.csv`.
- `class_mapping.json`.
- `dataset.py`.
- `preprocessing.py`.
- Notebook de exploración y comprobaciones.

---

## Fase 2 — Carga de modelos e inferencia

**Responsable principal:** Persona 1.

### Tareas

1. Cargar ResNet50 con `IMAGENET1K_V2`.
2. Cargar EfficientNet-B0 con `IMAGENET1K_V1`.
3. Poner ambos modelos en modo evaluación (`eval()`).
4. Ejecutar inferencia sobre las 300 imágenes.
5. Registrar logits o probabilidades, Top-1, Top-5 y confianza.
6. Calcular accuracy de cada modelo.
7. Calcular la tasa de acuerdo.
8. Registrar los casos en los que ambos aciertan, ambos fallan o solo uno acierta.
9. Guardar las predicciones en CSV.

### Formato mínimo de salida

`results/predictions/predictions.csv`

| image_id | true_label | resnet_top1 | efficientnet_top1 | resnet_confidence | efficientnet_confidence | resnet_top5 | efficientnet_top5 |
|---|---|---|---|---:|---:|---|---|

Conviene guardar también los índices de clase de ImageNet, no solo los nombres, para evitar ambigüedades.

### Entregables

- `classifiers.py`.
- CSV de predicciones.
- Métricas de clasificación.
- Prueba de carga e inferencia reproducible.

---

## Fase 3 — Generación de explicaciones

Esta fase se puede paralelizar casi por completo: una persona implementa Grad-CAM y otra Integrated Gradients. Ambas utilizan el mismo contrato de entrada y salida.

### 3A. Grad-CAM

**Responsable principal:** Persona 2.

1. Elegir la capa objetivo de cada arquitectura.
2. Comprobar que `pytorch-grad-cam` puede acceder a ella.
3. Implementar la generación del mapa para ResNet50.
4. Implementar la generación del mapa para EfficientNet-B0.
5. Aceptar una clase objetivo explícita.
6. Redimensionar el mapa al marco espacial acordado.
7. Guardar mapa bruto y normalizado.
8. Crear visualizaciones superpuestas sobre la imagen.
9. Probar con varias imágenes de distintas categorías.

Capas candidatas iniciales:

- ResNet50: `layer4[-1]`.
- EfficientNet-B0: `features[-1]`.

Hay que comprobar experimentalmente que las capas elegidas producen mapas válidos. No asumir que por ser candidatas ya están verificadas.

### 3B. Integrated Gradients

**Responsable principal:** Persona 3.

1. Implementar IG con Captum para ResNet50.
2. Implementar IG con Captum para EfficientNet-B0.
3. Recibir la clase objetivo explícitamente.
4. Definir el baseline acordado.
5. Fijar inicialmente `n_steps` (por ejemplo, 50) y comprobar el coste.
6. Obtener atribuciones por canal.
7. Convertirlas en un mapa 2D siguiendo una regla acordada.
8. Redimensionar al marco espacial común.
9. Guardar mapa bruto y normalizado.
10. Probar con imágenes de distintas categorías.

La reducción de los canales (por ejemplo, suma o norma de las atribuciones) debe acordarse y documentarse. No elegir una operación diferente en cada modelo.

### Contrato común para ambos módulos

Ambas funciones deberían aceptar, conceptualmente:

```python
generate_explanation(
    model,
    input_tensor,
    target_class,
    ...
)
```

Y devolver:

```python
{
    "raw_map": ...,          # mapa sin normalización final
    "normalized_map": ...,   # mapa 2D en [0, 1]
    "metadata": {
        "method": ...,
        "model": ...,
        "target_class": ...,
        "input_size": ...
    }
}
```

El formato exacto puede variar, pero debe ser idéntico entre los dos módulos.

### Organización de resultados

```text
results/explanations/
├── resnet50/
│   ├── gradcam/
│   └── integrated_gradients/
└── efficientnet_b0/
    ├── gradcam/
    └── integrated_gradients/
```

Cada mapa debe poder vincularse inequívocamente con `image_id`, modelo, método y clase objetivo.

### Entregables

- `gradcam.py`.
- `integrated_gradients.py`.
- Cuatro conjuntos de mapas.
- Ejemplos visuales de control.
- Metadatos de generación.

---

## Fase 4 — Similitud entre explicaciones

**Responsable principal:** Persona 2.  
**Inicio:** puede implementar las métricas con mapas sintéticos mientras las personas 2 y 3 terminan sus métodos.

### Tareas

1. Implementar correlación de Spearman.
2. Implementar Overlap@20%.
3. Verificar que ambas funciones aceptan dos mapas 2D de la misma resolución.
4. Probar casos sencillos conocidos (mapas iguales, mapas invertidos, mapas sin solapamiento).
5. Calcular las cuatro comparaciones de la matriz experimental.
6. Guardar resultados por imagen.
7. Calcular estadísticas agregadas.
8. Crear histogramas, boxplots y ejemplos visuales de similitud alta/baja.

### Definiciones

**Spearman:** correlación entre los rangos de importancia de los píxeles de dos mapas.

**Overlap@20%:** seleccionar el 20% de píxeles más relevantes de cada mapa y calcular el solapamiento. Si se usa Jaccard:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Hay que definir cómo se resuelven empates en el umbral y mantenerlo igual para todos los mapas.

### Comparaciones mínimas

- ResNet Grad-CAM vs EfficientNet Grad-CAM.
- ResNet IG vs EfficientNet IG.
- ResNet Grad-CAM vs ResNet IG.
- EfficientNet Grad-CAM vs EfficientNet IG.

### Entregables

- `similarity.py`.
- CSV de métricas por imagen y comparación.
- Figuras y resumen estadístico.

---

## Fase 5 — Fidelidad: deletion e insertion

**Responsable principal:** Persona 3.

### Tareas

1. Definir el baseline o estrategia de sustitución de regiones.
2. Definir los porcentajes de eliminación/inserción.
3. Ordenar los píxeles o regiones por importancia.
4. Implementar deletion.
5. Implementar insertion.
6. Medir la probabilidad de la clase objetivo en cada paso.
7. Calcular las curvas.
8. Calcular AUC.
9. Ejecutar el procedimiento para los cuatro tipos de explicación y ambos modelos.
10. Guardar curvas y métricas por imagen.

### Interpretación

- **Deletion:** se eliminan progresivamente las regiones más relevantes. Una explicación más fiel debería provocar una caída más rápida de la puntuación de la clase objetivo.
- **Insertion:** se incorporan progresivamente las regiones más relevantes a una imagen base. Una explicación más fiel debería recuperar la puntuación con mayor rapidez.

No debe interpretarse el AUC sin indicar la orientación de la curva y la convención de cálculo. Dejad claro qué significa un valor alto o bajo en vuestra implementación.

### Control del coste

Deletion e insertion pueden requerir muchas inferencias. Para controlar el coste:

- Utilizar batches cuando sea posible.
- Empezar con un número reducido de niveles de perturbación.
- Hacer una prueba piloto con pocas imágenes.
- Medir el tiempo antes de lanzar las 300 imágenes.
- Si es necesario, reducir el número de imágenes de esta fase y justificarlo; no cambiar silenciosamente la muestra.

### Entregables

- `fidelity.py`.
- Curvas deletion/insertion.
- AUC por imagen, modelo y método.
- Figuras comparativas.

---

## Fase 6 — Estabilidad y aleatorización

**Responsable principal:** Persona 1 para estabilidad y Persona 3 para aleatorización.  
**Apoyo:** Persona 2 revisa que las comparaciones de mapas utilicen las mismas métricas.

### 6A. Estabilidad ante transformaciones leves

1. Definir un conjunto pequeño de transformaciones (brillo, contraste, ruido leve o transformación geométrica).
2. Fijar sus intensidades y semillas.
3. Aplicar las transformaciones a las imágenes.
4. Comprobar si la predicción se mantiene.
5. Generar de nuevo las explicaciones.
6. Comparar mapas originales y transformados.
7. Registrar por separado los casos donde cambia la predicción.
8. Calcular Spearman y Overlap@20%.
9. Resumir estabilidad por modelo y método.

No basta con decir que una transformación es leve: hay que especificar su intensidad. Para transformaciones geométricas, además, hay que alinear los mapas antes de compararlos.

### 6B. Aleatorización progresiva de pesos

1. Conservar una copia intacta de cada modelo preentrenado.
2. Definir el orden de aleatorización de bloques/capas.
3. Aleatorizar progresivamente los pesos.
4. Generar mapas en cada nivel.
5. Compararlos con los mapas del modelo original.
6. Registrar los cambios de predicción.
7. Representar la similitud respecto al nivel de aleatorización.

La aleatorización debe ser reproducible y no debe modificar el modelo original que se usa para las demás evaluaciones.

### Entregables

- `stability.py`.
- `randomization.py`.
- CSV de resultados.
- Gráficas de estabilidad y sensibilidad a la aleatorización.

---

## Fase 7 — Impacto computacional

**Responsable principal:** Persona 1.

### Tareas

1. Integrar CodeCarbon.
2. Medir por separado inferencia y generación de explicaciones.
3. Medir deletion/insertion y estabilidad si resulta viable.
4. Registrar hardware, duración, tamaño de muestra y versiones.
5. Guardar energía y emisiones estimadas.
6. Documentar limitaciones de la estimación.

Evitar mezclar en una misma medición la descarga de pesos, la carga del dataset y el experimento, salvo que se quiera medir expresamente el coste total.

### Entregable

`results/impact/impact_metrics.csv` y una tabla resumen para la memoria.

---

## Fase 8 — Integración, análisis y memoria

**Responsables:** las tres personas.

### Integración técnica

1. Ejecutar el pipeline completo sobre una imagen.
2. Ejecutarlo sobre un pequeño lote de prueba.
3. Comprobar que los identificadores coinciden en todos los CSV.
4. Verificar que no faltan mapas ni métricas.
5. Ejecutar el experimento completo.
6. Guardar configuración, semillas y versiones.
7. Dejar instrucciones para reproducir los resultados.

### Análisis científico

Responder con resultados a estas preguntas:

1. ¿Los dos modelos destacan regiones similares?
2. ¿La coincidencia cambia entre Grad-CAM e IG?
3. ¿Qué método produce explicaciones más fieles según deletion/insertion?
4. ¿Qué método/modelo es más estable ante transformaciones leves?
5. ¿Las explicaciones cambian al aleatorizar los pesos?
6. ¿Hay diferencias entre categorías?
7. ¿Qué limitaciones impiden generalizar los resultados?

### Memoria sugerida

1. Introducción y pregunta de investigación.
2. Fundamentos: modelos y métodos de explicabilidad.
3. Dataset y selección de imágenes.
4. Diseño experimental y decisiones metodológicas.
5. Métricas de evaluación.
6. Resultados.
7. Discusión.
8. Limitaciones y amenazas a la validez.
9. Conclusiones.
10. Referencias.
11. Anexo de reproducibilidad.

### Presentación

La presentación debe centrarse en:

- Pregunta y motivación.
- Diseño experimental.
- Ejemplos visuales de explicaciones.
- Resultados cuantitativos.
- Interpretación y limitaciones.
- Conclusión principal.

---

# 4. Repartición del trabajo para tres personas

## 4.1. Principio de organización

La propuesta inicial asignaba:

- Preparación de datos e inferencia a Linda.
- Grad-CAM y similitud a Zhenhua.
- Integrated Gradients y fidelidad a Geer.
- Estabilidad, aleatorización e impacto a Linda.

La distribución es razonable como punto de partida, pero hay un riesgo: la persona que prepara los datos puede convertirse en cuello de botella y acumular demasiadas tareas de integración y evaluación.

Propongo conservar la especialización por componentes, pero adelantar el trabajo paralelo y repartir las evaluaciones adicionales.

## 4.2. Reparto propuesto

### Persona 1 — Datos, modelos, pipeline y estabilidad

**Responsabilidad principal:** que el experimento tenga una base común y reproducible.

| Tarea | Estimación |
|---|---:|
| Preparar Imagenette y selección de 300 imágenes | 3 h |
| Implementar preprocesamiento y dataset | 2 h |
| Cargar modelos y ejecutar inferencia | 3 h |
| CSV de predicciones y métricas de clasificación | 1,5 h |
| Estabilidad ante transformaciones | 3 h |
| Integración de CodeCarbon | 1,5 h |
| Pruebas de integración del pipeline | 2 h |
| **Total individual estimado** | **16 h** |

**Entregables:**
- `dataset.py`
- `preprocessing.py`
- `classifiers.py`
- `stability.py`
- `selected_images.csv`
- `predictions.csv`
- mediciones de impacto

### Persona 2 — Grad-CAM y similitud

**Responsabilidad principal:** generar explicaciones Grad-CAM y analizar la coincidencia entre mapas.

| Tarea | Estimación |
|---|---:|
| Implementar Grad-CAM para ResNet50 | 2 h |
| Implementar Grad-CAM para EfficientNet-B0 | 2 h |
| Normalización, guardado y visualización | 2 h |
| Implementar Spearman | 1,5 h |
| Implementar Overlap@20% | 1,5 h |
| Comparaciones y análisis estadístico de similitud | 2 h |
| Pruebas con mapas sintéticos y reales | 2 h |
| **Total individual estimado** | **13 h** |

**Entregables:**
- `gradcam.py`
- `similarity.py`
- mapas Grad-CAM de ambos modelos
- CSV de similitud
- visualizaciones comparativas

### Persona 3 — Integrated Gradients, fidelidad y aleatorización

**Responsabilidad principal:** generar explicaciones IG y evaluar su fidelidad y sensibilidad a los pesos.

| Tarea | Estimación |
|---|---:|
| Implementar IG para ResNet50 | 2 h |
| Implementar IG para EfficientNet-B0 | 2 h |
| Baseline, reducción de canales y normalización | 1,5 h |
| Implementar deletion | 2 h |
| Implementar insertion y AUC | 2,5 h |
| Implementar aleatorización progresiva | 3 h |
| Pruebas y validación de resultados | 2 h |
| **Total individual estimado** | **15 h** |

**Entregables:**
- `integrated_gradients.py`
- `fidelity.py`
- `randomization.py`
- mapas IG de ambos modelos
- curvas y métricas de fidelidad
- resultados de aleatorización

### Trabajo compartido

Estas tareas no se asignan exclusivamente a una persona:

| Tarea | Participación |
|---|---|
| Cerrar decisiones metodológicas | Las tres |
| Revisar interfaces y formatos | Las tres |
| Validar ejemplos de mapas | Las tres |
| Ejecutar el experimento final | Las tres |
| Analizar resultados y extraer conclusiones | Las tres |
| Redactar y revisar la memoria | Las tres |
| Preparar y ensayar la defensa | Las tres |

La distribución técnica queda aproximadamente equilibrada: 16 h, 13 h y 15 h. Las horas son estimaciones iniciales, no compromisos rígidos. Si una tarea se complica, se redistribuye el apoyo sin perder la responsabilidad principal.

---

# 5. Cómo paralelizar para terminar antes

La clave es no esperar a que una persona termine todo para que empiece la siguiente. Hay que definir interfaces comunes y trabajar con pruebas pequeñas.

## 5.1. Paralelismo desde el primer día

| Persona 1 | Persona 2 | Persona 3 |
|---|---|---|
| Prepara dataset y modelos | Prepara módulo Grad-CAM con tensor de prueba | Prepara módulo IG con tensor de prueba |
| Define CSV de predicciones | Prueba capas objetivo | Prueba Captum y baseline |
| Entrega dataset/modelos | Implementa Grad-CAM para ambos modelos | Implementa IG para ambos modelos |
| Implementa estabilidad | Implementa similitud con mapas sintéticos | Implementa deletion/insertion con modelo de prueba |
| Integra CodeCarbon | Analiza similitud con mapas reales | Implementa aleatorización |

Mientras la Persona 1 prepara los datos, las Personas 2 y 3 no necesitan esperar: pueden desarrollar sus funciones con tensores aleatorios o una imagen de prueba y una interfaz acordada.

## 5.2. Contratos entre módulos

Acordad desde el principio estos contratos:

**Dataset**
- Devuelve `image_id`, etiqueta real, imagen original y entradas preprocesadas para cada modelo.

**Modelo**
- Recibe un tensor ya preprocesado.
- Devuelve logits/probabilidades.
- Permite calcular gradientes cuando se generan explicaciones.

**Explicabilidad**
- Recibe modelo, tensor y clase objetivo.
- Devuelve mapa 2D y metadatos.

**Evaluación**
- Recibe mapas o imágenes perturbadas.
- Devuelve métricas en estructuras simples y guardables.

**Resultados**
- Todas las salidas incluyen `image_id`, modelo, método y clase objetivo.

Con esto, cada persona puede programar sin conocer los detalles internos del módulo de otra.

## 5.3. Integración temprana obligatoria

No esperéis a tener las 300 imágenes procesadas para integrar.

Haced estos hitos:

1. **Prueba 1:** una imagen, dos modelos, cuatro mapas.
2. **Prueba 2:** cinco imágenes, cuatro mapas y métricas de similitud.
3. **Prueba 3:** diez imágenes y una ejecución de deletion/insertion.
4. **Prueba 4:** muestra completa de 300 imágenes para los experimentos principales.
5. **Prueba 5:** evaluaciones costosas de estabilidad, fidelidad y aleatorización.

Si algo falla, se detectará pronto y no después de horas de ejecución.

## 5.4. Git y coordinación

Recomiendo ramas por componente:

- `feature/dataset-models`
- `feature/gradcam-similarity`
- `feature/integrated-gradients-fidelity`
- `feature/stability-randomization`
- `main`

Normas:

- No trabajar directamente sobre `main`.
- Hacer commits pequeños y descriptivos.
- No subir el dataset ni resultados voluminosos al repositorio salvo que se acuerde.
- Añadir dependencias a `requirements.txt` y comunicar cambios.
- Revisar los cambios antes de fusionar.
- No modificar los contratos comunes sin avisar al resto.

---

# 6. Calendario recomendado

La propuesta inicial sitúa el desarrollo técnico entre el 11 y el 22 de noviembre, la integración entre el 20 y el 24 de noviembre, el análisis y la memoria entre el 25 de noviembre y el 15 de diciembre, y la defensa entre el 16 y el 18 de diciembre.

Partiendo de ese calendario, propongo esta secuencia:

| Periodo | Persona 1 | Persona 2 | Persona 3 | Hito compartido |
|---|---|---|---|---|
| 12–13 nov. | Entorno, dataset y selección | Preparar Grad-CAM y probar capas | Preparar IG y probar baseline | Contratos y decisiones metodológicas |
| 14–16 nov. | Modelos e inferencia | Grad-CAM para ambos modelos | IG para ambos modelos | Primera imagen con cuatro mapas |
| 17–19 nov. | Estabilidad y CodeCarbon | Similitud y visualización | Deletion/insertion y aleatorización | Prueba integrada con 5–10 imágenes |
| 20–22 nov. | Validación de datos y pipeline | Métricas finales de similitud | Fidelidad y aleatorización piloto | Ejecución completa o piloto ampliado |
| 23–24 nov. | Integración y documentación técnica | Integración y figuras | Integración y figuras | Notebook de viabilidad y propuesta revisada |
| 25 nov.–5 dic. | Análisis de estabilidad/impacto | Análisis de similitud | Análisis de fidelidad/aleatorización | Resultados consolidados |
| 6–15 dic. | Redacción | Redacción | Redacción | Memoria final revisada |
| 16–18 dic. | Presentación | Presentación | Presentación | Defensa |

Las fechas son una propuesta de organización basada en el calendario de la propuesta; ajustadlas si la asignatura establece hitos diferentes.

---

# 7. Criterios de finalización: checklist de control

## Organización y reproducibilidad

- [ ] Repositorio creado y estructura acordada.
- [ ] Entorno reproducible.
- [ ] Versiones de librerías registradas.
- [ ] Semillas fijadas.
- [ ] Configuración común.
- [ ] Instrucciones de ejecución en README.

## Dataset y clasificación

- [ ] Imagenette v2 descargado.
- [ ] 300 imágenes seleccionadas.
- [ ] 30 imágenes por categoría.
- [ ] CSV de selección guardado.
- [ ] Preprocesamiento de cada modelo comprobado.
- [ ] ResNet50 y EfficientNet-B0 cargados con los pesos correctos.
- [ ] Predicciones y Top-5 guardados.
- [ ] Accuracy y tasa de acuerdo calculadas.

## Explicabilidad

- [ ] Grad-CAM funciona en ResNet50.
- [ ] Grad-CAM funciona en EfficientNet-B0.
- [ ] IG funciona en ResNet50.
- [ ] IG funciona en EfficientNet-B0.
- [ ] Clase objetivo registrada.
- [ ] Baseline de IG documentado.
- [ ] Mapas guardados y vinculados a los identificadores.
- [ ] Mapas comparables espacialmente.

## Evaluación

- [ ] Spearman implementado.
- [ ] Overlap@20% implementado.
- [ ] Cuatro comparaciones realizadas.
- [ ] Deletion implementado.
- [ ] Insertion implementado.
- [ ] AUC calculadas e interpretadas.
- [ ] Transformaciones de estabilidad definidas.
- [ ] Predicciones preservadas comprobadas.
- [ ] Aleatorización progresiva implementada.
- [ ] Resultados por categoría analizados.

## Impacto y entrega

- [ ] Tiempos registrados.
- [ ] CodeCarbon integrado.
- [ ] Hardware y versiones documentados.
- [ ] Figuras y tablas finales generadas.
- [ ] Memoria revisada.
- [ ] Notebook reproducible.
- [ ] Presentación preparada.
- [ ] Limitaciones y uso de asistentes de IA documentados según las normas de la asignatura.

---

# 8. Riesgos principales y cómo evitarlos

| Riesgo | Consecuencia | Prevención |
|---|---|---|
| Cada persona usa imágenes distintas | Resultados incomparables | CSV único de selección |
| Diferente clase objetivo | Comparación injusta | Clase objetivo común en el análisis principal |
| Recortes distintos | Mapas espacialmente desalineados | Documentar y comprobar el marco espacial |
| Mapas con formatos distintos | Métricas incorrectas | Contrato común de salida |
| IG demasiado lento | Retraso del experimento | Piloto, batches y ajuste de pasos justificado |
| Deletion/insertion costoso | Ejecución demasiado larga | Probar con pocas imágenes y medir el coste |
| Transformaciones que cambian la clase | Estabilidad mal interpretada | Registrar si la predicción se conserva |
| Aleatorización modifica el modelo original | Resultados contaminados | Copias independientes de los modelos |
| Resultados difíciles de reproducir | Memoria poco sólida | Configuración, semillas, versiones y CSV |
| Interpretar similitud como razonamiento idéntico | Conclusiones excesivas | Explicar las limitaciones de las métricas |

---

# 9. Qué deberíais hacer primero: plan de las próximas 48 horas

Para empezar sin bloquearos, propongo este orden concreto.

## Reunión inicial (30–45 minutos, las tres personas)

1. Confirmar la estructura del repositorio.
2. Confirmar la selección de 300 imágenes y la semilla.
3. Acordar la clase objetivo común para las comparaciones principales.
4. Acordar baseline de IG y convención de normalización.
5. Acordar el formato de salida de los mapas.
6. Crear issues o tareas en GitHub y asignar responsables.

## Después de la reunión: trabajo simultáneo

**Persona 1**
- Crear entorno y repositorio.
- Descargar Imagenette.
- Seleccionar las 300 imágenes.
- Preparar dataset y carga de modelos.

**Persona 2**
- Instalar y probar `pytorch-grad-cam`.
- Verificar las capas candidatas de ResNet50 y EfficientNet-B0.
- Crear una función que genere y guarde un mapa Grad-CAM.

**Persona 3**
- Instalar y probar Captum.
- Implementar IG para un modelo.
- Comprobar baseline, número de pasos y formato de salida.
- Empezar a diseñar deletion/insertion.

## Primer hito de equipo

Antes de ampliar el experimento, las tres personas deben conseguir:

- Una imagen identificada.
- ResNet50 y EfficientNet-B0 ejecutándose.
- Una predicción por modelo.
- Cuatro mapas de atribución.
- Una visualización que permita comprobar que los mapas están correctamente superpuestos.
- Una primera comparación de similitud.

Si esto funciona, ya tendréis validada la arquitectura experimental. A partir de ahí, ampliar a 300 imágenes será principalmente ejecutar el pipeline y analizar resultados.

---

# 10. Referencias de la propuesta

Se mantienen como punto de partida las referencias incluidas en el documento:

1. Torchvision — documentación de modelos y pesos preentrenados.
2. Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*.
3. Sundararajan, Taly y Yan (2017) — *Axiomatic Attribution for Deep Networks*.
4. Quantus — biblioteca para evaluación de explicaciones.
5. Howard — Imagenette, Fast.ai.
6. Adebayo et al. (2018) — *Sanity Checks for Saliency Maps*.
7. Reglamento (UE) 2024/1689 — referencia regulatoria indicada en la propuesta.

Antes de entregar, comprobad las referencias, versiones y condiciones de uso en las fuentes originales.

---

# 11. Recomendación final

La prioridad no debería ser completar cuanto antes cada métrica de forma aislada, sino **validar pronto un pipeline mínimo de extremo a extremo**.

El orden más eficiente es:

1. Dataset y modelos funcionando.
2. Cuatro explicaciones para una misma imagen.
3. Formato común y comparación de mapas.
4. Piloto de fidelidad y estabilidad.
5. Ejecución ampliada.
6. Análisis y memoria.

La distribución propuesta mantiene responsables claros, pero evita que una persona tenga que terminar su parte para que las demás puedan empezar. Las tareas de Grad-CAM, IG, similitud y fidelidad pueden desarrollarse en paralelo si se acuerdan las interfaces desde el primer día.

**Regla de oro del proyecto:** cualquier resultado debe poder rastrearse hasta la imagen utilizada, el modelo, sus pesos, el método, la clase objetivo, la configuración y la versión del código.
