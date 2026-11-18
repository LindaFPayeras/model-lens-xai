# Preparación de datos: estado e integración

Este documento describe el trabajo relacionado con
[`notebooks/00_preparacion_datos.ipynb`](../notebooks/00_preparacion_datos.ipynb),
qué archivos utiliza y qué necesita saber el resto del equipo para integrar
selección de imágenes y preprocesamiento.

## Alcance actual

La preparación se apoya en dos módulos:

- [`src/data/dataset.py`](../src/data/dataset.py): selecciona imágenes de
  validación de Imagenette, construye sus metadatos y guarda el manifiesto CSV.
- [`src/data/preprocesing.py`](../src/data/preprocesing.py): abre una imagen
  como RGB y aplica las transformaciones asociadas a los pesos de ResNet50 y
  EfficientNet-B0.

El notebook importa y recarga esos módulos, ejecuta la selección con
`number_class=30`, verifica que existan 300 imágenes repartidas en 10 clases,
muestra una imagen representativa de cada clase y valida el preprocesamiento
de toda la selección con ResNet50 y EfficientNet-B0.

## Selección y manifiesto de imágenes

`dataset.extractData()` recorre las carpetas de clase de
`data/imagenette2-320/val`, selecciona aleatoriamente 30 imágenes por carpeta
y devuelve un DataFrame con estas columnas:

| Columna | Significado |
|---|---|
| `image_id` | Nombre de la imagen sin extensión. |
| `path` | Ruta relativa a la raíz del proyecto. |
| `true_label` | Nombre de la carpeta de clase de Imagenette. |

Con las diez clases de Imagenette, el resultado esperado es una selección de
300 imágenes. El CSV se escribe en `metadata/selected_images.csv`. La carpeta
`metadata/` debe existir antes de ejecutar el guardado; la función actual no la
crea automáticamente.

El manifiesto actual tiene 300 registros y 30 imágenes por cada una de las
diez clases. Las rutas son relativas a la raíz del repositorio y permiten que
las demás personas carguen la misma muestra sin copiar las imágenes al CSV.
El CSV sirve como entrada compartida para inferencia y evaluaciones posteriores;
no contiene los píxeles de las imágenes.

### Reproducibilidad y formato del manifiesto

La celda principal genera `df` con
`dataset.extractData(number_class=30)`, por lo que utiliza la semilla por
defecto de `extractData()`, `seed=44`.

El notebook incluye además una comprobación explícita con `seed=42`: genera
dos selecciones independientes y compara sus listas de `image_id`. Esta
comprobación confirma que se puede reproducir exactamente la selección cuando
la semilla se pasa de forma explícita.

`DataFrame.to_csv()` se llama sin `index=False`, por lo que el CSV generado
incluye una columna adicional de índice llamada `Unnamed: 0` al leerlo con
`pandas.read_csv()`. Las columnas de datos siguen siendo `image_id`, `path` y
`true_label`.

## Carga de imágenes y preprocesamiento

`dataset.get_img(row)` recibe una fila del manifiesto, resuelve su ruta desde
la raíz del proyecto, comprueba que el archivo exista y lo abre como RGB.
Devuelve un diccionario con `image_id`, `path`, `true_label` e `image` (imagen
PIL). Esto permite conservar la imagen original para visualización,
transformaciones o explicaciones.

`preprocesing.preprocess_image(img_path)` recibe una ruta y devuelve una tupla:

1. La imagen original en RGB.
2. El tensor transformado para ResNet50.
3. El tensor transformado para EfficientNet-B0.

Las transformaciones se obtienen mediante `ResNet50_Weights.DEFAULT.transforms()`
y `EfficientNet_B0_Weights.DEFAULT.transforms()` de torchvision. Por tanto,
cada modelo debe recibir su tensor correspondiente, no la imagen PIL ni el
tensor del otro modelo. Para reproducibilidad entre entornos, el equipo debe
mantener compatibles las versiones de torchvision y los pesos usados.

## Validaciones realizadas en el notebook

Después de crear el manifiesto, el notebook realiza las siguientes
comprobaciones:

- verifica que `len(df)` sea 300;
- verifica que existan exactamente diez valores de `true_label`;
- verifica que todas las clases tengan 30 imágenes;
- genera dos selecciones con `seed=42` y comprueba que sus identificadores
  coincidan;
- lee el CSV con `pandas.read_csv()` y comprueba su contenido;
- muestra en una figura una imagen representativa de cada una de las diez
  clases.

El notebook recorre también las 300 filas de `df`, ejecuta
`preprocesing.preprocess_image()` para cada ruta y comprueba:

- que los tensores de ResNet50 tengan forma `(3, 224, 224)`;
- que los tensores de EfficientNet-B0 tengan forma `(3, 224, 224)`;
- que ambos tensores contengan solamente valores finitos.

La ejecución guardada confirma que las 300 imágenes se preprocesaron sin
errores y que los tensores obtenidos tienen ambas formas esperadas.
La conclusión actual es que el pipeline de preprocesamiento está validado para
la muestra seleccionada.

## Impacto en los demás módulos

- **Inferencia (`classifiers.py`):** consumir el manifiesto para recorrer la
  muestra y usar el tensor correcto para cada modelo. Mantener `image_id` y
  `true_label` en cada predicción para poder unir los resultados con otros CSV.
- **Explicaciones:** conservar la imagen original y aplicar a cada modelo su
  propio preprocesamiento antes de generar la explicación.
- **Estabilidad:** partir de la imagen original, aplicar la transformación
  definida por el experimento y volver a preprocesar por modelo. Registrar
  semilla e intensidad de la transformación.
- **Resultados compartidos:** tomar `image_id` del manifiesto como clave de
  unión entre predicciones, explicaciones y métricas.

## Compatibilidad a revisar durante la integración

- En el código, la función para cargar una imagen se llama `get_img`; el texto
  explicativo del notebook menciona `get_image`. El resto del equipo debe
  importar el nombre que realmente existe o acordar renombrarlo conjuntamente.
- El nombre actual del módulo es `preprocesing.py` (no
  `preprocessing.py`), y el notebook lo importa con ese nombre.
- La celda del notebook que describe la creación del manifiesto llama a
  `extractData()` y `createFil_selectedImages()`. La llamada a `get_img()` se
  debe realizar cuando se necesite cargar los píxeles; no forma parte de la
  generación del CSV de selección.
- La validación actual recorre las 300 imágenes y guarda sus conclusiones en
  las salidas del notebook. Las dimensiones y la ausencia de errores pueden
  variar si el entorno cambia sus pesos o versiones de torchvision.

## Cómo reproducir

Desde la raíz del repositorio y con el entorno del proyecto activado:

1. Asegurarse de que Imagenette está disponible en
   `data/imagenette2-320/val`.
2. Asegurarse de que existe la carpeta `metadata/`.
3. Abrir y ejecutar `notebooks/00_preparacion_datos.ipynb` con el kernel del
   entorno del proyecto.
4. Comprobar que `metadata/selected_images.csv` contiene 300 filas de datos,
   30 por clase, y que las rutas referenciadas existen.
5. Ejecutar las comprobaciones de conteo y reproducibilidad; para fijar
   explícitamente la selección, pasar `seed=42` o otra semilla acordada a
   `extractData()`.
6. Ejecutar la celda de visualización para revisar una muestra por clase y la
   validación completa del preprocesamiento.
7. Confirmar que no se han registrado errores y que ambos modelos producen
   tensores de forma `(3, 224, 224)`.

La descarga de Imagenette no forma parte de este notebook; el dataset no debe
subirse al repositorio si el equipo mantiene la convención indicada en la guía
del proyecto.
