# Trabajo N°01 - Grupo 03
TITULO: Predicción de resistencia del concreto

Problema general

La dificultad para estimar de manera anticipada y precisa la resistencia a compresión del concreto limita el control de calidad, la optimización de mezclas y la toma de decisiones técnicas en proyectos de ingeniería civil.

Problemas específicos

No se conoce con precisión qué variables de la mezcla influyen en mayor medida en la resistencia a compresión del concreto.
Los métodos tradicionales de control dependen de ensayos de laboratorio que requieren tiempo para obtener resultados finales.
Existe incertidumbre en la relación entre los componentes del concreto y su resistencia final.
No siempre se emplean herramientas estadísticas o predictivas para optimizar el diseño de mezclas de concreto.

Variable dependiente:

- Resistencia a compresión del concreto

Variables independientes:

- Cemento
- Agua
- Agregado fino
- Agregado grueso
- Ceniza volante
- Escoria
- Superplastificante
- Edad del concreto

| Principio PCS      | Aplicación en este proyecto                                                                                                                                                                           | Justificación                                                                                                                                                                                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Predictability** | Evaluar la capacidad del modelo para predecir correctamente la resistencia a compresión del concreto a partir de sus componentes de mezcla, como cemento, agua, agregados, aditivos y edad de curado. | En el control de calidad del concreto, una predicción incorrecta puede generar decisiones técnicas equivocadas, como aceptar una mezcla que no alcance la resistencia esperada o rechazar una mezcla que sí cumple con los parámetros requeridos.           |
| **Computability**  | Usar modelos entrenables y ejecutables en GitHub, Google Colab o computadoras con recursos moderados, como regresión lineal, árbol de decisión, Random Forest o modelos similares.                    | El modelo debe ser viable para estudiantes o investigadores que no cuenten con equipos de alta capacidad computacional. Además, debe poder procesar los datos de dosificación y generar resultados estadísticos de forma práctica y reproducible.           |
| **Stability**      | Comparar los resultados del modelo usando diferentes particiones de datos, semillas aleatorias y métodos de validación, como entrenamiento/prueba o validación cruzada.                               | Un modelo confiable no debe funcionar bien solo con una división específica de los datos. Debe mantener un desempeño estable al variar la muestra de entrenamiento y prueba, demostrando que puede generalizar sus predicciones a nuevos diseños de mezcla. |

prediccion-resistencia-concreto/
│
├── data/
│   ├── concrete_compressive_strength_clean.csv
│   ├── resistencia_compresion_concreto.csv
│   ├── concrete_compressive_strength_original_headers.csv
│   └── data_dictionary.csv
│
├── docs/
│   ├── DATASET_CITATION.txt
│   ├── citation.bib
│   └── Concrete_Readme_original.txt
│
├── README_dataset.md
└── CITATION.cff
