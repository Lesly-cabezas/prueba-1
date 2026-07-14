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
# T2: Análisis Exploratorio y Plan Algorítmico
## Predicción de la resistencia a compresión del concreto — Grupo 03

**Objetivo del T2:** verificar la calidad del conjunto de datos, documentar las decisiones de limpieza, efectuar un análisis exploratorio y definir los modelos de regresión y las métricas que se utilizarán.

**Fuente oficial:** Yeh, I.-C. (1998). *Concrete Compressive Strength* [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5PK67.

El conjunto tiene **1030 observaciones**, **8 variables predictoras cuantitativas** y **1 variable objetivo cuantitativa**. La tarea es de **regresión**.
## 1. Librerías y configuración reproducible
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
