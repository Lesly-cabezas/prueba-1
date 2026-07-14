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
## 2. Carga e identificación del conjunto de datos
# El notebook busca primero una copia local, lo que facilita subirlo a GitHub.
# Si no la encuentra, usa una copia pública del mismo dataset UCI.
candidates = [
    Path('concrete.csv'),
    Path('data/concrete.csv'),
    Path('/mnt/data/concrete.csv'),
]

for path in candidates:
    if path.exists():
        DATA_PATH = path
        break
else:
    DATA_PATH = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/concrete.csv'

df = pd.read_csv(DATA_PATH)
print('Fuente leída:', DATA_PATH)
print('Dimensiones:', df.shape)
df.head()
Fuente leída: concrete.csv
Dimensiones: (1030, 9)

### Diccionario de variables
| Variable | Rol | Unidad | Descripción |
|---|---|---|---|
| cement | Predictora | kg/m³ | Contenido de cemento |
| slag | Predictora | kg/m³ | Escoria granulada de alto horno |
| ash | Predictora | kg/m³ | Ceniza volante |
| water | Predictora | kg/m³ | Agua |
| superplastic | Predictora | kg/m³ | Superplastificante |
| coarseagg | Predictora | kg/m³ | Agregado grueso |
| fineagg | Predictora | kg/m³ | Agregado fino |
| age | Predictora | días | Edad de curado |
| strength | Objetivo | MPa | Resistencia a compresión del concreto |

Los valores cero de escoria, ceniza volante o superplastificante representan **ausencia del componente** y no se consideran valores perdidos.

## 3. Auditoría de calidad y limpieza
quality = pd.DataFrame({
    'tipo': df.dtypes.astype(str),
    'nulos': df.isna().sum(),
    'valores_unicos': df.nunique(),
    'mínimo': df.min(),
    'máximo': df.max(),
})
quality
print('Total de valores nulos:', int(df.isna().sum().sum()))
print('Filas duplicadas exactas:', int(df.duplicated().sum()))
print('Registros totales:', len(df))

mix_cols = ['cement','slag','ash','water','superplastic','coarseagg','fineagg']
print('Composiciones de mezcla únicas (sin considerar edad):', df[mix_cols].drop_duplicates().shape[0])
Total de valores nulos: 0
Filas duplicadas exactas: 25
Registros totales: 1030
Composiciones de mezcla únicas (sin considerar edad): 427
### Decisión sobre duplicados

Se identifican filas exactamente repetidas. No se eliminan de manera automática porque pueden corresponder a réplicas experimentales válidas. Sin embargo, un reparto aleatorio podría ubicar observaciones de la misma mezcla en entrenamiento y prueba, produciendo una evaluación demasiado optimista. Para aplicar el principio de **estabilidad PCS**, se realizarán dos comprobaciones:

1. **KFold aleatorio**, comparable con gran parte de la literatura.
2. **GroupKFold por composición de mezcla**, que impide que una misma dosificación aparezca simultáneamente en entrenamiento y prueba.

Además, se compararán resultados con y sin duplicados exactos como análisis de sensibilidad.

## 4. Estadística descriptiva
df.describe().T

## 5. Distribuciones
labels = {
    'cement': 'Cemento (kg/m³)', 'slag': 'Escoria (kg/m³)',
    'ash': 'Ceniza volante (kg/m³)', 'water': 'Agua (kg/m³)',
    'superplastic': 'Superplastificante (kg/m³)',
    'coarseagg': 'Agregado grueso (kg/m³)',
    'fineagg': 'Agregado fino (kg/m³)', 'age': 'Edad (días)',
    'strength': 'Resistencia (MPa)'
}

fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, col in zip(axes.ravel(), df.columns):
    ax.hist(df[col], bins=25, edgecolor='black', linewidth=0.5)
    ax.set_title(labels[col], fontsize=10)
    ax.set_ylabel('Frecuencia')
    ax.grid(alpha=0.2)
fig.suptitle('Distribución de las variables', fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()

**Hallazgos iniciales:** la edad, la escoria, la ceniza volante y el superplastificante presentan asimetría y concentración de valores en cero. Esto es coherente con mezclas que no incorporan determinados materiales suplementarios. La resistencia se distribuye entre valores bajos y altos, con una media cercana a 36 MPa.

