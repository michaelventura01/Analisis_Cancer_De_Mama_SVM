# Análisis del notebook: clasificación de cáncer de mama con SVM

**Notebook analizado:** [Caso_Practico_1.ipynb](https://github.com/michaelventura01/Analisis_Cancer_De_Mama_SVM/blob/main/Caso_Practico_1.ipynb)  
**Fecha del análisis:** 30 de agosto de 2026

## Resumen ejecutivo

El notebook implementa una máquina de vectores de soporte (SVM) para clasificar tumores de mama como malignos (`1`) o benignos (`0`), a partir de 30 variables morfológicas. Incluye preparación de datos, escalado, selección de hiperparámetros mediante validación cruzada y visualizaciones con PCA, t-SNE y UMAP.

El modelo reportado obtiene resultados muy altos: **98 % de exactitud** y **AUC-ROC de 0.998** en el conjunto de prueba. Sin embargo, existe una **fuga de información** durante el escalado de las variables, por lo que esas métricas probablemente son algo optimistas. El notebook es un buen ejercicio de aprendizaje, pero no proporciona todavía evidencia suficiente para uso clínico.

## Flujo implementado

1. Carga un dataset de cáncer de mama desde una URL externa.
2. Elimina las columnas `Unnamed: 32` e `id`.
3. Convierte el diagnóstico: `M` a `1` y `B` a `0`.
4. Separa las variables predictoras (`X`) de la variable objetivo (`y`).
5. Estandariza las variables con `StandardScaler`.
6. Divide los datos en entrenamiento (70 %) y prueba (30 %).
7. Ajusta una SVM con búsqueda en rejilla (`GridSearchCV`) y validación cruzada de 5 particiones, optimizando AUC-ROC.
8. Evalúa el mejor modelo con reporte de clasificación, AUC-ROC y matriz de confusión.
9. Explora la estructura de los datos con PCA, t-SNE y UMAP.

## Resultados reportados

El mejor modelo fue una SVM lineal con:

```text
C = 0.1
kernel = linear
gamma = scale
```

Resultados en el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.98 |
| AUC-ROC | 0.9978 |
| Precisión, maligno | 0.98 |
| Recall / sensibilidad, maligno | 0.97 |
| F1, maligno | 0.98 |

Matriz de confusión:

```text
[[107, 1],
 [  2, 61]]
```

Interpretación:

- Se clasificaron correctamente 107 casos benignos y 61 malignos.
- Hubo 1 falso positivo: un caso benigno predicho como maligno.
- Hubo 2 falsos negativos: dos casos malignos predichos como benignos.

En un escenario médico, los falsos negativos son especialmente relevantes, porque pueden retrasar la detección o el tratamiento de un tumor maligno. Por ello, el recall de la clase maligna debe revisarse junto con la exactitud global.

## Hallazgos positivos

- La eliminación de `id` y de la columna vacía evita utilizar información no predictiva.
- La codificación de la variable objetivo es clara y apropiada para una clasificación binaria.
- La SVM es una elección razonable para este conjunto de datos de tamaño pequeño o medio y con variables numéricas.
- La búsqueda de hiperparámetros se realiza sólo sobre el conjunto de entrenamiento mediante validación cruzada.
- El uso de AUC-ROC como métrica de selección es preferible a usar únicamente accuracy, especialmente si las clases no están perfectamente balanceadas.
- El notebook presenta varias perspectivas exploratorias de los datos con PCA, t-SNE y UMAP.

## Problema crítico: fuga de información durante el escalado

El notebook estandariza todo el dataset antes de separar entrenamiento y prueba:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)
```

Esto permite que la media y la desviación estándar del conjunto de prueba influyan en la transformación aplicada al entrenamiento. Aunque no utiliza las etiquetas de prueba, sigue filtrando información estadística del test al proceso de entrenamiento.

### Impacto

- El conjunto de prueba deja de ser completamente independiente.
- La evaluación puede ser ligeramente más favorable de lo que sería con datos realmente nuevos.
- La validación cruzada dentro de `GridSearchCV` también usa características que fueron escaladas con información de todos los registros.

### Corrección recomendada

Primero se debe dividir el dataset y luego ajustar el escalador únicamente con entrenamiento. La opción más segura es integrar escalado y modelo en un `Pipeline`:

```python
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])

param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__kernel": ["linear", "rbf"],
    "svc__gamma": ["scale", "auto"]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)
```

Con este enfoque, cada partición de validación cruzada ajusta su propio escalador utilizando solamente los registros de entrenamiento correspondientes.

## Oportunidades de mejora

### División estratificada

La división actual no usa `stratify=y`. Añadirlo preserva la proporción de casos benignos y malignos en entrenamiento y prueba:

```python
train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

### Métricas más completas

Además de accuracy y AUC-ROC, se recomienda incluir:

- Curva ROC y curva Precision-Recall.
- Especificidad, sensibilidad y valor predictivo negativo.
- Intervalos de confianza mediante bootstrap o validación cruzada repetida.
- Curva de calibración, ya que el modelo genera probabilidades.
- Análisis explícito del umbral de decisión para priorizar sensibilidad si el objetivo es reducir falsos negativos.

### Comparación con modelos base

Conviene comparar la SVM con modelos simples y explicables, por ejemplo:

- Regresión logística regularizada.
- k vecinos más cercanos.
- Árbol de decisión y bosque aleatorio.
- XGBoost o Gradient Boosting, si el objetivo es rendimiento predictivo.

La comparación debe realizarse con el mismo esquema de validación y las mismas métricas.

### Visualizaciones

PCA, t-SNE y UMAP son útiles para exploración, pero no validan el desempeño predictivo. También se observan títulos reutilizados que mencionan `final_result` y `scale`, variables que no pertenecen a este dataset. Deben renombrarse para referirse a `diagnosis`.

### Reproducibilidad

- La fuente de datos externa no está fijada a una versión concreta; se recomienda guardar una copia del dataset o emplear una fuente estable y documentada.
- Las celdas de instalación de paquetes y los números de ejecución están desordenados. Es recomendable ejecutar el notebook desde cero antes de publicarlo.
- Conviene fijar las versiones de las bibliotecas en un archivo `requirements.txt` o `environment.yml`.
- Para un uso real, debe validarse el modelo en una cohorte externa e independiente.

## Conclusión

El notebook presenta un flujo de trabajo de aprendizaje sólido y obtiene resultados plausibles para este dataset. La SVM lineal parece separar bien ambas clases, pero las métricas actuales no deben tomarse como estimación definitiva del rendimiento debido a la fuga de información generada por el escalado previo a la división.

Tras reemplazar el escalado global por un `Pipeline`, repetir la evaluación y ampliar el análisis de sensibilidad, calibración y validación externa, el proyecto tendrá una base metodológica mucho más robusta.
