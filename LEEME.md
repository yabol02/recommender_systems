# Collaborative Filtering - User-based KNN

Sistema de recomendación por filtrado colaborativo usuario-usuario para predecir ratings en pares usuario-ítem no observados, aplicado a un dataset de valoraciones de libros.


## Objetivo y alcance

El sistema trabaja con dos entradas:

| Fichero | Formato | Rol |
|---|---|---|
| `train.csv` | `user_id, item_id, rating` | Histórico de ratings conocidos |
| `test.csv` | `ID, user_id, item_id` | Casos donde hay que estimar rating |

La salida es `predictions.csv` con columnas `ID` y `rating`.

## Ejecución

La ejecución se realiza desde `main.py`.

Comando habitual en este entorno:

```bash
uv run main.py
```

También puede ejecutarse con Python directamente activando el entorno virtual correspondiente:

```bash
python main.py
```

## Flujo del sistema

La arquitectura sigue una cadena simple y separada por responsabilidades:
1. Carga de datos.
2. Diagnóstico de *cold start*.
3. Construcción de matriz dispersa usuario-ítem.
4. Predicción KNN por lotes.
5. Persistencia de resultados.

Esta separación permite probar cada parte de forma aislada y sustituir componentes (métrica, estrategia) sin alterar el resto.

## Decisiones de diseño

### 1) Matriz dispersa en CSR con apoyo CSC

Se utiliza CSR (Compressed Sparse Row) para el cálculo de similitudes usuario-usuario porque está optimizada para operaciones por filas.
Además, se crea una vista CSC (Compressed Sparse Column) para recuperar de forma eficiente los usuarios que han puntuado un ítem concreto.

Decisión clave:
- CSR reduce coste en operaciones de similitud por lotes.
- CSC evita escanear usuarios irrelevantes al buscar vecinos válidos para un ítem.

Trade-off:
- Se paga un pequeño coste extra de memoria y conversión para disponer de ambos formatos.

### 2) Similitud euclídea sobre vectores sparse

La métrica elegida es:

$$
\text{sim}(u,v) = \frac{1}{1 + \lVert u-v \rVert_2}
$$

La distancia se calcula mediante expansión algebraica:

$$
\lVert u-v \rVert^2 = \lVert u \rVert^2 + \lVert v \rVert^2 - 2(u\cdot v)
$$

Decisión clave:
- Esta formulación permite aprovechar multiplicación sparse y retrasar la densificación al último paso.

Trade-off:
- Se interpreta ausencia de rating como valor 0 en el espacio vectorial, una simplificación habitual en CF basada en memoria.

### 3) Predicción con media ponderada por similitud

La estimación se construye con los K vecinos más similares que además han puntuado el ítem:

$$
\hat r_{u,i} = \frac{\sum_{v\in\mathcal N_k} \text{sim}(u,v)\, r_{v,i}}{\sum_{v\in\mathcal N_k} \text{sim}(u,v)}
$$

Decisión clave:
- Se priorizan vecinos más cercanos mediante pesos de similitud.
- Si no hay señal útil (denominador cero), se usa media simple de vecinos disponibles para evitar inestabilidad numérica.

### 4) Selección de vecinos eficiente

Para encontrar los top-K se usa selección parcial (`np.argpartition`) en lugar de ordenación completa.

Decisión clave:
- Coste `O(n)` frente a `O(n log n)` de un sort completo, reduciendo tiempo en el paso más repetido del pipeline.

Trade-off:
- No se obtiene ranking completo, solo los elementos necesarios para predecir.

### 5) Procesamiento por lotes (*batching*)

La similitud entre todos los usuarios de test y train puede ser muy grande en memoria.
Por eso, el sistema procesa usuarios de test por bloques de tamaño configurable.

Decisión clave:
- Mantener en RAM solo una submatriz de similitudes por iteración, controlando el pico de memoria.

Trade-off:
- Más lotes implican más iteraciones y algo más de overhead de control.

### 6) Manejo explícito de *cold start*

Antes de predecir, cada caso de test se clasifica en cuatro estados:

| Estado | Causa | Comportamiento |
|---|---|---|
| `OK` | Usuario e ítem presentes en train | Se predice con KNN normalmente |
| `COLD_USER` | El usuario no tiene ningún rating en train | Se asigna el rating de fallback |
| `UNKNOWN_ITEM` | El ítem nunca fue puntuado en train | Se asigna el rating de fallback |
| `BOTH` | Ni usuario ni ítem están en train | Se asigna el rating de fallback |

Decisión clave:
- Separar explícitamente los casos sin soporte en entrenamiento evita predicciones espurias.
- Aplicar un fallback conservador (`3.0` por defecto) garantiza cobertura total del test.

Trade-off:
- El fallback mejora robustez operativa, pero no personaliza. Es una elección pragmática; alternativas más ricas serían usar la media global del ítem o la media del usuario sobre otros ítems.

El diagnóstico se imprime antes de empezar a predecir:

```
================= Cold-start diagnosis =================
  OK (user + item in train)            23,938  ( 55.3 %)
  Cold user  (user not in train)        3,225  (  7.4 %)
  Unknown item (item not in train)     14,617  ( 33.7 %)
  Both missing                          1,540  (  3.6 %)
  Total                                43,320
========================================================
```

### 7) Extensibilidad por registros

La métrica de similitud y la estrategia de agregación se resuelven por nombre en registros dedicados (`SIMILARITY_REGISTRY`, `STRATEGY_REGISTRY`).

Decisión clave:
- El motor de predicción no depende de implementaciones concretas, solo de interfaces (`SimilarityFunction`, `PredictionStrategy`).
- Añadir una nueva métrica o estrategia no requiere modificar el flujo principal.

Trade-off:
- Requiere mantener coherencia entre los nombres usados en configuración y las claves registradas.

## Parametrización actual

En la configuración por defecto del proyecto:

- `K = 5` vecinos.
- `batch_size = 4096` usuarios de test por lote.
- `similarity = "euclidean"`.
- `strategy = "w_mean"`.
- `fallback = 3.0` para *cold start*.

Estos valores priorizan estabilidad y consumo de memoria razonable frente a una búsqueda exhaustiva de hiperparámetros.

## Resultado esperado

El sistema genera un fichero **<*K*>NN_<*SIMILARITY_FUNCTION*>_<*STRATEGY_FUNCTION*>.csv** en el directorio `preds` con un rating para cada ID de test.
Además, imprime por consola:

- Tamaño y cardinalidad de train/test.
- Resumen de *cold start* con desglose por estado.
- Progreso de lotes durante la predicción.

Esto facilita validar tanto calidad operativa (cobertura) como coste computacional (tiempo/memoria).