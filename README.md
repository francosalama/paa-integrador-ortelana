# Ortelana Textil — Clasificación de `Format_Reporte`

Trabajo práctico integrador grupal IADS5 (Procesamiento de Aprendizaje Automático).

**Integrantes:** Camila Belén Capua, Franco Salama, Juan Pablo Ruax Debruijn, Lorena Wajchman, Tomas Rossi y Miguel Ángel Tadakuma

## Problema
- **Principal:** clasificación multiclase de `Format_Reporte` (B, C, M, S) con `Año, Metros1, Metros2`. El modelo final no usa
  `Sucursal_ID` ni `Tela_ID`: el formato es constante para cada par (Sucursal, Tela), así que para un par conocido no hace
  falta machine learning (alcanza con su formato histórico). El modelo tiene sentido para pares nuevos y se valida para ese uso.
- **Secundario (regresión):** `volumen_anual_metros` (suma anual de los metros con dato de cada fila) con `Sucursal_ID, Tela_ID, Año`, usando `Ridge(alpha=1)`. No usamos `Metros2` porque mide un período distinto según el formato (un mes en M, un semestre en S y C) y no es comparable entre filas. Las columnas `Metros1`–`Metros12` no entran como features porque son los sumandos del target.
- `Format_Reporte` es la granularidad temporal con la que la sucursal consolidó las ventas: M mensual (12 períodos), B bimestral (6), S semestral (2) y C ciclo estacional (2).
- Se excluyen `Metros3`–`Metros12`: su patrón de faltantes revela la clase.

## Estructura
```
actividad_01.ipynb … actividad_16.ipynb   un notebook por actividad (ejecutar en orden)
ventas_historicas_ortelana_argentina.csv
particion_train_test.csv            índices train/test (80/20, estratificado, random_state=42)
models/modelo_productivo.joblib     pipeline final serializado
src/predecir.py                     script de predicción por línea de comandos
requirements.txt                    dependencias
```

## Reproducción
```
python -m pip install -r requirements.txt
for n in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16; do
    jupyter nbconvert --to notebook --execute --inplace actividad_$n.ipynb
done
python src/predecir.py --entrada nuevos_registros.csv --salida predicciones.csv
```
El CSV de entrada debe tener las columnas `Año, Metros1, Metros2` (se admiten faltantes). Otras columnas, como `Sucursal_ID`
o `Tela_ID`, se conservan en la salida pero el modelo no las usa.

## Modelo final
Pipeline: `ColumnTransformer` (`Año`, `Metros1`, `Metros2`: imputación mediana, sin escalado porque Random Forest es
invariante a la escala) + `RandomForestClassifier(n_estimators=300, max_features=3, min_samples_leaf=3, random_state=42)`.
Los hiperparámetros se eligieron con `GridSearchCV` validado con CV agrupada por par (`StratifiedGroupKFold`), que
imita el uso real: clasificar pares (Sucursal, Tela) que el modelo no vio.

| Etapa | F1 macro |
|---|---:|
| **Selección: GridSearchCV, CV agrupada por par (5 folds, train) — estimación para pares nuevos** | **0.8868** |
| Referencia: misma configuración con CV aleatoria (5 folds, train) | 0.9094 |
| Final: test (20%, evaluado una sola vez) | 0.9107 |

Accuracy test: 0.9087

```
              precision    recall  f1-score   support

           B     0.9722    0.9589    0.9655       146
           C     0.9527    0.9360    0.9443       172
           M     0.8092    0.9150    0.8589       153
           S     0.9123    0.8387    0.8739       186

    accuracy                         0.9087       657
   macro avg     0.9116    0.9122    0.9107       657
weighted avg     0.9122    0.9087    0.9092       657
```

## Limitaciones
- Todos los pares (Sucursal, Tela) de test aparecen en train (partición aleatoria pedida), así que el test mide pares
  conocidos. La estimación para pares nuevos es la de la CV agrupada (0.887), algo menor.
- Los modelos con `Sucursal_ID` y `Tela_ID` llegan a 0.97–0.99 con la partición aleatoria, pero caen a 0.64–0.70 con la
  CV agrupada (actividad 13): memorizan el formato fijo de cada par. Por eso quedaron descartados.
- Las confusiones más frecuentes son entre S, C y M cuando el volumen es atípico para la clase real (actividad 14).
- Supone que `Metros1` y `Metros2` están disponibles al momento de predecir.

## Regresión (problema secundario)
`Ridge(alpha=1)` con one-hot (k-1 dummies) de sucursal y tela y `Año` escalado: RMSE test 1279 m, R² 0.699
(baseline de la media: RMSE 2332 m). Su error es equivalente al de `LinearRegression` (RMSE CV 1317 vs 1311 m, dentro de
1 error estándar) y es el modelo más regularizado que cumple la regla de 1 SE.

## Versiones
- python: 3.12.10
- pandas: 3.0.5
- numpy: 2.5.2
- scikit-learn: 1.9.0
- xgboost: 3.4.1
- scipy: 1.18.1
- joblib: 1.6.0
- matplotlib: 3.11.1
- seaborn: 0.13.2
