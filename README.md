# Ortelana Textil — Clasificación de `Format_Reporte`

Trabajo práctico integrador grupal IADS5 (Procesamiento de Aprendizaje Automático).

**Integrantes:** Camila Belén Capua, Franco Salama, Juan Pablo Ruax Debruijn, Lorena Wajchman, Tomas Rossi y Miguel Ángel Tadakuma

## Problema
- **Principal:** clasificación multiclase de `Format_Reporte` (B, C, M, S) con `Sucursal_ID, Tela_ID, Año, Metros1, Metros2`.
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
El CSV de entrada debe tener las columnas `Sucursal_ID, Tela_ID, Año, Metros1, Metros2` (se admiten faltantes).

## Modelo final
Pipeline: `ColumnTransformer` (numéricas: imputación mediana, sin escalado porque Random Forest es invariante a la escala;
categóricas: imputación moda + `OneHotEncoder(handle_unknown='ignore')`) + `RandomForestClassifier(random_state=42)` con {'modelo__max_depth': None, 'modelo__min_samples_leaf': 1, 'modelo__n_estimators': 100}.

| Etapa | F1 macro |
|---|---:|
| Selección: GridSearchCV, CV 5 folds sobre train | 0.9651 |
| Referencia: CV agrupada por par (Sucursal, Tela), pares nuevos | 0.6430 |
| **Final: test (20%, evaluado una sola vez)** | **0.9706** |

Accuracy test: 0.9711

```
              precision    recall  f1-score   support

           B     0.9653    0.9521    0.9586       146
           C     0.9881    0.9651    0.9765       172
           M     0.9503    1.0000    0.9745       153
           S     0.9783    0.9677    0.9730       186

    accuracy                         0.9711       657
   macro avg     0.9705    0.9712    0.9706       657
weighted avg     0.9714    0.9711    0.9711       657
```

## Limitaciones
- El formato es constante para cada par (Sucursal, Tela): la partición aleatoria pedida deja en train registros del
  mismo par que aparecen en test, por lo que el 0.97 mide generalización a nuevos años de pares conocidos.
- Con una CV agrupada por par (`StratifiedGroupKFold`, actividad 13), el mismo pipeline cae a F1 macro 0.643 (y k-NN k=1
  a 0.698): **ambos modelos memorizan el formato fijo de cada par**. El clasificador no generaliza a pares nuevos y
  **no debe pasar a producción** para sucursales o telas nuevas. Para pares conocidos alcanza con una tabla de búsqueda
  del formato histórico; para pares nuevos, el punto de partida sería un modelo sin identificadores (RF solo con
  volumen: 0.839 en la CV agrupada).

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
