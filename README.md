# Ortelana Textil — Clasificación de `Format_Reporte`

Trabajo práctico integrador grupal IADS5 (Procesamiento de Aprendizaje Automático).

**Integrantes:** Camila Belén Capua, Franco Salama, Juan Pablo Ruax Debruijn, Lorena Wajchman, Tomas Rossi y Miguel Ángel Tadakuma

## Problema
- **Principal:** clasificación multiclase de `Format_Reporte` (B, C, M, S) con `Sucursal_ID, Tela_ID, Año, Metros1, Metros2`.
- **Secundario (regresión):** `Metros2` con `Sucursal_ID, Tela_ID, Año, Metros1`.
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
Pipeline: `ColumnTransformer` (numéricas: imputación mediana + `StandardScaler`; categóricas: imputación moda +
`OneHotEncoder(handle_unknown='ignore')`) + `RandomForestClassifier(random_state=42)` con {'modelo__max_depth': None, 'modelo__min_samples_leaf': 1, 'modelo__n_estimators': 100}.

| Etapa | F1 macro |
|---|---:|
| Selección: GridSearchCV, CV 5 folds sobre train | 0.9640 |
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
  mismo par que aparecen en test, por lo que el resultado mide generalización a nuevos años de pares conocidos,
  no a sucursales o telas nuevas.
- La regresión de `Metros2` no usa el formato, por lo que mezcla períodos de distinta duración (un mes, un bimestre, un semestre o un ciclo).

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
