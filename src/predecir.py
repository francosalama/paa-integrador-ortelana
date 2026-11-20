"""Predice Format_Reporte para registros nuevos de Ortelana Textil.

Validez: el modelo solo es confiable para pares (Sucursal_ID, Tela_ID) que ya aparecen en el histórico.
Con validación agrupada por par (pares nuevos) su F1 macro cae a ~0,64, por lo que no debe usarse en
producción para sucursales o telas nuevas (ver actividad 13 y README).

Uso:
    python src/predecir.py --entrada registros.csv [--salida predicciones.csv] [--modelo models/modelo_productivo.joblib]
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
MODELO_POR_DEFECTO = RAIZ / 'models' / 'modelo_productivo.joblib'
COLUMNAS = ['Sucursal_ID', 'Tela_ID', 'Año', 'Metros1', 'Metros2']


def main():
    parser = argparse.ArgumentParser(description='Predicción de Format_Reporte (Ortelana Textil)')
    parser.add_argument('--entrada', required=True, help='CSV con las columnas ' + ', '.join(COLUMNAS))
    parser.add_argument('--salida', default=None, help='CSV de salida (si se omite, se imprime en pantalla)')
    parser.add_argument('--modelo', default=str(MODELO_POR_DEFECTO), help='ruta al pipeline serializado')
    args = parser.parse_args()

    datos = pd.read_csv(args.entrada)
    faltan = [c for c in COLUMNAS if c not in datos.columns]
    if faltan:
        sys.exit(f'Error: faltan columnas en la entrada: {faltan}')

    modelo = joblib.load(args.modelo)
    X = datos[COLUMNAS]
    salida = datos.copy()
    salida['Format_Reporte_predicho'] = modelo.predict(X)
    proba = pd.DataFrame(modelo.predict_proba(X), columns=[f'prob_{c}' for c in modelo.classes_])
    salida = pd.concat([salida.reset_index(drop=True), proba.round(4)], axis=1)

    if args.salida:
        salida.to_csv(args.salida, index=False)
        print(f'{len(salida)} predicciones guardadas en {args.salida}')
    else:
        print(salida.to_string(index=False))


if __name__ == '__main__':
    main()
