"""Predice Format_Reporte para registros nuevos de Ortelana Textil.

El modelo usa solo Año, Metros1 y Metros2: no usa Sucursal_ID ni Tela_ID. Para un par (Sucursal, Tela) ya
conocido alcanza con su formato histórico (es constante por par); el modelo está pensado para pares nuevos,
donde su F1 macro estimado con validación agrupada por par es ~0,89 (ver actividades 13-15 y README).
Si la entrada trae otras columnas (por ejemplo Sucursal_ID o Tela_ID), se conservan en la salida pero no se usan.

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
COLUMNAS = ['Año', 'Metros1', 'Metros2']


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
