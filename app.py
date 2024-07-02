from flask import Flask, render_template, request, send_file, make_response
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.io as pio
import base64
import traceback
import logging

app = Flask(__name__)

# Nombres de las nuevas variables
target_vars = [
    'Corriente congelacion F1',
    'Corriente congelacion F2',
    'Corriente congelacion F3',
    'Corriente resistencia deshielo'
]

# Cargar los modelos y los escaladores
models = {}
scalers = {}
try:
    models = {var: joblib.load(f'models/modelo_rf_{var}.pkl') for var in target_vars}
    scalers = {var: joblib.load(f'models/scaler_rf_{var}.pkl') for var in target_vars}
except Exception as e:
    print(f"Error cargando modelos o escaladores: {e}")
    print(traceback.format_exc())

def predict_rf_hourly(models, scalers, target_date, steps=24):
    target_date = pd.to_datetime(target_date)
    prediction_dates = pd.date_range(start=target_date, periods=steps, freq='H')
    prediction_hours = prediction_dates.hour
    prediction_days = prediction_dates.dayofyear
    prediction_dayofweek = prediction_dates.dayofweek
    X_pred = pd.DataFrame({
        'hour': prediction_hours,
        'day': prediction_days,
        'dayofweek': prediction_dayofweek
    })

    predictions = {}
    for var, model_rf in models.items():
        forecast = model_rf.predict(X_pred)
        forecast = scalers[var].inverse_transform(forecast.reshape(-1, 1)).flatten()
        predictions[var] = np.round(forecast, 2)  # Redondear predicciones a 2 decimales

    prediction_df = pd.DataFrame(predictions, index=prediction_dates)
    prediction_df.index.name = 'Time'
    return prediction_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            target_date = request.form['target_date']
            
            predictions_df = predict_rf_hourly(models, scalers, target_date)

            # Crear el primer gráfico interactivo con Plotly (Corrientes de las tres fases)
            fig1 = go.Figure()
            for var in ['Corriente congelacion F1', 'Corriente congelacion F2', 'Corriente congelacion F3']:
                fig1.add_trace(go.Scatter(
                    x=predictions_df.index,
                    y=predictions_df[var],
                    mode='lines+markers',
                    name=f'{var} Predicha'
                ))
            fig1.update_layout(
                title=f'Predicción de Corrientes de las Tres Fases para el {target_date}',
                xaxis_title='Hora',
                yaxis_title='Corriente',
                hovermode='x'
            )
            plot_html1 = pio.to_html(fig1, full_html=False)

            # Crear el segundo gráfico interactivo con Plotly (Corriente resistencia deshielo)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Corriente resistencia deshielo'],
                mode='lines+markers',
                name='Corriente resistencia deshielo Predicha'
            ))
            fig2.update_layout(
                title=f'Predicción de Corriente Resistencia Deshielo para el {target_date}',
                xaxis_title='Hora',
                yaxis_title='Corriente',
                hovermode='x'
            )
            plot_html2 = pio.to_html(fig2, full_html=False)

            # Calcular desbalance de corriente
            predictions_df['PromedioFases'] = (predictions_df['Corriente congelacion F1'] + predictions_df['Corriente congelacion F2'] + predictions_df['Corriente congelacion F3']) / 3
            predictions_df['Desbalance1'] = abs(predictions_df['Corriente congelacion F1'] - predictions_df['PromedioFases']) / predictions_df['PromedioFases'] * 100  # Convertir a porcentaje
            predictions_df['Desbalance2'] = abs(predictions_df['Corriente congelacion F2'] - predictions_df['PromedioFases']) / predictions_df['PromedioFases'] * 100  # Convertir a porcentaje
            predictions_df['Desbalance3'] = abs(predictions_df['Corriente congelacion F3'] - predictions_df['PromedioFases']) / predictions_df['PromedioFases'] * 100  # Convertir a porcentaje
            predictions_df['Desbalance Corriente'] = predictions_df[['Desbalance1', 'Desbalance2', 'Desbalance3']].max(axis=1)

            # Crear el tercer gráfico interactivo con Plotly (Desbalance de Corriente)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=predictions_df.index,
                y=predictions_df['Desbalance Corriente'],
                mode='lines+markers',
                name='Desbalance Corriente Predicha'
            ))
            fig3.update_layout(
                title=f'Predicción de Desbalance de Corriente para el {target_date}',
                xaxis_title='Hora',
                yaxis_title='Porcentaje (%)',
                hovermode='x'
            )
            plot_html3 = pio.to_html(fig3, full_html=False)

            # Renderizar la tabla y los gráficos en la página
            table_html = predictions_df.to_html(classes='data', header="true")
            return render_template('index.html', plot_html1=plot_html1, plot_html2=plot_html2, plot_html3=plot_html3, table_html=table_html)
        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Error procesando la solicitud: {e}")
            print(error_message)
            return f"Error: {error_message}", 500
        
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

