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

def load_model_and_scaler(var):
    try:
        model = joblib.load(f'models/modelo_rf_{var}.pkl')
        scaler = joblib.load(f'models/scaler_rf_{var}.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error cargando modelo o escalador: {e}")
        print(traceback.format_exc())
        return None, None

def predict_rf_hourly(model, scaler, target_date, steps=24):
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

    forecast = model.predict(X_pred)
    forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
    return np.round(forecast, 2), prediction_dates

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            target_date = request.form['target_date']
            predictions = {}
            for var in target_vars:
                model, scaler = load_model_and_scaler(var)
                if model and scaler:
                    predictions[var], prediction_dates = predict_rf_hourly(model, scaler, target_date)

            predictions_df = pd.DataFrame(predictions, index=prediction_dates)
            predictions_df.index.name = 'Time'

            # Crear gráficos y tabla aquí como en tu código original

            table_html = predictions_df.to_html(classes='data', header="true")
            return render_template('index.html', plot_html1=plot_html1, plot_html2=plot_html2, plot_html3=plot_html3, table_html=table_html)
        except Exception as e:
            error_message = traceback.format_exc()
            print(f"Error procesando la solicitud: {e}")
            print(error_message)
            return f"Error: {error_message}", 500
        
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



