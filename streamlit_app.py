# Import delle librerie necessarie
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta

# Definizione delle funzioni ottimizzate

def formatta_numero_optimizzato(numero):
    """Formatta il numero con il punto come separatore delle migliaia in modo più diretto."""
    return "{:,}".format(numero).replace(',', '.')

def formatta_data_optimizzata(data):
    """Formatta la data in formato italiano più efficientemente."""
    return data.strftime('%d %B %Y')

def prepara_dati_e_modello(traffic_df):
    """Prepara i dati e addestra il modello Prophet."""
    # Preparazione del DataFrame per Prophet
    traffic_df.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
    traffic_df['ds'] = pd.to_datetime(traffic_df['ds'])

    # Definizione delle date degli aggiornamenti Google come festività
    updates = pd.DataFrame({
        'holiday': 'Google Update',
        'ds': pd.to_datetime([
            '2015-07-17', '2016-01-08', '2016-09-27', '2017-03-08', '2017-07-09',
            '2018-03-08', '2018-04-17', '2018-08-01', '2019-03-12', '2019-06-03',
            '2019-09-24', '2019-10-25', '2019-12-09', '2020-01-13', '2020-05-04',
            '2020-12-03', '2021-06-02', '2021-07-01', '2021-11-17', '2022-05-25',
            '2023-09-15', '2023-10-05', '2023-11-02', '2023-11-08', '2024-03-05'
        ]),
        'lower_window': 0,
        'upper_window': 1,
    })

    # Addestramento del modello
    m = Prophet(holidays=updates)
    m.fit(traffic_df)

    return m

def effettua_previsioni(modello, periodi=365):
    """Genera previsioni future con il modello addestrato."""
    future = modello.make_future_dataframe(periods=periodi)
    forecast = modello.predict(future)
    return forecast

# Esempio di utilizzo delle funzioni ottimizzate
# Nota: Le chiamate alle funzioni saranno decommentate e utilizzate nel codice finale per dimostrare l'utilizzo.

    traffic_df = pd.read_csv('percorso_del_file_csv')
    modello = prepara_dati_e_modello(traffic_df)
    previsioni = effettua_previsioni(modello)

# Nota: Questo è un esempio di come potrebbero essere utilizzate le funzioni ottimizzate. 
# Le funzioni di lettura del file CSV, visualizzazione dei risultati e gestione dell'UI di Streamlit 
# devono essere integrate nel contesto dell'applicazione Streamlit completa.

# Commento: Le chiamate funzionali e la gestione del flusso di lavoro dell'applicazione Streamlit
# saranno adeguatamente integrate e gestite nel codice completo dell'applicazione.
