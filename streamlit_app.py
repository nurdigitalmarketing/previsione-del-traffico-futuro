import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

st.title('Previsione del Traffico Futuro')

# Selettore della fonte dei dati
data_source = st.selectbox('Seleziona la fonte dei dati:', ['Google Analytics', 'Ahrefs'])

uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")

if uploaded_file is not None:
    if data_source == 'Ahrefs':
        # Caricamento e preparazione dei dati Ahrefs
        traffic = pd.read_csv(uploaded_file)
        traffic.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
    elif data_source == 'Google Analytics':
        # Caricamento e preparazione dei dati Google Analytics
        # Saltiamo le righe di intestazione/commento e rinominiamo le colonne
        traffic = pd.read_csv(uploaded_file, skiprows=7)
        traffic.rename(columns={'Ennesima settimana': 'ds', 'Utenti': 'y'}, inplace=True)
        # Qui assumiamo che 'ds' sia un contatore di settimane e lo convertiamo in data
        # La logica di conversione può variare in base al contesto specifico
        traffic['ds'] = pd.to_datetime('2023-03-20') + pd.to_timedelta(traffic['ds'].astype(int) * 7, unit='d')
    
    st.write("Anteprima dei dati caricati:")
    st.write(traffic.head())

    # Definizione delle festività e degli eventi speciali
    updates = pd.DataFrame({
        'holiday': 'Google Update',
        'ds': pd.to_datetime([
            '2015-07-17', '2016-01-08', '2016-09-27', '2017-03-08', '2017-07-09',
            '2018-03-08', '2018-04-17', '2018-08-01', '2019-03-12', '2019-06-03',
            '2019-09-24', '2019-10-25', '2019-12-09', '2020-01-13', '2020-05-04',
            '2020-12-03', '2021-06-02', '2021-07-01', '2021-11-17', '2022-05-25',
            '2023-09-15', '2023-10-05', '2023-11-02', '2023-11-08', '2024-03-05',
        ]),
        'lower_window': 0,
        'upper_window': 14,
    })
    
    m = Prophet(holidays=updates)
    m.fit(traffic)

    # Creazione del dataframe per le future previsioni
    future = m.make_future_dataframe(periods=365)
    
    # Generazione delle previsioni
    forecast = m.predict(future)
    
    # Visualizzazione dei risultati
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Opzione per scaricare le previsioni come CSV
    st.download_button(label="Scarica le previsioni in formato CSV",
                       data=forecast.to_csv().encode('utf-8'),
                       file_name='previsioni_traffico_futuro.csv',
                       mime='text/csv')
