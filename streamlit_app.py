import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta

st.title('Previsione del Traffico Futuro')

# Campo di selezione per l'origine dei dati
origine_dati = st.selectbox("Seleziona l'origine dei dati:", ['Scegli...', 'Google Analytics', 'Ahrefs'])

# Input per le date di inizio e fine, visibili solo se l'origine dei dati è Google Analytics
if origine_dati == 'Google Analytics':
    data_inizio = st.text_input('Inserisci la data di inizio (YYYYMMDD):', '20230320')
    data_fine = st.text_input('Inserisci la data di fine (YYYYMMDD):', '20240320')

# Caricamento del file CSV
uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    if origine_dati == 'Google Analytics':
        traffic = pd.read_csv(uploaded_file)
        start_date = datetime.strptime(data_inizio, '%Y%m%d')
        traffic['ds'] = traffic['Date'].apply(lambda x: start_date + timedelta(weeks=int(x)))
        traffic.rename(columns={'Organic Traffic': 'y'}, inplace=True)
    elif origine_dati == 'Ahrefs':
        traffic = pd.read_csv(uploaded_file)
        if 'Date' not in traffic.columns:
            st.error("Assicurati che il file CSV abbia la colonna 'Date' nel formato 'YYYY-MM-DD'.")
        else:
            traffic.rename(columns={'Date': 'ds', 'Traffic': 'y'}, inplace=True)
            traffic['ds'] = pd.to_datetime(traffic['ds'])

    if origine_dati in ['Google Analytics', 'Ahrefs']:
        st.write("Anteprima dei dati caricati:")
        st.write(traffic.head())

        # Controllo che il DataFrame abbia le colonne necessarie per Prophet
        if not {"ds", "y"}.issubset(traffic.columns):
            st.error("Assicurati che il DataFrame abbia le colonne 'ds' per la data e 'y' per la variabile target.")
        else:
            # Definizione delle festività e degli eventi speciali
        # Definizione delle festività e degli eventi speciali
        updates = pd.DataFrame({
            'holiday': 'Google Update',
            'ds': pd.to_datetime(['2015-07-17', '2016-01-08',
                                  '2016-09-27', '2017-03-08', '2017-07-09', '2018-03-08', '2018-04-17',
                                  '2018-08-01', '2019-03-12', '2019-06-03', '2019-09-24', '2019-10-25',
                                  '2019-12-09', '2020-01-13', '2020-05-04', '2020-12-03', '2021-06-02',
                                  '2021-07-01', '2021-11-17', '2022-05-25', '2023-09-15', '2023-10-05',
                                  '2023-11-02', '2023-11-08', '2024-03-05']),
            'lower_window': 0,
            'upper_window': 14,
        })

            # Addestramento del modello con le festività
            m = Prophet(holidays=holidays)
            m.fit(traffic)

            # Creazione del dataframe per le future previsioni
            future = m.make_future_dataframe(periods=365)

            # Generazione delle previsioni
            forecast = m.predict(future)

            # Visualizzazione dei risultati
            st.subheader("Previsioni del traffico futuro")
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            
            st.subheader("Componenti del forecast")
            fig2 = plot_components_plotly(m, forecast)
            st.plotly_chart(fig2)
            
            # Opzione per scaricare le previsioni come CSV
            st.download_button(label="Scarica le previsioni in formato CSV",
                               data=forecast.to_csv().encode('utf-8'),
                               file_name='previsioni_traffico_futuro.csv',
                               mime='text/csv')
