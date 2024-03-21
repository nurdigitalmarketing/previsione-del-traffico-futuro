import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from datetime import datetime, timedelta
import io

st.title('Previsione del Traffico Futuro')

# Caricamento del file CSV
uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    # Leggere il contenuto del file come stringa per verificare l'intestazione
    content = uploaded_file.getvalue().decode("utf-8")
    
    # Rilevare il formato del file basandosi sull'intestazione
    if "Esportazione CSV rapporto" in content:
        # Analytics format
        # Estrarre le date di inizio e fine dall'intestazione
        lines = content.split("\n")
        start_date_str = lines[5].split(": ")[1]
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        
        # Convertire il CSV in DataFrame, saltando le righe di intestazione
        df = pd.read_csv(io.StringIO(content), skiprows=9)
        
        # Calcolare la data effettiva per ogni riga basandosi sull'indice settimanale
        df['Date'] = df['Date'].apply(lambda x: start_date + timedelta(weeks=int(x)))
    else:
        # Ahrefs format o altro formato compatibile
        df = pd.read_csv(io.StringIO(content))
        # Assumiamo che il file Ahrefs sia già nel formato corretto
        
    # Rinominare le colonne per la compatibilità con Prophet
    df.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
    
    
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
