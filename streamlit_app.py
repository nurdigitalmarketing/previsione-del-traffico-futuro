import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
import numpy as np

def formatta_numero(numero):
    """Formatta il numero con il punto come separatore delle migliaia.
    
    Args:
        numero (int): Il numero da formattare.
    
    Returns:
        str: Il numero formattato con il punto come separatore delle migliaia.
    """
    num_str = f"{numero:,}"
    return num_str.replace(',', '.')

st.title('Previsione del Traffico Futuro')

st.markdown(
"""
## Introduzione

Questo strumento è stato sviluppato per fornire previsioni sul traffico futuro basandosi sull'export dei dati degli utenti da Google Analytics o sui dati di ricerca organica da Ahrefs/Semrush. Attraverso l'utilizzo di modelli di previsione avanzati, facilita la comprensione delle tendenze future basate sui dati storici.

## Funzionamento

Per garantire previsioni accurate, segui i passaggi dettagliati relativi all'origine dei tuoi dati. Ecco come preparare i dati esportati da Google Analytics e Ahrefs/Semrush.
""")

with st.expander("Da Google Analytics"):
    st.markdown(
    """
    1. **Esportazione dei dati:**
       - Accedi a Google Analytics.
       - Vai alla sezione "Rapporti" e seleziona le metriche di traffico che desideri analizzare (es. utenti, sessioni).
       - Esporta i dati nel formato CSV.

    2. **Pulizia dei dati:**
       - Apri il file CSV con un editor di fogli di calcolo (es. Excel, Google Sheets).
       - Assicurati che le colonne siano nominate correttamente: la colonna con le date deve essere rinominata in `Date` e la colonna con i volumi di traffico in `Organic Traffic`.
       - Elimina eventuali righe o colonne non necessarie che non contengono dati relativi al traffico o alle date.

        _Qui puoi trovare un [esempio](https://drive.google.com/file/d/1v4ZpiG8Kijwn1uRm02S1yMRQkWH1G4ov/view?usp=sharing) di come dovrebbe apparire._

    3. **Selezione del range di date:**
       - Nello strumento, specifica il range di date che vuoi analizzare inserendo la data di inizio e di fine nel formato `YYYYMMDD`.
       - Questo aiuterà lo strumento a calibrare correttamente le previsioni sul periodo di interesse.

    4. **Caricamento del file:**
       - Utilizza il pulsante di upload per caricare il tuo file CSV pulito.
    """
    )

with st.expander("Da Ahrefs"):
    st.markdown(
    """
    1. **Esportazione dei dati:**
       - Accedi ad Ahrefs e vai alla sezione di ricerca organica per il tuo sito.
       - Seleziona il periodo di tempo desiderato e esporta i dati relativi al traffico di ricerca organica.

    2. **Pulizia dei dati:**
       - Apri il file esportato con un software di fogli di calcolo.
       - Rinomina la colonna con le date in `Date` e quella con i volumi di traffico in `Traffic`.
       - Rimuovi le righe e le colonne non pertinenti che non contengono dati di traffico o date.

       _Qui puoi trovare un [esempio](https://drive.google.com/file/d/1v4cqG_v8b85t9A7OImsAINKGUCh_eRey/view?usp=sharing) di come dovrebbe apparire._

    3. **Caricamento del file:**
       - Carica il file CSV pulito attraverso l'interfaccia di caricamento fornita dallo strumento.
    """
    )

st.markdown ('---')

# Campo di selezione per l'origine dei dati
origine_dati = st.selectbox("Seleziona l'origine dei dati:", ['Scegli...', 'Google Analytics', 'Ahrefs', 'Semrush'])

# Input per le date di inizio e fine, visibili solo se l'origine dei dati è Google Analytics
if origine_dati == 'Google Analytics':
    data_inizio = st.text_input('Inserisci la data di inizio (YYYYMMDD):', '20230320')
    data_fine = st.text_input('Inserisci la data di fine (YYYYMMDD):', '20240320')

# Caricamento del file CSV
uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    traffic = pd.read_csv(uploaded_file)
    if origine_dati in ['Google Analytics', 'Ahrefs', 'Semrush']:
        traffic['ds'] = pd.to_datetime(traffic['ds'])
        traffic.rename(columns={'Organic Traffic': 'y'}, inplace=True)

        # Verifica delle colonne
        if not {"ds", "y"}.issubset(traffic.columns):
            st.error("Assicurati che il DataFrame abbia le colonne 'ds' per la data e 'y' per la variabile target.")
        else:
            # Preparazione dei dati e addestramento del modello
            m = Prophet()
            m.fit(traffic)

            # Assumendo che `traffic` sia il DataFrame contenente i tuoi dati di input
            
            # Calcolo della lunghezza del periodo di input in giorni
            lunghezza_periodo_input = (traffic['ds'].max() - traffic['ds'].min()).days
            
            # Creazione di un DataFrame futuro per le previsioni che corrisponde alla lunghezza del periodo di input
            future = m.make_future_dataframe(periods=lunghezza_periodo_input)
            
            # Esecuzione delle previsioni
            forecast = m.predict(future)
            
            # Calcolo della lunghezza del periodo di input
            lunghezza_periodo_input = (traffic['ds'].max() - traffic['ds'].min()).days + 1
            future = m.make_future_dataframe(periods=lunghezza_periodo_input)
            forecast = m.predict(future)

            # Calcolo dell'incremento del traffico per il periodo di input
            traffic_input = forecast[(forecast['ds'] >= data_inizio_input) & (forecast['ds'] <= data_fine_input)]
            incremento_input = traffic_input['yhat'].iloc[-1] - traffic_input['yhat'].iloc[0]
            percentuale_incremento_input = (incremento_input / traffic_input['yhat'].iloc[0]) * 100
            
            # Calcolo dell'incremento del traffico per il periodo di previsione
            traffic_previsione = forecast[(forecast['ds'] >= data_inizio_previsione) & (forecast['ds'] <= data_fine_previsione)]
            incremento_previsione = traffic_previsione['yhat'].iloc[-1] - traffic_previsione['yhat'].iloc[0]
            percentuale_incremento_previsione = (incremento_previsione / traffic_previsione['yhat'].iloc[0]) * 100

            st.info(f"""
                **Stima dell'aumento del traffico con il metodo NUR®:**
                - Incremento previsto: {formatta_numero(int(incremento))} utenti
                - **Incremento percentuale:** {percentuale_incremento:.2f}%
            """)


            # Visualizzazione delle previsioni
            st.subheader("Previsioni del traffico futuro")
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            
            # Opzione per scaricare le previsioni
            st.download_button(label="Scarica le previsioni in formato CSV", data=forecast.to_csv().encode('utf-8'), file_name='previsioni_traffico_futuro.csv', mime='text/csv')
