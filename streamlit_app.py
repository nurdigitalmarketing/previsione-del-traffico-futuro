import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta

def formatta_numero(numero):
    """Formatta il numero con il punto come separatore delle migliaia.

    Args:
        numero (int): Il numero da formattare.

    Returns:
        str: Il numero formattato con il punto come separatore delle migliaia.
    """
    num_str = f"{numero:,}"  # Formatta il numero con la virgola come separatore delle migliaia
    return num_str.replace(',', '.')  # Sostituisce la virgola con il punto

    st.title('Previsione del Traffico Futuro')
    
    st.markdown(
    """
    ## Introduzione
    
    Questo strumento è stato sviluppato per fornire previsioni sul traffico futuro basandosi sull'export dei dati degli utenti da _Google Analytics_ o sui dati di ricerca organica da _Ahrefs_ o _Semrush_. Attraverso l'utilizzo di modelli di previsione avanzati, facilita la comprensione delle tendenze future basate sui dati storici.
    
    ## Funzionamento
    
    Per garantire previsioni accurate, segui i passaggi dettagliati relativi all'origine dei tuoi dati. Ecco come preparare i dati esportati da Google Analytics, Ahrefs o Semrush.
    """)
    
    with st.expander("Da Google Analytics"):
        st.markdown(
        """
        1. **Esportazione dei dati:**
           - Accedi a Google Analytics.
           - Vai alla sezione "Rapporti" e seleziona le metriche di traffico che desideri analizzare (es. utenti, sessioni).
           - Esporta i dati nel formato CSV.
    
        2. **Pulizia dei dati:**
           - Apri il file CSV su Google Fogli.
           - Elimina tutte le righe dalla 1 alla 8 e dalla riga dopo l'ultima "Ennesima settimana" del primo blocco di data e utenti.
           - Fatto ciò, assicurati che le colonne siano nominate correttamente: la colonna con le date deve essere rinominata in `Date` e la colonna con i volumi di traffico in `Organic Traffic`.
    
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
           - Apri il file esportato con Google Fogli.
           - Rinomina la colonna con le date in `Date` e quella con i volumi di traffico in `Organic Traffic`.
           - Rimuovi le righe e le colonne non pertinenti che non contengono dati di traffico o date.
    
           _Qui puoi trovare un [esempio](https://drive.google.com/file/d/1v4cqG_v8b85t9A7OImsAINKGUCh_eRey/view?usp=sharing) di come dovrebbe apparire._
    
        3. **Caricamento del file:**
           - Carica il file CSV pulito attraverso l'interfaccia di caricamento fornita dallo strumento.
        """
        )
    
    with st.expander("Da Semrush"):
        st.markdown(
        """
        1. **Esportazione dei dati:**
           - Accedi a Semrush e vai alla sezione di _panoramica dominio_ per il tuo sito.
           - Seleziona il periodo di tempo desiderato (i.e. 2A) e esporta i dati relativi al traffico di ricerca organica.
    
        2. **Pulizia dei dati:**
           - Apri il file esportato con Google Fogli.
           - Rimuovi le colonne A, B, C e D e le colonne dalla 3 alla 9 comprese.
           - Rinomina la colonna `Summary` in `Date` e la cella sotto `Summary` in `Organic Traffic`.
           - Esporta il file in CSV ed importalo nello strumento [CSV rows and columns converter](https://onlinecsvtools.com/convert-csv-rows-to-columns) cliccando su _import from file_ sotto _input csv_.
           - A questo punto, sotto _output csv_, copia il risultato ottenuto.
           - Apri un nuovo foglio su Google Fogli e incolla il risultato ottenuto. Vai su `Dati` e clicca su `Suddividi il testo in colonne`.
           - Esporta il file in formato CSV.
    
           _Qui puoi trovare un [esempio](https://drive.google.com/file/d/1ZkfuqbHcxQhm5zX8L_nKf0OPWAYTg3Hr/view?usp=sharing) di come dovrebbe apparire._
    
        3. **Caricamento del file:**
           - Carica il file CSV pulito attraverso l'interfaccia di caricamento fornita dallo strumento.
        """
        )
    
    st.markdown ('---')

def calcola_incremento_traffico(data_inizio, data_fine, traffic):
    # Conversione delle date di inizio e fine dei dati storici in datetime
    data_inizio_datetime = datetime.strptime(data_inizio, '%Y%m%d')
    data_fine_datetime = datetime.strptime(data_fine, '%Y%m%d')
    
    # Definizione degli eventi speciali (aggiornamenti Google)
    updates = pd.DataFrame({
      'holiday': 'Google Update',
      'ds': pd.to_datetime(['2015-07-17', '2016-01-08', '2016-09-27', '2017-03-08', '2017-07-09', 
                            '2018-03-08', '2018-04-17', '2018-08-01', '2019-03-12', '2019-06-03', 
                            '2019-09-24', '2019-10-25', '2019-12-09', '2020-01-13', '2020-05-04', 
                            '2020-12-03', '2021-06-02', '2021-07-01', '2021-11-17', '2022-05-25', 
                            '2023-09-15', '2023-10-05', '2023-11-02', '2023-11-08', '2024-03-05']),
      'lower_window': 0,
      'upper_window': 1,
    })
    
    m = Prophet(holidays=updates)
    m.fit(traffic)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    
    # Calcolo e visualizzazione dell'incremento previsto del traffico
    traffic_dati_storici = forecast[(forecast['ds'] >= data_inizio_datetime) & (forecast['ds'] <= data_fine_datetime)]['yhat'].sum()
    traffic_dati_previsione = forecast[forecast['ds'] > data_fine_datetime]['yhat'].sum()
    incremento = traffic_dati_previsione - traffic_dati_storici
    percentuale_incremento = (incremento / traffic_dati_storici) * 100 if traffic_dati_storici != 0 else float('inf')

    # Visualizzazione dei risultati
    st.info(f"""
        **Stima dell'aumento del traffico con il metodo NUR®:**
        - Si stima un aumento di traffico da {formatta_numero(int(traffic_dati_storici))} utenti nel periodo dei dati storici a {formatta_numero(int(traffic_dati_previsione))} utenti nell'ultimo mese del periodo di previsione.
        - **Incremento percentuale:** {percentuale_incremento:.2f}%
    """)

    # Visualizzazione delle previsioni
    st.subheader("Previsioni del traffico futuro")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Opzione per scaricare le previsioni in formato CSV
    st.download_button(label="Scarica le previsioni in formato CSV",
                       data=forecast.to_csv().encode('utf-8'),
                       file_name='previsioni_traffico_futuro.csv',
                       mime='text/csv')
