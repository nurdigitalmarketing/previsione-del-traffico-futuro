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
    num_str = f"{numero:,}"  # Formatta il numero con la virgola come separatore delle migliaia
    return num_str.replace(',', '.')  # Sostituisce la virgola con il punto

# Esempio di utilizzo della funzione
numero_formattato = formatta_numero(1234567)
print(numero_formattato)  # Output: 1.234.567


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

# Campo di selezione per l'origine dei dati
origine_dati = st.selectbox("1- Seleziona l'origine dei dati:", ['Scegli...', 'Google Analytics', 'Ahrefs', 'Semrush'])

# Input per le date di inizio e fine, visibili solo se l'origine dei dati è Google Analytics
if origine_dati == 'Google Analytics':
    data_inizio = st.text_input('Inserisci la data di inizio e premi "Invio" (YYYYMMDD):')
    data_fine = st.text_input('Inserisci la data di fine e premi "Invio" (YYYYMMDD):')

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
            traffic.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
            traffic['ds'] = pd.to_datetime(traffic['ds'])
    elif origine_dati == 'Semrush':  # Gestione del caso Semrush
        traffic = pd.read_csv(uploaded_file)
        # Assicurati che il file CSV abbia una colonna 'Date' nel formato 'YYYY-MM'
        if 'Date' not in traffic.columns or 'Organic Traffic' not in traffic.columns:
            st.error("Assicurati che il file CSV abbia le colonne 'Date' nel formato 'YYYY-MM' e 'Organic Traffic'.")
        else:
            # Conversione del formato della data da 'YYYY-MM' a datetime, aggiungendo '-01' per completare la data
            traffic['ds'] = pd.to_datetime(traffic['Date'] + '-01')
            traffic.rename(columns={'Organic Traffic': 'y'}, inplace=True)

    # Verifica delle colonne per tutti i casi
    if origine_dati in ['Google Analytics', 'Ahrefs', 'Semrush']:
        if not {"ds", "y"}.issubset(traffic.columns):
            st.error("Assicurati che il DataFrame abbia le colonne 'ds' per la data e 'y' per la variabile target.")
        else:
            updates = pd.DataFrame({
              'holiday': 'Google Update',
              'ds': pd.to_datetime(['2015-07-17', '2016-01-08',
                                    '2016-09-27', '2017-03-08', '2017-07-09', '2018-03-08', '2018-04-17',
                                    '2018-08-01', '2019-03-12', '2019-06-03', '2019-09-24', '2019-10-25',
                                    '2019-12-09', '2020-01-13', '2020-05-04', '2020-12-03', '2021-06-02',
                                    '2021-07-01', '2021-11-17', '2022-05-25', '2023-09-15', '2023-10-05',
                                    '2023-11-02', '2023-11-08', '2024-03-05']),
              'lower_window': 0,
              'upper_window': 1,
            })

            m = Prophet(holidays=updates)
            m.fit(traffic)
            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            # Calcolo e visualizzazione dell'incremento previsto del traffico
            inizio_previsioni = forecast['ds'].min()
            fine_previsioni = forecast['ds'].max()
            traffic_primo_mese = forecast[forecast['ds'] == inizio_previsioni]['yhat'].sum()
            traffic_ultimo_mese = forecast[forecast['ds'] == fine_previsioni]['yhat'].sum()
            incremento = traffic_ultimo_mese - traffic_primo_mese
            percentuale_incremento = (incremento / traffic_primo_mese) * 100

            st.info(f"""
                **Stima dell'aumento del traffico con il metodo NUR®:**
                - Si stima un aumento di traffico da {formatta_numero(int(traffic_primo_mese))} utenti nel primo mese a {formatta_numero(int(traffic_ultimo_mese))} utenti nell'ultimo mese del periodo di previsione.
                - **Incremento percentuale:** {percentuale_incremento:.2f}%
            """)
            
            ## st.write("Anteprima dei dati caricati:")
            ## st.write(traffic.head())

            st.subheader("Previsioni del traffico futuro")
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)
            
            ## st.subheader("Componenti del forecast")
            ## fig2 = plot_components_plotly(m, forecast)
            ## st.plotly_chart(fig2)
            
            st.download_button(label="Scarica le previsioni in formato CSV",
                               data=forecast.to_csv().encode('utf-8'),
                               file_name='previsioni_traffico_futuro.csv',
                               mime='text/csv')
