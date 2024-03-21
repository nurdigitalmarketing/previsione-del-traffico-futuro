import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
import numpy as np
import locale

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

st.markdown("""
## Introduzione

Questo strumento è stato sviluppato per fornire previsioni sul traffico futuro basandosi sull'export dei dati degli utenti da Google Analytics, sui dati di ricerca organica da Ahrefs, e ora anche dai dati di Semrush. Attraverso l'utilizzo di modelli di previsione avanzati, facilita la comprensione delle tendenze future basate sui dati storici.
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

st.markdown('---')

# Campo di selezione per l'origine dei dati
origine_dati = st.selectbox("Seleziona l'origine dei dati:", ['Scegli...', 'Google Analytics', 'Ahrefs', 'Semrush'])

if origine_dati != 'Scegli...':
    uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
    if uploaded_file is not None:
        if origine_dati in ['Google Analytics', 'Ahrefs', 'Semrush']:
            # Gestione dei dati specifica per ogni origine dati
            # [Inserire qui la logica di lettura e preparazione dei dati specifica per ogni origine]
            pass

        # Qui viene utilizzata una lista di festività come esempio; adattala secondo le tue necessità
        holidays = pd.DataFrame({
          'holiday': 'special_event',
          'ds': pd.to_datetime(['2022-01-01', '2022-12-25']),
          'lower_window': 0,
          'upper_window': 1,
        })

        m = Prophet(holidays=holidays)
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

        st.subheader("Previsioni del traffico futuro")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)
        
        st.download_button(label="Scarica le previsioni in formato CSV",
                           data=forecast.to_csv(index=False).encode('utf-8'),
                           file_name='previsioni_traffico_futuro.csv',
                           mime='text/csv')
