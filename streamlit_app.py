import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from prophet.plot import plot_plotly, plot_components_plotly
from datetime import datetime, timedelta
from pandas.tseries.offsets import DateOffset
import numpy as np

def formatta_numero(numero):
    """Formatta il numero con il punto come separatore delle migliaia."""
    num_str = f"{numero:,}"  # Formatta il numero con la virgola come separatore delle migliaia
    return num_str.replace(',', '.')  # Sostituisce la virgola con il punto

# Esempio di utilizzo della funzione
numero_formattato = formatta_numero(1234567)
print(numero_formattato)  # Output: 1.234.567

# Crea una riga con 3 colonne
col1, col2, col3 = st.columns([1, 7, 1])

# Colonna per l'immagine (a sinistra)
with col1:
    # Assicurati di avere un'immagine nel percorso specificato o passa un URL diretto
    st.image("https://raw.githubusercontent.com/nurdigitalmarketing/previsione-del-traffico-futuro/9cdbf5d19d9132129474936c137bc8de1a67bd35/Nur-simbolo-1080x1080.png", width=80)

# Colonna per il titolo e il testo "by NUR® Digital Marketing" (al centro)
with col2:
    st.title('Previsione del Traffico Futuro')
    st.markdown('###### by NUR® Digital Marketing')

# Colonna vuota (a destra) per spaziatura - opzionale
with col3:
    st.write("")

st.markdown("""
## Introduzione

Questo strumento è stato sviluppato per fornire previsioni sul traffico futuro basandosi sull'export dei dati degli utenti da _Google Analytics_. Attraverso l'utilizzo di modelli di previsione avanzati, facilita la comprensione delle tendenze future basate sui dati storici.

## Funzionamento

Per garantire previsioni accurate, segui i passaggi dettagliati relativi all'origine dei tuoi dati. Ecco come preparare i dati esportati da Google Analytics.
""")

with st.expander("Istruzioni"):
    st.markdown("""
    1. **Esportazione dei dati:**
       - Accedi a Looker Studio.
       - Vai al report per l'esportazione dei dati [cliccando qui](https://lookerstudio.google.com/reporting/12aaee27-8de8-4a87-a62e-deb8a8c4d8f0).
       - Clicca con il tasto destro sulla tabella e poi su "Esporta".
       - Esporta i dati nel formato _Google Fogli_.

    2. **Pulizia dei dati:**
       - Si dovrebbe essere aperto automaticamente in Google Fogli.
       - Rinomina le colonne: quella con le date deve essere rinominata in `Date` e la colonna con i volumi di traffico in `Organic Traffic`.

        _Qui puoi trovare un [esempio](https://drive.google.com/file/d/1v4cqG_v8b85t9A7OImsAINKGUCh_eRey/view?usp=drive_link) di come dovrebbe apparire._

    3. **Caricamento del file:**
       - Utilizza il pulsante di upload per caricare il tuo file CSV pulito.
    """)

st.markdown('---')

# Caricamento del file CSV
uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    traffic = pd.read_csv(uploaded_file)
    if 'Date' not in traffic.columns:
        st.error("Assicurati che il file CSV abbia la colonna 'Date' nel formato 'YYYY-MM-DD'.")
    else:
        traffic.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
        traffic['ds'] = pd.to_datetime(traffic['ds'])

    # Verifica delle colonne
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

        # Mappatura dei mesi in italiano e funzione per formattare le date
        mesi_italiani = {1: 'gennaio', 2: 'febbraio', 3: 'marzo', 4: 'aprile', 5: 'maggio', 6: 'giugno',
                         7: 'luglio', 8: 'agosto', 9: 'settembre', 10: 'ottobre', 11: 'novembre', 12: 'dicembre'}
        
        def formatta_data(data):
            giorno = data.day
            mese = mesi_italiani[data.month]
            anno = data.year
            return f"{giorno} {mese} {anno}"

        # Calcolo delle date e delle somme per il confronto
        fine_ultimo_periodo = forecast['ds'].max()
        inizio_ultimo_periodo = fine_ultimo_periodo - DateOffset(days=365)
        inizio_periodo_precedente = inizio_ultimo_periodo - DateOffset(days=365)
        somma_ultimo_periodo = forecast[(forecast['ds'] > inizio_ultimo_periodo) & (forecast['ds'] <= fine_ultimo_periodo)]['yhat'].sum()
        somma_periodo_precedente = forecast[(forecast['ds'] > inizio_periodo_precedente) & (forecast['ds'] <= inizio_ultimo_periodo)]['yhat'].sum()

        incremento = somma_ultimo_periodo - somma_periodo_precedente
        percentuale_incremento = (incremento / somma_periodo_precedente) * 100
        
        # Messaggio di confronto
        messaggio = f"""
            **Confronto del traffico tra i periodi:**
            - Dal {formatta_data(inizio_periodo_precedente + DateOffset(days=1))} al {formatta_data(inizio_ultimo_periodo)}: {formatta_numero(int(somma_periodo_precedente))} utenti
            - Dal {formatta_data(inizio_ultimo_periodo + DateOffset(days=1))} al {formatta_data(fine_ultimo_periodo)}: {formatta_numero(int(somma_ultimo_periodo))} utenti
            - **{'Incremento' if percentuale_incremento > 0 else 'Decremento'} percentuale:** {"-" if percentuale_incremento < 0 else ""}{abs(percentuale_incremento):.2f}%
        """
        
        if percentuale_incremento > 0:
            st.success(messaggio)
        else:
            st.error(messaggio)

        st.subheader("Previsioni del traffico futuro")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)


        st.download_button(label="Scarica le previsioni in formato CSV",
                           data=forecast.to_csv().encode('utf-8'),
                           file_name='previsioni_traffico_futuro.csv',
                           mime='text/csv')
