import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import numpy as np
import plotly.graph_objects as go
import os

# Funzione per formattare i numeri
def formatta_numero(numero):
    """Formatta il numero con il punto come separatore delle migliaia."""
    num_str = f"{numero:,}"
    return num_str.replace(',', '.')

# Funzione per caricare e pulire i dati
def carica_dati(uploaded_file):
    try:
        traffic = pd.read_csv(uploaded_file)
        if 'Date' not in traffic.columns:
            st.error("Assicurati che il file CSV abbia la colonna 'Date' nel formato 'YYYY-MM-DD'.")
            return None
        traffic.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
        traffic['ds'] = pd.to_datetime(traffic['ds'])
        return traffic
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {e}")
        return None

# Funzione per creare il modello di previsione
def crea_modello(traffic):
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
    return m

# Funzione per calcolare il confronto del traffico
def calcola_confronto(forecast):
    fine_ultimo_periodo = forecast['ds'].max()
    inizio_ultimo_periodo = fine_ultimo_periodo - DateOffset(days=365)
    inizio_periodo_precedente = inizio_ultimo_periodo - DateOffset(days=365)
    
    somma_ultimo_periodo = forecast[(forecast['ds'] > inizio_ultimo_periodo) & (forecast['ds'] <= fine_ultimo_periodo)]['yhat'].sum()
    somma_periodo_precedente = forecast[(forecast['ds'] > inizio_periodo_precedente) & (forecast['ds'] <= inizio_ultimo_periodo)]['yhat'].sum()
    
    incremento = somma_ultimo_periodo - somma_periodo_precedente
    percentuale_incremento = (incremento / somma_periodo_precedente) * 100
    
    return inizio_periodo_precedente, inizio_ultimo_periodo, fine_ultimo_periodo, somma_periodo_precedente, somma_ultimo_periodo, percentuale_incremento

# Funzione per formattare la data
def formatta_data(data):
    mesi_italiani = {1: 'gennaio', 2: 'febbraio', 3: 'marzo', 4: 'aprile', 5: 'maggio', 6: 'giugno',
                     7: 'luglio', 8: 'agosto', 9: 'settembre', 10: 'ottobre', 11: 'novembre', 12: 'dicembre'}
    giorno = data.day
    mese = mesi_italiani[data.month]
    anno = data.year
    return f"{giorno} {mese} {anno}"

# Funzione per estrarre il nome del competitor dal file
def estrai_nome_competitor(file_path):
    file_name = os.path.basename(file_path)
    nome_competitor = file_name.split('-')[0]
    return nome_competitor

# Visualizzazione dell'immagine e del titolo
col1, col2 = st.columns([1, 7])
with col1:
    st.image("https://raw.githubusercontent.com/nurdigitalmarketing/previsione-del-traffico-futuro/9cdbf5d19d9132129474936c137bc8de1a67bd35/Nur-simbolo-1080x1080.png", width=80)
with col2:
    st.title('Previsione del Traffico Futuro')
    st.markdown('###### by [NUR® Digital Marketing](https://www.nur.it)')

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

# Caricamento del file CSV del sito cliente
st.header("Carica il file CSV del sito cliente")
uploaded_file_cliente = st.file_uploader("Carica il file CSV del traffico del sito cliente", type="csv")
if uploaded_file_cliente is not None:
    traffic_cliente = carica_dati(uploaded_file_cliente)
    if traffic_cliente is not None:
        modello_cliente = crea_modello(traffic_cliente)
        future_cliente = modello_cliente.make_future_dataframe(periods=365)
        forecast_cliente = modello_cliente.predict(future_cliente)
        
        # Calcolo confronto per il sito cliente
        inizio_periodo_precedente_cliente, inizio_ultimo_periodo_cliente, fine_ultimo_periodo_cliente, somma_periodo_precedente_cliente, somma_ultimo_periodo_cliente, percentuale_incremento_cliente = calcola_confronto(forecast_cliente)
        
        messaggio_cliente = f"""
            **Confronto del traffico tra i periodi (sito cliente):**
            - Dal {formatta_data(inizio_periodo_precedente_cliente + DateOffset(days=1))} al {formatta_data(inizio_ultimo_periodo_cliente)}: {formatta_numero(int(somma_periodo_precedente_cliente))} utenti
            - Dal {formatta_data(inizio_ultimo_periodo_cliente + DateOffset(days=1))} al {formatta_data(fine_ultimo_periodo_cliente)}: {formatta_numero(int(somma_ultimo_periodo_cliente))} utenti
            - **{'Incremento' if percentuale_incremento_cliente > 0 else 'Decremento'} percentuale:** {"-" if percentuale_incremento_cliente < 0 else ""}{abs(percentuale_incremento_cliente):.2f}%
        """
        
        st.info(messaggio_cliente)
        
        st.subheader("Previsioni del traffico futuro del sito cliente")
        fig1 = plot_plotly(modello_cliente, forecast_cliente)
        st.plotly_chart(fig1)

# Chiedi se ci sono competitor
competitors = st.checkbox("Vuoi aggiungere competitor?")

if competitors:
    uploaded_files_competitors = st.file_uploader("Carica i file CSV dei competitor", type="csv", accept_multiple_files=True)
    if uploaded_files_competitors:
        all_forecasts = {}
        for uploaded_file in uploaded_files_competitors:
            traffic_competitor = carica_dati(uploaded_file)
            if traffic_competitor is not None:
                modello_competitor = crea_modello(traffic_competitor)
                future_competitor = modello_competitor.make_future_dataframe(periods=365)
                forecast_competitor = modello_competitor.predict(future_competitor)
                
                nome_competitor = estrai_nome_competitor(uploaded_file.name)
                all_forecasts[nome_competitor] = forecast_competitor
        
        # Confronto tra il sito cliente e i competitor
        st.header("Confronto tra sito cliente e competitor")
        
        # Confronto per ciascun competitor
        for competitor_name, forecast_competitor in all_forecasts.items():
            inizio_periodo_precedente_competitor, inizio_ultimo_periodo_competitor, fine_ultimo_periodo_competitor, somma_periodo_precedente_competitor, somma_ultimo_periodo_competitor, percentuale_incremento_competitor = calcola_confronto(forecast_competitor)
            
            messaggio_competitor = f"""
                **Confronto del traffico tra i periodi (competitor {competitor_name}):**
                - Dal {formatta_data(inizio_periodo_precedente_competitor + DateOffset(days=1))} al {formatta_data(inizio_ultimo_periodo_competitor)}: {formatta_numero(int(somma_periodo_precedente_competitor))} utenti
                - Dal {formatta_data(inizio_ultimo_periodo_competitor + DateOffset(days=1))} al {formatta_data(fine_ultimo_periodo_competitor)}: {formatta_numero(int(somma_ultimo_periodo_competitor))} utenti
                - **{'Incremento' if percentuale_incremento_competitor > 0 else 'Decremento'} percentuale:** {"-" if percentuale_incremento_competitor < 0 else ""}{abs(percentuale_incremento_competitor):.2f}%
            """
            
            st.info(messaggio_competitor)
            
            # Confronto delle percentuali di crescita
            percentuale_confronto = percentuale_incremento_cliente - percentuale_incremento_competitor
            
            messaggio_confronto = f"""
                **Confronto della crescita percentuale tra sito cliente e competitor {competitor_name}:**
                - Sito cliente: {percentuale_incremento_cliente:.2f}%
                - Competitor {competitor_name}: {percentuale_incremento_competitor:.2f}%
                - **Differenza percentuale:** {percentuale_confronto:.2f}%
            """
            
            st.info(messaggio_confronto)
        
        # Grafico con le linee sovrapposte
        fig = go.Figure()
        
        # Aggiungi i dati del cliente
        fig.add_trace(go.Scatter(x=forecast_cliente['ds'], y=forecast_cliente['yhat'], mode='lines', name='Cliente'))
        
        # Aggiungi i dati dei competitor
        for competitor_name, forecast_competitor in all_forecasts.items():
            fig.add_trace(go.Scatter(x=forecast_competitor['ds'], y=forecast_competitor['yhat'], mode='lines', name=f'Competitor: {competitor_name}'))
        
        fig.update_layout(title='Previsioni del traffico futuro (Cliente vs Competitor)',
                          xaxis_title='Data',
                          yaxis_title='Traffico previsto',
                          template='plotly_white',
                          autosize=True,
                          width=1000,
                          height=600,
                          xaxis=dict(
                              tickformat='%Y-%m-%d',
                              dtick="M1",
                              tickangle=45
                          ))
        
        st.plotly_chart(fig)

# Sponsored content
st.markdown('**Sponsored**')
st.markdown('The Best AI [Summarizer](https://api.adzedek.com/click_linfo0314?chatbot_id=1715191360448x620213882279166000&operation_hash=b979afafb1a09c1b0090c00699565a13)')
