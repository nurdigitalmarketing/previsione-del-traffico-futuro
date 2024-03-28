# Ottimizzazione del codice Streamlit per la previsione del traffico futuro

# Import delle librerie necessarie
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import timedelta
from pandas.tseries.offsets import DateOffset

# Definizione della funzione per formattare i numeri
def formatta_numero(numero):
    """Formatta il numero con il punto come separatore delle migliaia."""
    return f"{numero:,}".replace(',', '.')  # Formatta e sostituisce la virgola con il punto

# Funzione per formattare le date in italiano
def formatta_data(data):
    """Formatta la data nel formato giorno mese anno in italiano."""
    mesi_italiani = {
        1: 'gennaio', 2: 'febbraio', 3: 'marzo', 4: 'aprile', 5: 'maggio',
        6: 'giugno', 7: 'luglio', 8: 'agosto', 9: 'settembre', 10: 'ottobre',
        11: 'novembre', 12: 'dicembre'
    }
    giorno, mese, anno = data.day, mesi_italiani[data.month], data.year
    return f"{giorno} {mese} {anno}"

# Costruzione dell'interfaccia utente con Streamlit
def build_ui():
    st.title('Previsione del Traffico Futuro')
    # Descrizione e istruzioni
    st.markdown("""
    ## Introduzione
    Questo strumento è sviluppato per fornire previsioni sul traffico futuro basandosi sull'export dei dati da _Google Analytics_.
    ## Funzionamento
    Per previsioni accurate, segui i passaggi per la preparazione dei dati.
    """)
    with st.expander("Istruzioni"):
        st.markdown("""
        1. **Esportazione dei dati da Looker Studio**.
        2. **Pulizia dei dati**: Rinomina le colonne in `Date` e `Organic Traffic`.
        3. **Caricamento del file**: Carica il tuo file CSV.
        """)

# Funzione per la previsione del traffico futuro
def forecast_traffic(uploaded_file):
    traffic = pd.read_csv(uploaded_file)
    if 'Date' not in traffic.columns:
        st.error("La colonna 'Date' è mancante.")
        return

    traffic.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
    traffic['ds'] = pd.to_datetime(traffic['ds'])

    # Verifica delle colonne necessarie
    if not {"ds", "y"}.issubset(traffic.columns):
        st.error("Mancano colonne necessarie.")
        return

    m = Prophet(holidays=get_holidays())
    m.fit(traffic)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    compare_traffic(forecast)  # Confronto del traffico e visualizzazione dei risultati

    # Visualizzazione delle previsioni
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Download delle previsioni
    st.download_button(
        label="Scarica le previsioni in formato CSV",
        data=forecast.to_csv().encode('utf-8'),
        file_name='previsioni_traffico_futuro.csv',
        mime='text/csv'
    )

# Funzione per ottenere le festività (aggiornamenti di Google)
def get_holidays():
    dates = ['2015-07-17', '2016-01-08', '2016-09-27', '2017-03-08', '2017-07-09',
             '2018-03-08', '2018-04-17', '2018-08-01', '2019-03-12', '2019-06-03',
             '2019-09-24', '2019-10-25', '2019-12-09', '2020-01-13', '2020-05-04',
             '2020-12-03', '2021-06-02', '2021-07-01', '2021-11-17', '2022-05-25',
             '2023-09-15', '2023-10-05', '2023-11-02', '2023-11-08', '2024-03-05']
    return pd.DataFrame({
        'holiday': 'Google Update',
        'ds': pd.to_datetime(dates),
        'lower_window': 0,
        'upper_window': 1,
    })

# Funzione per confrontare il traffico tra due periodi e visualizzare i risultati
def compare_traffic(forecast):
    fine_ultimo_periodo = forecast['ds'].max()
    inizio_ultimo_periodo = fine_ultimo_periodo - DateOffset(days=365)
    inizio_periodo_precedente = inizio_ultimo_periodo - DateOffset(days=365)
    
    # Calcolo delle somme per i due periodi
    somma_ultimo_periodo = forecast[(forecast['ds'] > inizio_ultimo_periodo) & (forecast['ds'] <= fine_ultimo_periodo)]['yhat'].sum()
    somma_periodo_precedente = forecast[(forecast['ds'] > inizio_periodo_precedente) & (forecast['ds'] <= inizio_ultimo_periodo)]['yhat'].sum()

    incremento = somma_ultimo_periodo - somma_periodo_precedente
    percentuale_incremento = (incremento / somma_periodo_precedente) * 100

    messaggio = f"""
    **Confronto del traffico tra i periodi:**
    - Dal {formatta_data(inizio_periodo_precedente + DateOffset(days=1))} al {formatta_data(inizio_ultimo_periodo)}: {formatta_numero(int(somma_periodo_precedente))} utenti
    - Dal {formatta_data(inizio_ultimo_periodo + DateOffset(days=1))} al {formatta_data(fine_ultimo_periodo)}: {formatta_numero(int(somma_ultimo_periodo))} utenti
    - **{'Incremento' if percentuale_incremento > 0 else 'Decremento'} percentuale:** {percentuale_incremento:.2f}%
    """
    
    if percentuale_incremento > 0:
        st.success(messaggio)
    else:
        st.error(messaggio)

# Esempio di come utilizzare le funzioni definite sopra (commentato per evitare l'esecuzione qui)
# Esempio di come utilizzare le funzioni definite sopra
build_ui()
# Aggiunta di un pulsante per caricare e analizzare il file CSV
if st.button("Carica e Analizza"):
    uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
    if uploaded_file is not None:
        forecast_traffic(uploaded_file)
