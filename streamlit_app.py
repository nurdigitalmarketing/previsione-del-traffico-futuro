import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import io

st.title('Previsione del Traffico Futuro')

uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    df = None
    
    # Verifica se il file sembra essere in formato Analytics
    if "Account:" in content and "Data di inizio:" in content:
        lines = content.split("\n")
        start_date_str = None
        
        # Cerca la data di inizio nel file
        for line in lines:
            if "Data di inizio:" in line:
                start_date_str = line.split(": ")[1].strip()
                break
        
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y%m%d")
                df = pd.read_csv(io.StringIO(content), skiprows=9)
                # Assumi che la prima colonna sia l'indice settimanale e la seconda il traffico organico
                df.columns = ['Week Index', 'y']
                df['ds'] = df.apply(lambda row: start_date + timedelta(weeks=int(row['Week Index'])), axis=1)
                df.drop(columns=['Week Index'], inplace=True)
            except ValueError as e:
                st.error(f"Errore nella conversione della data: {e}")
                st.stop()
        else:
            st.error("Data di inizio non trovata.")
            st.stop()
    else:
        # Assumi formato Ahrefs o simile
        df = pd.read_csv(io.StringIO(content))
        if 'Date' in df.columns and 'Organic Traffic' in df.columns:
            df.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
        else:
            st.error("Le colonne necessarie non sono state trovate.")
            st.stop()

    if 'ds' in df.columns and 'y' in df.columns:
        # Mostra l'anteprima dei dati
        st.write(df.head())

        # Definizione delle festività (aggiornamenti di Google come esempio)
        google_updates = pd.DataFrame({
            'holiday': 'google_update',
            'ds': pd.to_datetime(['2023-01-01', '2023-07-04', '2023-12-25']),
            'lower_window': 0,
            'upper_window': 1,
        })

        m = Prophet(holidays=google_updates)
        m.fit(df)
        
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        
        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig)

        # Opzione per scaricare le previsioni
        st.download_button(label="Scarica le previsioni in formato CSV",
                           data=forecast.to_csv().encode('utf-8'),
                           file_name='previsioni_traffico_futuro.csv',
                           mime='text/csv')
    else:
        st.error("Errore nel processamento dei dati.")
