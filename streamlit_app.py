import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import io

st.title('Previsione del Traffico Futuro')

# Caricamento del file CSV
uploaded_file = st.file_uploader("Carica il file CSV del traffico", type="csv")
if uploaded_file is not None:
    content = uploaded_file.getvalue().decode("utf-8")
    df = None
    
    # Rilevare il formato del file basandosi sull'intestazione
    if "Account:" in content:
        lines = content.split("\n")
        start_date_str = lines[5].split(": ")[1]
        start_date = datetime.strptime(start_date_str, "%Y%m%d")
        
        df = pd.read_csv(io.StringIO(content), skiprows=9)
        df.columns = ['Week Index', 'Organic Traffic']
        df['ds'] = df['Week Index'].apply(lambda x: start_date + timedelta(weeks=int(x)))
        df.drop(columns=['Week Index'], inplace=True)
    else:
        df = pd.read_csv(io.StringIO(content))
        if 'Date' in df.columns and 'Organic Traffic' in df.columns:
            df.rename(columns={'Date': 'ds', 'Organic Traffic': 'y'}, inplace=True)
    
    if df is not None and 'ds' in df.columns and 'y' in df.columns:
        st.write("Anteprima dei dati processati:")
        st.write(df.head())

        # Definizione degli update di Google come festività
        google_updates = pd.DataFrame({
            'holiday': 'google_update',
            'ds': pd.to_datetime([
                # Esempi di date di update di Google
                '2022-05-25', '2022-06-09', '2022-08-01', 
                '2022-12-12', '2023-03-15', '2023-07-20',
            ]),
            'lower_window': 0,
            'upper_window': 1,
        })

        m = Prophet(holidays=google_updates)
        m.fit(df)

        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        fig = plot_plotly(m, forecast)
        st.plotly_chart(fig)

        st.download_button(label="Scarica le previsioni in formato CSV",
                           data=forecast.to_csv().encode('utf-8'),
                           file_name='previsioni_traffico_futuro.csv',
                           mime='text/csv')
    else:
        st.error("Il file non ha il formato corretto o mancano le colonne necessarie.")
