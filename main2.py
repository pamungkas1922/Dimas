import streamlit as st
from datetime import date
import yfinance
from neuralprophet import NeuralProphet
from plotly import graph_objs as go
import pandas as pd
import datetime
import numpy as np

st.title("prediksi harga saham teratas indonesia")
st.sidebar.subheader("PREDIKSI HARGA SAHAM")
st.sidebar.subheader("RANGE DATA SET")

stocks = ["BBCA.JK", "BBRI.JK", "BYAN.JK", "BMRI.JK", "TLKM.JK"]
selected_stock = st.sidebar.selectbox("Pilih stok", stocks)

START = st.sidebar.date_input("Mulai Dari", datetime.date(2016,12,1))
END = st.sidebar.date_input("Sampai Dengan", datetime.date(2023,7,4))

period = 7

@st.cache
def load_data(stock):
    data = yfinance.download(stock, START, END)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

st.subheader("stock data")
st.write(data.tail())

# Plot stock data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data["Date"], y=data['Close'], name="stock_open"))
fig1.layout.update(title_text="Time series data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)      

if st.sidebar.button('Prediksi'):
    data = data[["Date", "Close"]]
    data= data.rename(columns={"Date": "ds", "Close": "y"})
    
    test_days = 50
    train_data = data[:-test_days]
    test_data = data[-test_days:]
    
    model = NeuralProphet(n_forecasts=period)
    model.fit(data)

    st.subheader('Hasil Prediksi')
    
    future = model.make_future_dataframe(data, periods=period, n_historic_predictions=len(data))
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'], format='%Y-%m-%d')
    
    data_stacked = forecast.set_index('ds').stack().reset_index()
    data_stacked.columns = ['ds', 'col',  'value']
    hasil = data_stacked.loc[(data_stacked['col'].str.contains('yhat')) & (~data_stacked['value'].isnull())]
    hasil = hasil.drop('col', axis=1)
    hasil = hasil.rename(columns={'ds': 'tanggal_mendatang', 'value': 'prediksi'})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name="Actual Data", line_color='blue'))
    fig.add_trace(go.Scatter(x=hasil['tanggal_mendatang'], y=hasil['prediksi'], name="Hasil Prediksi", line_color='lightblue'))
    fig.update_layout(title_text="Time series data with Prediction", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    st.dataframe(hasil, height=247, width=800)
    
    y_test = test_data['y']
    y_pred = forecast['yhat1'].iloc[-test_days:]
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    mape3 = mean_absolute_percentage_error(y_test, y_pred)
    st.write("Hasil MAPE: ", mape3)
