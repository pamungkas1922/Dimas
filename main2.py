import streamlit as st
from datetime import date
import yfinance
from neuralprophet import NeuralProphet
from plotly import graph_objs as go
import pandas as pd
import datetime
import numpy as np

# Define the correct username and password
CORRECT_USERNAME = "your_username"
CORRECT_PASSWORD = "your_password"

# Create a login function
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    submitted = st.button("Submit")

    if submitted:
        if username == CORRECT_USERNAME and password == CORRECT_PASSWORD:
            st.success("Logged in as {}".format(username))
            return True
        else:
            st.error("Incorrect username or password")
    return False

# Main function to display the app content
def main():
    st.title("prediksi harga saham teratas indonesia")
    st.sidebar.subheader("PREDIKSI HARGA SAHAM")
    st.sidebar.subheader("RANGE DATA SET")

    stocks = ["BBCA.JK", "BBRI.JK", "BYAN.JK", "BMRI.JK", "TLKM.JK"]
    selected_stock = st.sidebar.selectbox("Pilih stok", stocks)

    START = st.sidebar.date_input("Mulai Dari", datetime.date(2016, 12, 1))
    END = st.sidebar.date_input("Sampai Dengan", datetime.date(2023, 7, 4))

    # n_weeks = st.sidebar.slider("Prediksi berapa minggu", 1, 4)
    period = 7

    @st.cache
    def load_data(stock):
        data = yfinance.download(stock, START, END)
        data.reset_index(inplace=True)
        return data

    data = load_data(selected_stock)

    st.subheader("stock data")
    st.write(data.tail())

    # Plot stock data and predictions
    fig1 = go.Figure()

    # Plot actual stock data
    fig1.add_trace(go.Scatter(x=data["Date"], y=data['Close'], name="Actual Stock Price", line_color='blue'))

    if st.sidebar.button('Prediksi'):
        # predict future prices
        data_for_pred = data[["Date", "Close"]]
        data_for_pred = data_for_pred.rename(columns={"Date": "ds", "Close": "y"})

        test_days = 50
        train_data = data_for_pred[:-test_days]
        test_data = data_for_pred[-test_days:]

        model = NeuralProphet(n_forecasts=period)
        model.fit(train_data)

        # Predict future prices
        future = model.make_future_dataframe(train_data, periods=test_days + period, n_historic_predictions=len(train_data))
        forecast = model.predict(future)
        forecast['ds'] = pd.to_datetime(forecast['ds'], format='%Y-%m-%d')

        # Plot predicted stock prices
        fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], name="Predicted Stock Price", line_color='red'))

    fig1.layout.update(title_text="Time series data and Predictions", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig1)

    # Display the predicted values in a table
    if 'forecast' in locals():
        st.subheader('Hasil Prediksi')
        hasil = forecast[['ds', 'yhat1']].iloc[-period:]
        hasil.columns = ['tanggal_mendatang', 'prediksi']
        st.dataframe(hasil)

        y_test = test_data['y']
        y_pred = forecast['yhat1'].iloc[-test_days:]
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape3 = mean_absolute_percentage_error(y_test, y_pred)
        st.write("Hasil MAPE: ", mape3)

# Check if the user is logged in
logged_in = login()

if not logged_in:
    st.stop()  # Stop the app if not logged in

# Run the app for the logged-in user
main()
