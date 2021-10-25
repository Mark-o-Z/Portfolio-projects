import pandas as pd
import base64
import numpy as np
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
import plotly.express as px
import streamlit as st
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
st.title('Forecasting Dax index')

st.markdown("""
This app retrieves the list of the **Dax** indices  (from Wikipedia) and theirs corresponding **stock prices** (year-to-date)!
* **Python libraries:** pandas, numpy, streamlit, plotly, yfinance, fbprophet, datetime, base 64
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/DAX).
""")

def get_tickers():
    url = 'https://en.wikipedia.org/wiki/DAX'
    html = pd.read_html(url, header = 0)
    df = html[3]
    return df

df = get_tickers()

stock_names = df['Ticker symbol']
tickers = tuple(stock_names)

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

selected_stock = st.selectbox('Select stock for prediction', tickers)

n_months = st.slider('Months of prediction:', 1, 2, 3)
period = n_months * 30

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#data = load_data(tickers)

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

#data = data[['Date', 'Close']]

st.subheader('Data from selected DAX index')
st.write(data)

def filedownload(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="DAX_index.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(data), unsafe_allow_html=True)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text=(f'Plotting {selected_stock} index'), xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

df_train['ds'] = pd.to_datetime(df_train['ds'], format='%Y-%m-%d')
df_train['y'] = pd.to_numeric(df_train['y'],errors='ignore')

m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.15, daily_seasonality=False)
m.fit(df_train)

future = m.make_future_dataframe(periods=period, freq='D')
forecast = m.predict(future)

st.subheader('Forecast data')
#st.write(forecast.tail())
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

def filedownload(data):
    csv = forecast.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="DAX_forecast.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(forecast), unsafe_allow_html=True)


st.subheader(f'Forecast plot for {n_months} months')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("Forecast components")
fig2 =plot_components_plotly(m, forecast)
st.plotly_chart(fig2)

#df_cv = cross_validation(m, initial = '730 days', period='50 days', horizon = '100 days')
#st.subheader('Cross validation')
#st.write(df_cv.head())

#df_p = performance_metrics(df_cv)
#st.subheader('Performance metrics')
#st.write(df_p.head())
