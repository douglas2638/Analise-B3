import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import streamlit as st
import plotly.graph_objects as go
from src.loader import carregar_dados
from src.cleaner import limpar_dados
from src.indicators import media_movel, media_movel_exponencial, rsi

st.set_page_config(page_title="Dashboard B3", layout="wide")
st.title("📊 Dashboard de Análise da B3")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "b3.csv")

df = carregar_dados(CSV_PATH)
df = limpar_dados(df)

ativo = st.selectbox("Selecione o ativo:", df['codigo'].unique())
df_ativo = df[df['codigo'] == ativo].sort_values('data')

df_ativo['sma20'] = media_movel(df_ativo)
df_ativo['ema20'] = media_movel_exponencial(df_ativo)
df_ativo['rsi'] = rsi(df_ativo)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_ativo['data'], y=df_ativo['preco'], name='Preço'))
fig.add_trace(go.Scatter(x=df_ativo['data'], y=df_ativo['sma20'], name='SMA 20'))
fig.add_trace(go.Scatter(x=df_ativo['data'], y=df_ativo['ema20'], name='EMA 20'))
st.plotly_chart(fig, use_container_width=True)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df_ativo['data'], y=df_ativo['rsi'], name='RSI'))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
st.plotly_chart(fig_rsi, use_container_width=True)