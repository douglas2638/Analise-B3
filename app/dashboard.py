import sys, os, datetime as dt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

from core.indicators import sma, rsi, estocastico
from core.strategies import estrategia_mm_rsi_estocastico
from core.backtest import executar_backtest_com_stop
from core.metrics import retorno_total, drawdown_maximo

st.set_page_config(layout="wide")
st.title("📊 Estratégia Profissional B3 – Painel Administrativo")

with st.sidebar:
    st.header("📅 Período")
    data_inicio = st.date_input("Data inicial", dt.date.today() - dt.timedelta(days=365))
    data_fim = st.date_input("Data final", dt.date.today())

    st.header("⚙️ Estratégia")
    ticker = st.selectbox("Ativo", ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA"])
    mm_periodo = st.slider("Média Móvel", 5, 200, 20)

    rsi_periodo = st.slider("RSI Período", 5, 50, 14)
    rsi_buy = st.slider("RSI Compra", 10, 50, 40)
    rsi_sell = st.slider("RSI Venda", 50, 90, 60)

    stoch_k_periodo = st.slider("Estocástico %K", 5, 30, 14)
    stoch_d_periodo = st.slider("Estocástico %D", 2, 10, 3)
    stoch_buy = st.slider("Estocástico Compra", 5, 40, 20)
    stoch_sell = st.slider("Estocástico Venda", 60, 95, 80)

    st.header("🛡️ Risco")
    stop_loss = st.slider("Stop Loss (%)", 0.5, 15.0, 2.0) / 100
    take_profit = st.slider("Take Profit (%)", 1.0, 30.0, 4.0) / 100

df = yf.Ticker(ticker).history(period="2y").reset_index()
df = df.rename(columns={"Date":"data","High":"maxima","Low":"minima","Close":"preco"})
df["data"] = df["data"].dt.date

df = df[(df["data"] >= data_inicio) & (df["data"] <= data_fim)]
df["retorno"] = df["preco"].pct_change()

df["sma"] = sma(df, mm_periodo)
df["rsi"] = rsi(df, rsi_periodo)
df["stoch_k"], df["stoch_d"] = estocastico(df, stoch_k_periodo, stoch_d_periodo)

df = estrategia_mm_rsi_estocastico(df, rsi_buy, rsi_sell, stoch_buy, stoch_sell)
df = executar_backtest_com_stop(df, stop_loss, take_profit)

ret = retorno_total(df)
dd = drawdown_maximo(df)

st.subheader("🧑‍💼 KPIs Executivos")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Período", f"{data_inicio} → {data_fim}")
c2.metric("Retorno Total", f"{ret*100:.2f}%")
c3.metric("Drawdown Máx", f"{dd*100:.2f}%")
c4.metric("Risco", f"SL {stop_loss*100:.1f}% | TP {take_profit*100:.1f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["data"], y=df["equity_ativo"], name="Ativo"))
fig.add_trace(go.Scatter(x=df["data"], y=df["equity_estrategia"], name="Estratégia"))
fig.update_layout(title="Desempenho da Estratégia", yaxis_title="Equity")
st.plotly_chart(fig, use_container_width=True)
