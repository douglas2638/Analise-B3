import os
import sys
from datetime import date

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import streamlit as st
import plotly.graph_objects as go

from infra.config import DEFAULT_START_DATE, DEFAULT_TICKERS
from infra.logger import get_logger
from core.data_source import baixar_historico
from core.preprocessing import preparar_serie
from core.indicators import sma, ema, rsi, volatilidade_retorno
from core.strategies import estrategia_rsi
from core.backtest import executar_backtest
from core.metrics import retorno_total, drawdown_maximo, taxa_acerto, sharpe_diario

logger = get_logger()

st.set_page_config(page_title="Dashboard B3 Profissional", layout="wide")
st.title("📊 Dashboard B3")


with st.sidebar:
    st.header("⚙️ Parâmetros")
    ticker = st.selectbox("Ticker (.SA):", DEFAULT_TICKERS, index=0)
    inicio = st.text_input("Data início (YYYY-MM-DD):", DEFAULT_START_DATE)
    sma_p = st.slider("SMA (período)", 5, 200, 20)
    ema_p = st.slider("EMA (período)", 5, 200, 20)
    rsi_p = st.slider("RSI (período)", 5, 50, 14)
    rsi_buy = st.slider("RSI compra (<)", 5, 45, 30)
    rsi_sell = st.slider("RSI venda (>)", 55, 95, 70)

@st.cache_data(ttl=60*60, show_spinner=True)
def load_data(ticker: str, inicio: str):
    df = baixar_historico(ticker, inicio=inicio)
    return preparar_serie(df)

try:
    df = load_data(ticker, inicio)
    logger.info(f"Dados carregados: {ticker} | início={inicio} | linhas={len(df)}")
except Exception as e:
    st.error(f"Falha ao carregar dados para {ticker}. Detalhes: {e}")
    st.stop()

df["sma"] = sma(df, sma_p)
df["ema"] = ema(df, ema_p)
df["rsi"] = rsi(df, rsi_p)
df["vol"] = volatilidade_retorno(df, 20)

df = estrategia_rsi(df, rsi_compra=rsi_buy, rsi_venda=rsi_sell)
df = executar_backtest(df)

ret = retorno_total(df)
dd = drawdown_maximo(df)
hit = taxa_acerto(df)
sh = sharpe_diario(df)

st.subheader("📌 Visão Executiva")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📈 Retorno Total", f"{ret*100:.2f}%")
c2.metric("📉 Drawdown Máximo", f"{dd*100:.2f}%")
c3.metric("🎯 Taxa de Acerto", f"{hit*100:.1f}%")
c4.metric("⚖️ Sharpe (diário)", f"{sh:.2f}")

st.subheader("📈 Preço e Médias")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["data"], y=df["preco"], name="Preço"))
fig.add_trace(go.Scatter(x=df["data"], y=df["sma"], name=f"SMA {sma_p}"))
fig.add_trace(go.Scatter(x=df["data"], y=df["ema"], name=f"EMA {ema_p}"))
fig.update_layout(yaxis_title="Preço", xaxis_title="Data", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📉 RSI e níveis")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df["data"], y=df["rsi"], name="RSI"))
fig_rsi.add_hline(y=rsi_sell, line_dash="dash")
fig_rsi.add_hline(y=rsi_buy, line_dash="dash")
fig_rsi.update_layout(yaxis_title="RSI", xaxis_title="Data", hovermode="x unified")
st.plotly_chart(fig_rsi, use_container_width=True)

st.subheader("🏁 Desempenho: Estratégia vs Ativo")
fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=df["data"], y=df["equity_ativo"], name="Ativo (buy&hold)"))
fig_bt.add_trace(go.Scatter(x=df["data"], y=df["equity_estrategia"], name="Estratégia"))
fig_bt.update_layout(yaxis_title="Equity (base=1)", xaxis_title="Data", hovermode="x unified")
st.plotly_chart(fig_bt, use_container_width=True)

with st.expander("🔎 Ver dados (últimas 200 linhas)"):
    st.dataframe(df[["data","preco","retorno","sma","ema","rsi","sinal","retorno_estrategia","equity_ativo","equity_estrategia"]].tail(200))

st.caption(f"Atualizado em {date.today().isoformat()}")
