from __future__ import annotations
import pandas as pd
import yfinance as yf

def baixar_historico(ticker: str, inicio: str = "2023-01-01") -> pd.DataFrame:
    """Baixa OHLCV histórico. Para ações brasileiras use sufixo .SA (ex.: PETR4.SA)."""
    hist = yf.Ticker(ticker).history(start=inicio)
    if hist is None or hist.empty:
        raise ValueError(f"Sem dados retornados para ticker: {ticker}")

    df = hist.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "data"})
    elif "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "data"})

    df = df.rename(columns={
        "Open": "abertura",
        "High": "maxima",
        "Low": "minima",
        "Close": "preco",
        "Volume": "volume",
    })
    df["codigo"] = ticker.replace(".SA", "")
    return df[["data", "codigo", "abertura", "maxima", "minima", "preco", "volume"]]
