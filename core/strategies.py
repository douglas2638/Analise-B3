import pandas as pd

def estrategia_rsi(df: pd.DataFrame, rsi_compra: float = 30, rsi_venda: float = 70) -> pd.DataFrame:
    df = df.copy()
    df["sinal"] = 0
    df.loc[df["rsi"] < rsi_compra, "sinal"] = 1
    df.loc[df["rsi"] > rsi_venda, "sinal"] = -1
    return df
