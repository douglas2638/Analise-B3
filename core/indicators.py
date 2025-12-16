import pandas as pd

def sma(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    return df["preco"].rolling(periodo).mean()

def ema(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    return df["preco"].ewm(span=periodo, adjust=False).mean()

def volatilidade_retorno(df: pd.DataFrame, periodo: int = 20) -> pd.Series:
    return df["retorno"].rolling(periodo).std()

def rsi(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    delta = df["preco"].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))
