import pandas as pd

def retorno_total(df: pd.DataFrame) -> float:
    return float(df["equity_estrategia"].iloc[-1] - 1)

def drawdown_maximo(df: pd.DataFrame) -> float:
    equity = df["equity_estrategia"]
    pico = equity.cummax()
    drawdown = (equity - pico) / pico
    return float(drawdown.min())

def taxa_acerto(df: pd.DataFrame) -> float:
    trades = df[df["retorno_estrategia"].fillna(0) != 0]
    if trades.empty:
        return 0.0
    return float((trades["retorno_estrategia"] > 0).mean())

def sharpe_diario(df: pd.DataFrame, rf: float = 0.0) -> float:
    r = df["retorno_estrategia"].dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return float((r.mean() - rf) / r.std())
