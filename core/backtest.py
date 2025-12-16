import pandas as pd

def executar_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["retorno_estrategia"] = df["sinal"].shift(1) * df["retorno"]
    df["equity_ativo"] = (1 + df["retorno"].fillna(0)).cumprod()
    df["equity_estrategia"] = (1 + df["retorno_estrategia"].fillna(0)).cumprod()
    df["retorno_acumulado"] = df["equity_estrategia"] - 1
    return df
