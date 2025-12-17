def retorno_total(df):
    return df["equity_estrategia"].iloc[-1] - 1

def drawdown_maximo(df):
    equity = df["equity_estrategia"]
    pico = equity.cummax()
    return ((equity - pico) / pico).min()
