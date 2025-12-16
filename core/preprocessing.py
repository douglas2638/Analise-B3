import pandas as pd

def preparar_serie(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data")
    df["retorno"] = df["preco"].pct_change()
    return df
