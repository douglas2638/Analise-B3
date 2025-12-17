def sma(df, periodo=20):
    return df["preco"].rolling(periodo).mean()

def rsi(df, periodo=14):
    delta = df["preco"].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))

def estocastico(df, k_period=14, d_period=3):
    low_min = df["minima"].rolling(k_period).min()
    high_max = df["maxima"].rolling(k_period).max()
    k = 100 * (df["preco"] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return k, d
