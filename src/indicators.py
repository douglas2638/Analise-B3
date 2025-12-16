def media_movel(df, periodo=20):
    return df['preco'].rolling(periodo).mean()

def media_movel_exponencial(df, periodo=20):
    return df['preco'].ewm(span=periodo, adjust=False).mean()

def volatilidade(df, periodo=20):
    return df['variacao'].rolling(periodo).std()

def rsi(df, periodo=14):
    delta = df['preco'].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))