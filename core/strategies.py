def estrategia_mm_rsi_estocastico(df, rsi_buy, rsi_sell, stoch_buy, stoch_sell):
    df = df.copy()
    df["sinal"] = 0

    cruzou_cima = (
        (df["stoch_k"] > df["stoch_d"]) &
        (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1))
    )

    cruzou_baixo = (
        (df["stoch_k"] < df["stoch_d"]) &
        (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1))
    )

    df.loc[(df["rsi"] < rsi_buy) & (df["stoch_k"] < stoch_buy) & cruzou_cima, "sinal"] = 1
    df.loc[(df["rsi"] > rsi_sell) & (df["stoch_k"] > stoch_sell) & cruzou_baixo, "sinal"] = -1
    return df
