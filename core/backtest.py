def executar_backtest_com_stop(df, stop_loss, take_profit):
    # FIX DEFINITIVO: normaliza índice
    df = df.copy().reset_index(drop=True)

    df["posicao"] = 0
    df["retorno_estrategia"] = 0.0
    preco_entrada = None

    for i in range(1, len(df)):
        sinal = df.loc[i - 1, "sinal"]

        if sinal == 1 and df.loc[i - 1, "posicao"] == 0:
            df.loc[i, "posicao"] = 1
            preco_entrada = df.loc[i, "preco"]

        elif df.loc[i - 1, "posicao"] == 1:
            retorno = (df.loc[i, "preco"] - preco_entrada) / preco_entrada

            if retorno <= -stop_loss:
                df.loc[i, "retorno_estrategia"] = -stop_loss
                preco_entrada = None

            elif retorno >= take_profit:
                df.loc[i, "retorno_estrategia"] = take_profit
                preco_entrada = None

            else:
                df.loc[i, "posicao"] = 1
                df.loc[i, "retorno_estrategia"] = df.loc[i, "retorno"]

    df["equity_estrategia"] = (1 + df["retorno_estrategia"]).cumprod()
    df["equity_ativo"] = (1 + df["retorno"].fillna(0)).cumprod()
    return df
