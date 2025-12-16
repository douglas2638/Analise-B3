def limpar_dados(df):
    df.columns = df.columns.str.lower().str.strip()

    df['preco'] = df['preco'].str.replace(',', '.').astype(float)
    df['variacao'] = df['variacao'].str.replace(',', '.').astype(float)
    df['volume'] = df['volume'].astype(int)
    df['data'] = df['data'].astype(str)

    return df
