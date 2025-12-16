import pandas as pd
from datetime import date

def atualizar_dados():
    
    url = "https://exemplo-b3-dados.csv"
    df = pd.read_csv(url, sep=';', encoding='latin1')
    df['data_coleta'] = date.today().isoformat()
    df.to_csv('data/raw/b3.csv', index=False)

if __name__ == '__main__':
    atualizar_dados()