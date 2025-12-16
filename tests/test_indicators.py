import pandas as pd
from core.indicators import sma, ema, rsi

def test_sma_length():
    df = pd.DataFrame({"preco": [1,2,3,4,5]})
    assert len(sma(df, 2)) == 5

def test_ema_length():
    df = pd.DataFrame({"preco": [1,2,3,4,5]})
    assert len(ema(df, 2)) == 5

def test_rsi_length():
    df = pd.DataFrame({"preco": [1,2,3,2,1,2,3,4]})
    assert len(rsi(df, 2)) == 8
