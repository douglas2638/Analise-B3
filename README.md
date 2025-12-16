# 📊 Análise B3 Profissional (Python)

Projeto de portfólio focado em empregabilidade: dados reais, séries temporais, indicadores técnicos, **backtesting** e **dashboard executivo**.

## ✅ O que este projeto entrega
- Coleta de dados históricos reais (Yahoo Finance via `yfinance`) para tickers da B3 (ex.: `PETR4.SA`)
- Pré-processamento de séries temporais (retornos, normalizações)
- Indicadores técnicos: SMA, EMA, RSI, volatilidade
- Estratégia exemplo baseada em RSI
- Backtest com métricas executivas:
  - Retorno total
  - Drawdown máximo
  - Taxa de acerto
  - Sharpe (diário)
- Dashboard (Streamlit + Plotly) com KPIs e gráficos

## 🚀 Como executar (Windows)
```bash
cd Analise-B3
pip install -r requirements.txt
streamlit run app/dashboard.py
```

## 📌 Observação
Projeto educacional/analítico. Não é recomendação de investimento.
