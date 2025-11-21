"""
Exemplos de uso da API B3 Data Analyzer
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def exemplo_busca_dados_acao():
    """Exemplo: Buscar dados de uma ação"""
    response = requests.get(f"{BASE_URL}/stock/PETR4", params={"period": "6mo", "include_indicators": True})
    if response.status_code == 200:
        data = response.json()
        print(f"Dados de {data['symbol']}:")
        print(f"Período: {data['period']}")
        print(f"Pontos de dados: {data['data_points']}")
        print(f"Último preço: R$ {data['data'][-1]['close']:.2f}")
    else:
        print(f"Erro: {response.status_code} - {response.text}")

def exemplo_analise_tecnica():
    """Exemplo: Análise técnica completa"""
    payload = {
        "symbol": "VALE3",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "indicators": ["RSI", "MACD", "Moving_Averages"]
    }
    
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    if response.status_code == 200:
        analysis = response.json()
        print(f"Análise de {analysis['symbol']}:")
        print(f"Score: {analysis['score']}/100")
        print(f"Retorno anual: {analysis['statistics']['annual_return']:.2f}%")
        print(f"Recomendações: {analysis['recommendations']}")
    else:
        print(f"Erro: {response.status_code} - {response.text}")

def exemplo_analise_carteira():
    """Exemplo: Análise de carteira"""
    params = {
        "stocks": ["PETR4", "VALE3", "ITUB4", "BBDC4"],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "initial_investment": 10000
    }
    
    response = requests.get(f"{BASE_URL}/portfolio/analyze", params=params)
    if response.status_code == 200:
        portfolio = response.json()
        print("Análise da Carteira:")
        print(f"Retorno esperado: {portfolio['portfolio_statistics']['expected_annual_return']:.2f}%")
        print(f"Volatilidade: {portfolio['portfolio_statistics']['annual_volatility']:.2f}%")
        print(f"Score diversificação: {portfolio['diversification_score']:.1f}")
    else:
        print(f"Erro: {response.status_code} - {response.text}")

def exemplo_visao_mercado():
    """Exemplo: Visão geral do mercado"""
    response = requests.get(f"{BASE_URL}/market/overview")
    if response.status_code == 200:
        overview = response.json()
        print("Visão do Mercado:")
        print(f"Sentimento: {overview['market_sentiment']}")
        print(f"Variação média: {overview['average_change']:.2f}%")
        for stock, data in overview['stocks'].items():
            print(f"{stock}: R$ {data['current_price']:.2f} ({data['change']:+.2f}%)")
    else:
        print(f"Erro: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("=== Exemplos de Uso da API B3 Data Analyzer ===\n")
    
    print("1. Busca de dados de ação:")
    exemplo_busca_dados_acao()
    
    print("\n2. Análise técnica:")
    exemplo_analise_tecnica()
    
    print("\n3. Análise de carteira:")
    exemplo_analise_carteira()
    
    print("\n4. Visão do mercado:")
    exemplo_visao_mercado()