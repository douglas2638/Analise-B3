📊 B3 Analyse API
Uma API robusta em Python para análise de dados da Bolsa de Valores Brasileira (B3). Desenvolvida com FastAPI, oferece análise técnica, indicadores financeiros e ferramentas para tomada de decisão de investimentos.

🚀 Funcionalidades
📈 Análise Técnica
Indicadores Técnicos: RSI, MACD, Médias Móveis, Bollinger Bands

Análise de Tendências: Identificação de tendências de alta/baixa

Cálculo de Volatilidade: Risk metrics e drawdown

Suporte e Resistência: Identificação automática de níveis-chave

🏦 Dados do Mercado
Dados Históricos: Cotações diárias de ações da B3

Informações da Empresa: Dados fundamentais e setoriais

Visão do Mercado: Overview em tempo real

Análise de Carteira: Diversificação e correlação

⚡ API Features
Rate Limiting Inteligente: Proteção contra bloqueios do Yahoo Finance

Cache Multi-camadas: Performance otimizada

Fallback Automático: Dados mock para desenvolvimento

Documentação Interativa: Swagger UI e ReDoc

Health Checks: Monitoramento de serviços

🛠️ Tecnologias
Backend: FastAPI, Python 3.11+

Data Processing: Pandas, NumPy, SciPy

Data Collection: yfinance, Requests

Cache: In-memory caching

Container: Docker, Docker Compose

Documentation: Swagger UI, ReDoc

📦 Instalação
Pré-requisitos
Python 3.11 ou superior

pip (gerenciador de pacotes Python)

Instalação Local
Clone o repositório

bash
git clone https://github.com/douglas2638/analise-b3.git
cd analise-b3
Crie um ambiente virtual

bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# ou
venv\Scripts\activate  # Windows
Instale as dependências

bash
pip install -r requirements.txt
Execute a aplicação

bash
python run.py
# ou
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Instalação com Docker
Build e execute com Docker

bash
docker build -t b3-analyser .
docker run -p 8000:8000 b3-analyser
Ou com Docker Compose

bash
docker-compose up -d
🚀 Uso Rápido
Acesse a Documentação
Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc
