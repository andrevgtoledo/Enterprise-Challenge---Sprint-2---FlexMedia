Totem IA — Sprint 2

Continuação prática da arquitetura definida na Sprint 1 (Flexmedia Challenge)

Este repositório contém a implementação prática dos módulos definidos na Sprint 1, demonstrando coleta de dados via sensores (simulados), armazenamento em banco SQL, análise estatística, visualização em dashboard e aplicação de Machine Learning supervisionado.

Este documento serve como guia completo para execução, entendimento e operação do protótipo entregue nesta Sprint.

📌 1. Objetivo da Sprint 2

A Sprint 2 tem como foco transformar a arquitetura projetada na Sprint 1 em um protótipo funcional, com integração entre sensores simulados, backend simples, banco de dados SQL, análises e dashboard.

Os objetivos definidos pela Flexmedia e exigidos pela Sprint foram cumpridos de forma direta:

✔ Integração Sensor → Banco SQL → Análise
✔ Registro e estruturação de interações
✔ Dashboard simples em Python (Streamlit)
✔ ML supervisionado (classificação toque curto vs longo)
✔ Limpeza, padronização e validação de dados coletados
✔ Geração de gráficos, relatórios e prints para documentação
📡 2. Arquitetura Implementada (Sprint 2)

Esta Sprint implementa o fluxo completo:

Sensor Simulado (Python)
      ↓
Arquivo CSV / API (opcional)
      ↓
Armazenamento SQL (SQLite, simples)
      ↓
Análise estatística (Python + Pandas)
      ↓
Dashboard (Streamlit)
      ↓
Machine Learning supervisionado (RandomForest)

Componentes entregues:
Módulo	Arquivo	Descrição
Simulação de sensores	sensor_sim.py	Gera eventos coerentes com uso real do totem
Ingestão SQL	ingest_to_sql.py	Lê CSV e popula banco SQLite estruturado
Esquema de banco	flexmedia.sqlite	Armazena sessões e interações
Análise de dados	analysis.py	Limpeza, métricas, gráficos e relatório
Dashboard visual	dashboard_streamlit.py	Visualizações de métricas do totem
ML Supervisionado	ml_train.py	Classificação toque curto vs longo
Dataset simulado	sample_interactions.csv	>2.000 eventos reais simulados
🧪 3. Scripts Disponíveis
📍 3.1. Simulador de Sensores — sensor_sim.py

Gera interações simuladas contendo:

timestamp

sensor_id (touch/presence)

tipo de interação

duração do toque

idioma

conteúdo acessado

pergunta/resposta simulada

session_anon_id

Esses dados são gravados em CSV ou enviados ao backend.

📍 3.2. Ingestão para SQL — ingest_to_sql.py

Lê o CSV gerado pelo simulador e popula o banco SQLite:

Tabela sessao

Tabela interacao

Remove duplicações, converge sessões e limpa dados incoerentes.

📍 3.3. Análises — analysis.py

Gera:

Total de interações

Interações por tipo

Duração média

Distribuição de toques (short/long)

Top 10 perguntas

Gráficos (PNG)

Relatório JSON (report_summary.json)

📍 3.4. Dashboard — dashboard_streamlit.py

Interface simples que mostra:

Total de interações

Gráficos automáticos

Últimas interações em tabela

Métricas gerais

Execução com:

streamlit run dashboard_streamlit.py

📍 3.5. Machine Learning — ml_train.py

Treina um pequeno classificador RandomForest:

Entrada: duração do toque

Saída: short (≤0.5s) ou long (>0.5s)

Mostra métricas com classification_report.

🗃 4. Estrutura do Repositório
totem-ia-sprint2/
│
├── data/
│   ├── sample_interactions.csv     # Dados simulados
│   ├── report_summary.json         # Relatório gerado pela análise
│   └── flexmedia.sqlite            # Banco SQL populado
│
├── media/
│   ├── interacoes_por_tipo.png
│   ├── touch_dist.png
│   └── video_demo_link.txt
│
├── sensor_sim.py
├── ingest_to_sql.py
├── analysis.py
├── dashboard_streamlit.py
├── ml_train.py
└── README.md

▶️ 5. Como Executar o Projeto (Passo a Passo)
🔧 5.1. Criar ambiente virtual (opcional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

📦 5.2. Instalar dependências
pip install pandas streamlit matplotlib scikit-learn sqlite3

📥 5.3. Gerar dados simulados (opcional)
python sensor_sim.py


O arquivo será salvo em:

data/sample_interactions.csv

🗄 5.4. Inserir dados no banco SQL
python ingest_to_sql.py

📊 5.5. Rodar análises
python analysis.py


Resultados gerados:

media/interacoes_por_tipo.png

media/touch_dist.png

data/report_summary.json

📈 5.6. Rodar dashboard
streamlit run dashboard_streamlit.py


Abrirá no navegador (localhost:8501).

🤖 5.7. Rodar modelo de Machine Learning
python ml_train.py

📝 6. Documentação Técnica Entregue

A Sprint 2 entrega:

✔ Arquitetura implementada

Representação clara do fluxo:

Sensor → CSV → ingestão SQL → análise → dashboard

✔ Prints de execução

Gráficos

Métricas

Tabelas no dashboard

Execução do ingest e dataset

✔ Fluxo de dados (entrada → processamento → saída)

Entrada: dados brutos dos sensores

Processamento: limpeza, validação, padronização, persistência

Saída: métricas, gráficos, relatório, modelo de ML

🎥 7. Demonstração em Vídeo (Requisito da Sprint)

Incluir no arquivo: media/video_demo_link.txt

Roteiro recomendado (4–5 minutos)

Mostrar sensor_sim.py sendo executado

Rodar ingest_to_sql.py e mostrar banco populado

Rodar analysis.py e exibir gráficos

Abrir painel Streamlit

Executar ml_train.py e mostrar classificação

Conclusão e próximos passos

🏁 8. Conclusão

Este repositório cumpre integralmente os requisitos da Sprint 2:

✔ Sensor → Banco → Análise → Dashboard

✔ Dataset estruturado e limpo

✔ Visualizações e métricas

✔ Modelo de Machine Learning simples

✔ Documentação completa para reprodutibilidade

Em caso de evolução futura (Sprint 3), este protótipo servirá como base para:

Backend completo em FastAPI

Integração com Google Gemini / STT / TTS

Dashboard avançado

Totem físico real (ESP32/Câmera/Touch)
