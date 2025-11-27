🧠 Totem IA — Sprint 2
Flexmedia Challenge — FIAP
Integração entre Sensores, Banco de Dados, Análise Estatística e Machine Learning

Este repositório apresenta a entrega completa da Sprint 2 do projeto Totem IA, dando continuidade ao planejamento técnico desenvolvido na Sprint 1.
A etapa atual materializa a integração entre sensores (simulados), armazenamento SQL, tratamento de dados, visualização analítica e aprendizado supervisionado aplicado ao contexto do Totem Flexmedia.

🎯 1. Objetivos da Sprint 2

Implementar um pipeline funcional dados → SQL → análise → visualização.

Registrar e estruturar informações simuladas de sensores associados ao Totem IA.

Criar métricas e gráficos iniciais para acompanhamento do uso.

Demonstrar um exemplo de Machine Learning supervisionado com dataset simples.

Garantir integridade, limpeza e padronização dos dados coletados.

Validar a arquitetura definida na Sprint 1 em um ambiente prático.

🏛️ 2. Arquitetura Desenvolvida
Simulador de sensores
        ↓
CSV de eventos do totem
        ↓
Ingestão para banco SQL (SQLite)
        ↓
Tratamento e análise (Pandas)
        ↓
Geração de gráficos (Matplotlib)
        ↓
Dashboard interativo (Streamlit)
        ↓
Modelo supervisionado (RandomForest)


Esta arquitetura representa uma versão prática e reduzida do fluxo de dados real previsto para o Totem Flexmedia.

📁 3. Estrutura do Repositório
Enterprise-Challenge---Sprint-2---FlexMedia/
│
├── data/
│   ├── sample_interactions.csv       → Dados simulados do totem
│   ├── flexmedia.sqlite              → Banco SQL estruturado
│   └── report_summary.json           → Métricas geradas na análise
│
├── media/
│   ├── interacoes_por_tipo.png       → Gráfico de tipos de interação
│   ├── touch_dist.png                → Gráfico de duração dos toques
│
├── sensor_sim.py                     → Simulador de sensores
├── ingest_to_sql.py                  → Ingestão e modelagem em SQL
├── analysis.py                       → Análises estatísticas
├── dashboard_streamlit.py            → Dashboard analítico
├── ml_train.py                       → Modelo ML supervisionado
└── README.md

🧩 4. Módulos do Projeto

A seguir estão todos os scripts utilizados na Sprint 2.

📌 4.1. Simulação de Sensores

Arquivo: sensor_sim.py

import csv, time, random, uuid
from datetime import datetime
import os

CSV_OUT = 'data/sample_interactions.csv'

def random_interaction(session_id):
    sensor = random.choice(['touch_1','touch_2','pres_1'])
    event = random.choices(['touch','presence','qr'], weights=[0.7,0.25,0.05])[0]
    duration = round(random.uniform(0.05,4.0) if event=='touch' else 0.0,3)
    value = 1 if event in ('touch','presence') else 0
    lang = random.choices(['pt-BR','en-US','es-ES'], weights=[0.7,0.2,0.1])[0]
    content_id = str(uuid.uuid4()) if random.random() < 0.6 else None
    pergunta, resposta = None, None
    if event == 'touch' and random.random() < 0.4:
        pergunta = random.choice(["Qual é esse animal?","Horário?","Onde fica o banheiro?"])
        resposta = "Resposta simulada."
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'sensor_id': sensor,
        'event_type': event,
        'duration': duration,
        'value': value,
        'session_anon_id': session_id,
        'language': lang,
        'content_id': content_id,
        'pergunta': pergunta,
        'resposta': resposta
    }

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = None
        for s in range(50):
            session_id = str(uuid.uuid4())
            for i in range(random.randint(3,20)):
                row = random_interaction(session_id)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                time.sleep(0.01)
    print('CSV gerado:', CSV_OUT)

📌 4.2. Ingestão e Banco SQL

Arquivo: ingest_to_sql.py

import sqlite3, csv

DB = 'data/flexmedia.sqlite'
CSV = 'data/sample_interactions.csv'

SCHEMA = '''
CREATE TABLE IF NOT EXISTS sessao (
  id TEXT PRIMARY KEY,
  idioma TEXT,
  inicio TEXT,
  fim TEXT,
  duracao_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS interacao (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sessao_id TEXT,
  timestamp TEXT,
  sensor_id TEXT,
  tipo TEXT,
  pergunta TEXT,
  resposta TEXT,
  content_id TEXT,
  duration REAL,
  value INTEGER
);

CREATE INDEX IF NOT EXISTS idx_interacao_sessao ON interacao(sessao_id);
'''

def ingest(csv_path, db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sess_id = row['session_anon_id']

            cur.execute('SELECT id FROM sessao WHERE id=?',(sess_id,))
            if cur.fetchone() is None:
                cur.execute(
                    'INSERT INTO sessao (id, idioma, inicio) VALUES (?,?,?)',
                    (sess_id, row['language'], row['timestamp'])
                )

            cur.execute(
                '''INSERT INTO interacao 
                (sessao_id, timestamp, sensor_id, tipo, pergunta, resposta, content_id, duration, value)
                VALUES (?,?,?,?,?,?,?,?,?)''',
                (sess_id, row['timestamp'], row['sensor_id'], row['event_type'], 
                 row['pergunta'], row['resposta'], row['content_id'], 
                 float(row['duration'] or 0), int(row['value'] or 0))
            )

    conn.commit()
    conn.close()
    print("Ingestão concluída.")

if __name__ == "__main__":
    ingest(CSV, DB)

📌 4.3. Análises Estatísticas

Arquivo: analysis.py

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

DB = 'data/flexmedia.sqlite'
conn = sqlite3.connect(DB)

df = pd.read_sql_query('SELECT * FROM interacao', conn, parse_dates=['timestamp'])
df = df.drop_duplicates(subset=['timestamp','sensor_id','tipo'])

df['touch_type'] = df['duration'].apply(
    lambda d: 'none' if d==0 else ('short' if d <= 0.5 else 'long')
)

os.makedirs('media', exist_ok=True)

plt.figure()
df['tipo'].value_counts().plot(kind='bar')
plt.title('Interações por Tipo')
plt.tight_layout()
plt.savefig('media/interacoes_por_tipo.png')

plt.figure()
df['touch_type'].value_counts().plot(kind='bar')
plt.title('Distribuição de Toques (Short vs Long)')
plt.tight_layout()
plt.savefig('media/touch_dist.png')

report = {
    "total_interacoes": len(df),
    "interacoes_por_tipo": df['tipo'].value_counts().to_dict(),
    "duracao_media": float(df['duration'].mean() or 0)
}

with open('data/report_summary.json','w',encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("Análises concluídas.")

📌 4.4. Dashboard

Arquivo: dashboard_streamlit.py

import streamlit as st
import sqlite3
import pandas as pd

DB = "data/flexmedia.sqlite"

st.title("Dashboard Totem IA — Sprint 2")

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM interacao", conn, parse_dates=['timestamp'])

st.metric("Total de Interações", len(df))

st.subheader("Interações por Tipo")
st.bar_chart(df['tipo'].value_counts())

st.subheader("Últimas 20 Interações")
st.dataframe(df.sort_values('timestamp', ascending=False).head(20))

📌 4.5. Machine Learning Supervisionado

Arquivo: ml_train.py

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DB = "data/flexmedia.sqlite"
conn = sqlite3.connect(DB)

df = pd.read_sql_query(
    "SELECT duration FROM interacao WHERE duration > 0",
    conn
)

df['label'] = (df['duration'] > 0.5).astype(int)

X = df[['duration']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

▶️ 5. Como Executar
1. Instalar dependências
pip install pandas streamlit matplotlib scikit-learn

2. Gerar os dados simulados
python sensor_sim.py

3. Carregar os dados no banco SQL
python ingest_to_sql.py

4. Executar a análise exploratória
python analysis.py

5. Abrir o dashboard
streamlit run dashboard_streamlit.py

6. Treinar o modelo de Machine Learning
python ml_train.py

🏁 6. Conclusão

A Sprint 2 comprova a viabilidade da integração entre sensores físicos (simulados), banco de dados SQL, análises estatísticas e modelos de Machine Learning aplicados ao Totem IA.
Os resultados obtidos consolidam a fundação técnica necessária para as próximas fases do projeto e validam a arquitetura proposta na Sprint 1.
