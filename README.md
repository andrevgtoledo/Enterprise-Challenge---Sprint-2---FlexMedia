Totem IA — Sprint 2
Projeto Flexmedia Challenge – FIAP
Continuação da Arquitetura Definida na Sprint 1

Este repositório contém a implementação prática da Sprint 2, demonstrando a integração entre sensores (simulados), banco de dados SQL, análise estatística e visualização simples em Python, além de um modelo supervisionado básico.

A Sprint 2 transforma a arquitetura definida anteriormente em execução real, formando um pipeline funcional fim a fim.

1. Objetivos da Sprint 2

Demonstrar a integração sensor → armazenamento SQL → análise → visualização.

Registrar e estruturar dados de interação do Totem IA.

Criar visualizações iniciais utilizando Python (Streamlit).

Aplicar Machine Learning supervisionado para classificar tipos de interação.

Garantir dados limpos, padronizados e prontos para análise.

2. Arquitetura Implementada (Sprint 2)
Simulador de sensores (Python)
        ↓
Arquivo CSV (sample_interactions.csv)
        ↓
Ingestão e estruturação em SQL (SQLite)
        ↓
Limpeza, agregação e análises (Pandas)
        ↓
Dashboard simples em Python (Streamlit)
        ↓
Modelo supervisionado (RandomForest) – toque curto vs longo

3. Estrutura do Repositório
totem-ia-sprint2/
│
├── data/
│   ├── sample_interactions.csv       (dados simulados)
│   ├── flexmedia.sqlite              (banco SQL estruturado)
│   └── report_summary.json           (sumário das análises)
│
├── media/
│   ├── interacoes_por_tipo.png       (gráfico gerado)
│   ├── touch_dist.png                (gráfico gerado)
│
├── sensor_sim.py                     (simulador de sensores)
├── ingest_to_sql.py                  (ingestão para SQL)
├── analysis.py                       (análises e gráficos)
├── dashboard_streamlit.py            (dashboard web)
├── ml_train.py                       (modelo ML)
└── README.md

4. Scripts da Sprint 2

Abaixo está o código completo de cada módulo implementado nesta Sprint.

4.1. Simulador de Sensores — sensor_sim.py

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


4.2. Ingestão para SQL — ingest_to_sql.py
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
                cur.execute('INSERT INTO sessao (id, idioma, inicio) VALUES (?,?,?)',
                            (sess_id, row['language'], row['timestamp']))

            cur.execute(
            '''INSERT INTO interacao 
            (sessao_id, timestamp, sensor_id, tipo, pergunta, resposta, content_id, duration, value)
            VALUES (?,?,?,?,?,?,?,?,?)''',
            (sess_id, row['timestamp'], row['sensor_id'], row['event_type'], 
             row['pergunta'], row['resposta'], row['content_id'], 
             float(row['duration'] or 0), int(row['value'] or 0)))

    conn.commit()
    conn.close()
    print("Ingestão concluída.")

if __name__ == "__main__":
    ingest(CSV, DB)

4.3. Análises e Gráficos — analysis.py
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

4.4. Dashboard em Streamlit — dashboard_streamlit.py
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

4.5. Machine Learning Supervisionado — ml_train.py
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DB = "data/flexmedia.sqlite"
conn = sqlite3.connect(DB)

df = pd.read_sql_query("SELECT duration FROM interacao WHERE duration > 0", conn)
df['label'] = (df['duration'] > 0.5).astype(int)

X = df[['duration']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

5. Como Executar o Projeto
Instalar dependências
pip install pandas streamlit matplotlib scikit-learn

Gerar dados simulados
python sensor_sim.py

Ingerir no banco SQL
python ingest_to_sql.py

Executar análises e gerar gráficos
python analysis.py

Abrir dashboard
streamlit run dashboard_streamlit.py

Executar Machine Learning supervisionado
python ml_train.py

6. Fluxo Completo da Sprint 2
Sensores simulados
→ CSV gerado
→ Banco SQL estruturado
→ Limpeza e análise
→ Geração de gráficos
→ Dashboard
→ Modelo supervisionado

Conclusão

A Sprint 2 entrega um pipeline funcional que conecta sensores simulados, banco SQL, análises quantitativas, visualização e Machine Learning.
Todos os requisitos técnicos e funcionais da Sprint foram atendidos com sucesso e contribuem diretamente para a evolução do Totem IA nas próximas etapas.
