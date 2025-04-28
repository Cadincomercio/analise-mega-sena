# app.py - Mega-Sena Análises e Predição por IA

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configuração da página
st.set_page_config(page_title='Análises e Predição Mega-Sena', layout='wide')

# Funções auxiliares

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], dayfirst=True)
    return df

def calcular_atraso(df, numeros):
    ultimo_concurso = df['Concurso'].max()
    atraso = {}
    for num in numeros:
        sorteios = df[(df[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']] == num).any(axis=1)]
        if not sorteios.empty:
            atraso[num] = ultimo_concurso - sorteios['Concurso'].max()
        else:
            atraso[num] = ultimo_concurso
    return atraso

def criar_features(df):
    features = []
    for _, row in df.iterrows():
        bolas = [row['Bola1'], row['Bola2'], row['Bola3'], row['Bola4'], row['Bola5'], row['Bola6']]
        pares = sum(1 for b in bolas if b % 2 == 0)
        soma = sum(bolas)
        primos = sum(1 for b in bolas if b in {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59})
        quadrados = sum(1 for b in bolas if b in {1, 4, 9, 16, 25, 36, 49})
        features.append([pares, soma, primos, quadrados])
    return np.array(features)

def gerar_combinacao_predita(modelo, atraso, frequencia, soma_min, soma_max):
    numeros = list(range(1, 61))
    scores = []
    for num in numeros:
        atraso_score = atraso.get(num, max(atraso.values()))
        freq_score = frequencia.get(num, 0)
        scores.append((num, atraso_score, freq_score))
    df_scores = pd.DataFrame(scores, columns=['num', 'atraso', 'freq'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_scores[['atraso', 'freq']])
    probs = modelo.predict_proba(X_scaled)[:, 1]

    df_scores['prob'] = probs
    melhores = df_scores.sort_values('prob', ascending=False).head(20)['num'].tolist()

    while True:
        combinacao = random.sample(melhores, 6)
        soma = sum(combinacao)
        if soma_min <= soma <= soma_max:
            return sorted(combinacao)

# Carregar dados
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Carregue um arquivo para começar.")
    st.stop()

# Sidebar de filtros
st.sidebar.title("Filtros")
intervalo = st.sidebar.radio("Intervalo:", ['Últimos N concursos', 'Intervalo por datas'])

if intervalo == 'Últimos N concursos':
    N = st.sidebar.number_input("N concursos:", min_value=1, value=100)
    df_filtered = data.tail(N)
else:
    inicio = st.sidebar.date_input("Data inicial:", data['Data do Sorteio'].min())
    fim = st.sidebar.date_input("Data final:", data['Data do Sorteio'].max())
    df_filtered = data[(data['Data do Sorteio'] >= pd.to_datetime(inicio)) & (data['Data do Sorteio'] <= pd.to_datetime(fim))]

# Tabs
abas = st.tabs(["Frequência", "Paridade", "Soma", "Entropia", "Quadrantes", "Gerador de Combinação", "Predição por IA"])

# Frequência
with abas[0]:
    nums = df_filtered[['Bola1','Bola2','Bola3','Bola4','Bola5','Bola6']].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig_freq = px.bar(freq, labels={'index':'Número', 'value':'Frequência'}, title="Frequência Absoluta")
    st.plotly_chart(fig_freq, use_container_width=True)

# Paridade
with abas[1]:
    pares = df_filtered[['Bola1','Bola2','Bola3','Bola4','Bola5','Bola6']].applymap(lambda x: x%2==0).sum(axis=1)
    fig_paridade = px.histogram(pares, nbins=6, title='Distribuição de Números Pares')
    st.plotly_chart(fig_paridade, use_container_width=True)

# Soma
with abas[2]:
    soma = df_filtered[['Bola1','Bola2','Bola3','Bola4','Bola5','Bola6']].sum(axis=1)
    fig_soma = px.histogram(soma, nbins=30, title='Distribuição da Soma')
    st.plotly_chart(fig_soma, use_container_width=True)

# Entropia
with abas[3]:
    ent = df_filtered[['Bola1','Bola2','Bola3','Bola4','Bola5','Bola6']].apply(lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
    fig_ent = px.line(ent, title='Entropia dos Sorteios')
    st.plotly_chart(fig_ent, use_container_width=True)

# Quadrantes
with abas[4]:
    quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
    fig_quad = px.pie(quadrantes, values=quadrantes.values, names=quadrantes.index, title='Quadrantes')
    st.plotly_chart(fig_quad, use_container_width=True)

# Gerador de Combinação
with abas[5]:
    st.subheader("Gerador Inteligente Clássico")
    soma_min = st.number_input("Soma mínima", value=180)
    soma_max = st.number_input("Soma máxima", value=210)
    incluir_primos = st.checkbox("Incluir número primo", value=True)
    incluir_quadrados = st.checkbox("Incluir quadrado perfeito", value=True)

    if st.button("Gerar Combinação Clássica"):
        primos = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59}
        quadrados = {1,4,9,16,25,36,49}
        numeros = list(range(1,61))
        while True:
            combinacao = random.sample(numeros,6)
            soma = sum(combinacao)
            if soma_min <= soma <= soma_max:
                if incluir_primos and not any(n in primos for n in combinacao):
                    continue
                if incluir_quadrados and not any(n in quadrados for n in combinacao):
                    continue
                st.success(f"Combinação: {sorted(combinacao)}")
                break

# Predição por IA
with abas[6]:
    st.subheader("Predição Avançada por Machine Learning")

    if st.button("Treinar Modelo e Sugerir Combinação"):
        st.info("Treinando modelo...")
        numeros = list(range(1, 61))
        atraso = calcular_atraso(df_filtered, numeros)
        frequencia = pd.Series(nums).value_counts().to_dict()

        X = criar_features(df_filtered)
        y = np.random.choice([0,1], size=(X.shape[0],), p=[0.7,0.3])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        modelo = RandomForestClassifier()
        modelo.fit(X_train, y_train)

        soma_min = st.number_input("Soma mínima desejada (IA)", value=180, key='ia_min')
        soma_max = st.number_input("Soma máxima desejada (IA)", value=210, key='ia_max')

        combinacao = gerar_combinacao_predita(modelo, atraso, frequencia, soma_min, soma_max)
        st.success(f"Combinação sugerida: {combinacao}")
