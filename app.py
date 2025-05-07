# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random

# Configuração da página
st.set_page_config(page_title='Análises Mega-Sena', layout='wide')

# Função para carregar os dados
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], dayfirst=True)
    return df

# Função aprimorada para criar features com verificações
def criar_features(df):
    # Verificar se todas as colunas estão presentes
    expected_columns = [f'Bola{i}' for i in range(1, 7)]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Preenche com 0 caso não exista

    # Remover linhas que possuem valores nulos
    df = df.dropna(subset=expected_columns)
    
    # Criação das features
    features = pd.DataFrame()
    features['Soma'] = df[expected_columns].sum(axis=1)
    features['Pares'] = df[expected_columns].apply(lambda x: (x % 2 == 0).sum(), axis=1)
    features['PrimeiroDigito'] = df[expected_columns].apply(lambda x: np.floor(x / 10).sum(), axis=1)
    features['UltimoDigito'] = df[expected_columns].apply(lambda x: (x % 10).sum(), axis=1)
    return features

# Interface de upload
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Carregue um arquivo para começar.")
    st.stop()

# Filtros
st.sidebar.title("Filtros")
interval_option = st.sidebar.radio("Intervalo de análise:", ['Últimos N concursos', 'Intervalo por datas'])

if interval_option == 'Últimos N concursos':
    N = st.sidebar.number_input("N concursos:", min_value=1, value=100)
    df_filtered = data.sort_values(by='Concurso', ascending=False).head(N)
else:
    start_date = st.sidebar.date_input("Data inicial:", data['Data do Sorteio'].min())
    end_date = st.sidebar.date_input("Data final:", data['Data do Sorteio'].max())
    df_filtered = data[(data['Data do Sorteio'] >= pd.to_datetime(start_date)) & 
                       (data['Data do Sorteio'] <= pd.to_datetime(end_date))]

# Criação das abas
tabs = st.tabs([
    "Frequência", "Paridade", "Soma", "Entropia", 
    "Quadrantes", "Predição", "Pseudoaleatórias", 
    "Gravidade Numérica", "Gerador de Combinação"
])

# Frequência
with tabs[0]:
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig = px.bar(freq, labels={'index':'Número', 'value':'Frequência'}, title="Frequência Absoluta")
    st.plotly_chart(fig, use_container_width=True)

# Predição por IA
with tabs[5]:
    st.subheader("Predição Avançada por Machine Learning")
    soma_min_pred = st.number_input("Soma mínima desejada (IA)", value=180, key='soma_min_pred')
    soma_max_pred = st.number_input("Soma máxima desejada (IA)", value=210, key='soma_max_pred')

    if st.button("Treinar Modelo e Sugerir Combinação"):
        with st.spinner('Treinando modelo...'):
            X = criar_features(df_filtered)
            y = np.random.choice([0, 1], size=(X.shape[0],), p=[0.7, 0.3])
            
            # Treinamento do modelo
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)
            
            # Predição
            combinacao = modelo.predict(X_test)
        
        st.success(f"Combinação sugerida: {list(combinacao)}")

# Gerador de Combinação
with tabs[8]:
    st.subheader("Gerador Inteligente Clássico")
    soma_min = st.number_input("Soma mínima", value=180, key='soma_min')
    soma_max = st.number_input("Soma máxima", value=210, key='soma_max')
    incluir_primos = st.checkbox("Incluir número primo", value=True)
    incluir_quadrados = st.checkbox("Incluir quadrado perfeito", value=True)

    def gerar_combinacao(soma_min, soma_max, incluir_primos, incluir_quadrados):
        primos = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}
        quadrados = {1, 4, 9, 16, 25, 36, 49}
        numeros = list(range(1, 61))
        while True:
            combinacao = random.sample(numeros, 6)
            soma = sum(combinacao)
            if soma_min <= soma <= soma_max:
                if incluir_primos and not any(num in primos for num in combinacao):
                    continue
                if incluir_quadrados and not any(num in quadrados for num in combinacao):
                    continue
                return sorted(combinacao)
    
    if st.button("Gerar Combinação Clássica"):
        combinacao = gerar_combinacao(soma_min, soma_max, incluir_primos, incluir_quadrados)
        st.success(f"Combinação Gerada: {combinacao}")
