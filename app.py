import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import entropy
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Configuração da página
st.set_page_config(page_title='Análises Mega-Sena', layout='wide')

# Função para carregar os dados
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], dayfirst=True)
    return df

# Interface de upload
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
    st.success("✅ Dados carregados com sucesso!")
else:
    st.warning("⚠️ Carregue um arquivo para começar.")
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

# Filtro de apenas ganhadores
winners_only = st.sidebar.checkbox("Analisar apenas concursos com ganhadores", value=False)
if winners_only:
    df_filtered = df_filtered[df_filtered['Ganhadores 6 acertos'] > 0]

# Criação das abas
tabs = st.tabs([
    "Frequência", "Paridade", "Soma", "Entropia", 
    "Quadrantes", "Modulares", "Diferenças Absolutas", 
    "Primeiro/Último Dígito", "Pseudoaleatórias", 
    "Gravidade Numérica", "Gerador de Combinação", "Predição por IA"
])

# ------------------------
# 🎯 Frequência
# ------------------------
with tabs[0]:
    st.subheader("Frequência Absoluta dos Números")
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig = px.bar(freq, labels={'index': 'Número', 'value': 'Frequência'}, title="Frequência Absoluta")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# ⚖️ Paridade
# ------------------------
with tabs[1]:
    st.subheader("Distribuição de Pares nos Sorteios")
    pares = df_filtered[[f'Bola{i}' for i in range(1,]()]()_
