# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import entropy
import random

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

# Filtro de apenas ganhadores
winners_only = st.sidebar.checkbox("Analisar apenas concursos com ganhadores", value=False)
if winners_only:
    df_filtered = df_filtered[df_filtered['Ganhadores 6 acertos'] > 0]

# Criação das abas
tabs = st.tabs([
    "Frequência", "Paridade", "Soma", "Entropia", 
    "Quadrantes", "Modulares", "Diferenças Absolutas", 
    "Primeiro/Último Dígito", "Pseudoaleatórias", 
    "Gravidade Numérica"
])

# Frequência
with tabs[0]:
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig = px.bar(freq, labels={'index':'Número', 'value':'Frequência'}, title="Frequência Absoluta")
    st.plotly_chart(fig, use_container_width=True)

# Paridade
with tabs[1]:
    pares = df_filtered[[f'Bola{i}' for i in range(1,7)]].applymap(lambda x: x % 2 == 0).sum(axis=1)
    fig_pares = px.histogram(pares, nbins=6, title='Distribuição de Pares nos Sorteios')
    st.plotly_chart(fig_pares, use_container_width=True)

# Soma
with tabs[2]:
    df_filtered['Soma'] = df_filtered[[f'Bola{i}' for i in range(1,7)]].sum(axis=1)
    fig_soma = px.histogram(df_filtered, x='Soma', nbins=30, title='Distribuição da Soma dos Números')
    st.plotly_chart(fig_soma, use_container_width=True)

# Entropia
with tabs[3]:
    ent = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(
        lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
    fig_ent = px.line(ent, title='Entropia dos Sorteios')
    st.plotly_chart(fig_ent, use_container_width=True)

# Quadrantes
with tabs[4]:
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
    fig_quad = px.pie(values=quadrantes.values, names=quadrantes.index, title='Distribuição por Quadrantes')
    st.plotly_chart(fig_quad, use_container_width=True)

# Modulares
with tabs[5]:
    mod_5 = pd.Series(nums % 5).value_counts().sort_index()
    mod_7 = pd.Series(nums % 7).value_counts().sort_index()
    mod_10 = pd.Series(nums % 10).value_counts().sort_index()
    st.write("Módulo 5:", mod_5)
    st.write("Módulo 7:", mod_7)
    st.write("Módulo 10:", mod_10)

# Diferenças Absolutas
with tabs[6]:
    diffs = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: np.diff(np.sort(x)), axis=1).explode()
    fig_diffs = px.histogram(diffs, nbins=20, title='Diferenças Absolutas entre Números')
    st.plotly_chart(fig_diffs, use_container_width=True)

# Primeiro/Último Dígito
with tabs[7]:
    primeiro_digito = pd.Series(nums // 10).value_counts().sort_index()
    ultimo_digito = pd.Series(nums % 10).value_counts().sort_index()
    st.write("Primeiro Dígito:", primeiro_digito)
    st.write("Último Dígito:", ultimo_digito)

# Gravidade Numérica
with tabs[8]:
    gravity = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(np.mean, axis=1)
    fig_gravity = px.histogram(gravity, title='Gravidade Numérica dos Sorteios')
    st.plotly_chart(fig_gravity, use_container_width=True)

# Gerador de Combinação
with tabs[9]:
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
            combinacao = random.sample(numeros,

