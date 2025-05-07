# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

# Carregar o arquivo
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Carregue um arquivo para começar.")
    st.stop()

# Configuração dos Filtros
st.sidebar.title("Filtros")
interval_option = st.sidebar.radio("Escolha o filtro:", ['Últimos N Concursos', 'Intervalo por Datas'])

if interval_option == 'Últimos N Concursos':
    N = st.sidebar.number_input("Quantidade de concursos:", min_value=1, value=100)
    df_filtered = data.sort_values(by='Concurso', ascending=False).head(N)

elif interval_option == 'Intervalo por Datas':
    start_date = st.sidebar.date_input("Data inicial:", data['Data do Sorteio'].min())
    end_date = st.sidebar.date_input("Data final:", data['Data do Sorteio'].max())
    if start_date < pd.to_datetime('1996-03-11'):
        st.error("Data inicial não pode ser anterior a 11/03/1996.")
    elif end_date > data['Data do Sorteio'].max():
        st.error("Data final não pode ser superior à data do último sorteio.")
    else:
        df_filtered = data[(data['Data do Sorteio'] >= pd.to_datetime(start_date)) & 
                           (data['Data do Sorteio'] <= pd.to_datetime(end_date))]

# Filtro de concursos premiados
winners_only = st.sidebar.checkbox("Apenas concursos com ganhadores", value=False)
if winners_only:
    df_filtered = df_filtered[df_filtered['Ganhadores 6 acertos'] > 0]

# Verificar se há dados
if df_filtered.empty:
    st.warning("Nenhum resultado encontrado para os filtros selecionados.")
    st.stop()

# Tabs para visualização
tabs = st.tabs([
    "Frequência", "Paridade", "Soma", "Entropia",
    "Quadrantes", "Modulares", "Diferenças Absolutas",
    "Primeiro/Último Dígito", "Pseudoaleatórias",
    "Gravidade Numérica", "Gerador de Combinação", "Predição por IA"
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
    ent = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
    fig_ent = px.line(ent, title='Entropia dos Sorteios')
    st.plotly_chart(fig_ent, use_container_width=True)

# Quadrantes
with tabs[4]:
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
    fig_quad = px.pie(values=quadrantes.values, names=quadrantes.index, title='Distribuição por Quadrantes')
    st.plotly_chart(fig_quad, use_container_width=True)
