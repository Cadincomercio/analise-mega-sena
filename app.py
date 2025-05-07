# app.py
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
    try:
        df = pd.read_excel(file, engine='openpyxl')
        df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], dayfirst=True)
        st.success("Dados carregados com sucesso!")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return pd.DataFrame()

# Interface de upload
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
    if data.empty:
        st.error("Erro: A planilha está vazia ou corrompida.")
        st.stop()
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

# Validação
if df_filtered.empty:
    st.error("Não existem resultados para os filtros aplicados.")
    st.stop()
else:
    st.success(f"{len(df_filtered)} concursos selecionados para análise.")

# Criação das abas
tabs = st.tabs([
    "Entropia", "Quadrantes", "Modulares", "Diferenças Absolutas",
    "Primeiro/Último Dígito", "Pseudoaleatórias", 
    "Gravidade Numérica", "Gerador de Combinação", "Predição por IA"
])

# Entropia
with tabs[0]:
    try:
        ent = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(
            lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
        fig_ent = px.line(ent, title='Entropia dos Sorteios')
        st.plotly_chart(fig_ent, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Entropia: {e}")

# Quadrantes
with tabs[1]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
        fig_quad = px.pie(values=quadrantes.values, names=quadrantes.index, title='Distribuição por Quadrantes')
        st.plotly_chart(fig_quad, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Quadrantes: {e}")

# Modulares
with tabs[2]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        mod_5 = pd.Series(nums % 5).value_counts().sort_index()
        mod_7 = pd.Series(nums % 7).value_counts().sort_index()
        mod_10 = pd.Series(nums % 10).value_counts().sort_index()
        st.write("Módulo 5:", mod_5)
        st.write("Módulo 7:", mod_7)
        st.write("Módulo 10:", mod_10)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Modulares: {e}")

# Diferenças Absolutas
with tabs[3]:
    try:
        diffs = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: np.diff(np.sort(x)), axis=1).explode()
        fig_diffs = px.histogram(diffs, nbins=20, title='Diferenças Absolutas entre Números')
        st.plotly_chart(fig_diffs, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Diferenças Absolutas: {e}")

# Primeiro/Último Dígito
with tabs[4]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        primeiro_digito = pd.Series(nums // 10).value_counts().sort_index()
        ultimo_digito = pd.Series(nums % 10).value_counts().sort_index()
        st.write("Primeiro Dígito:", primeiro_digito)
        st.write("Último Dígito:", ultimo_digito)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Primeiro/Último Dígito: {e}")

# Pseudoaleatórias
with tabs[5]:
    try:
        st.write("Em construção...")
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Pseudoaleatórias: {e}")

# Gravidade Numérica
with tabs[6]:
    try:
        gravity = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(np.mean, axis=1)
        fig_gravity = px.histogram(gravity, title='Gravidade Numérica dos Sorteios')
        st.plotly_chart(fig_gravity, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao gerar a aba de Gravidade Numérica: {e}")
