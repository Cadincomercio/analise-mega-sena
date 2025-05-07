
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

# Funções auxiliares
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], dayfirst=True)
    return df

# Carregar arquivo
uploaded_file = st.sidebar.file_uploader("Carregar histórico Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Carregue um arquivo para começar.")
    st.stop()

# Seção de Filtros
st.sidebar.title("Filtros")
interval_option = st.sidebar.radio("Escolha o filtro:", ['Últimos N Concursos', 'Intervalo por Datas'])
winners_only = st.sidebar.checkbox("Apenas concursos com ganhadores", value=False)

# Últimos N Concursos
if interval_option == 'Últimos N Concursos':
    N = st.sidebar.number_input("Quantidade de concursos:", min_value=1, value=100)
    df_filtered = data.sort_values(by='Concurso', ascending=False).head(N)

# Intervalo por Datas
else:
    start_date = st.sidebar.date_input("Data inicial:", data['Data do Sorteio'].min())
    end_date = st.sidebar.date_input("Data final:", data['Data do Sorteio'].max())
    df_filtered = data[(data['Data do Sorteio'] >= pd.to_datetime(start_date)) & 
                       (data['Data do Sorteio'] <= pd.to_datetime(end_date))]

# Apenas concursos com ganhadores
if winners_only:
    df_filtered = df_filtered[df_filtered['Ganhadores 6 acertos'] > 0]

# Exibindo a tabela filtrada
st.write("Concursos carregados:", len(df_filtered))
st.write("Período analisado:", df_filtered['Data do Sorteio'].min().date(), "até", df_filtered['Data do Sorteio'].max().date())
st.dataframe(df_filtered)

# Abas de Análise
tabs = st.tabs(["Frequência", "Paridade", "Soma", "Entropia", "Quadrantes", "Modulares",
                "Diferenças Absolutas", "Primeiro/Último Dígito", "Pseudoaleatórias", 
                "Gravidade Numérica", "Gerador de Combinação", "Predição por IA"])

# Frequência
with tabs[0]:
    nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig = px.bar(freq, labels={'index':'Número', 'value':'Frequência'}, title="Frequência Absoluta")
    st.plotly_chart(fig, use_container_width=True)
