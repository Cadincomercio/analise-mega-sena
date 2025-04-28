# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.stats import entropy

# Configuração da página
st.set_page_config(page_title='Análises Mega-Sena', layout='wide')

# Função para carregar dados
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
    return df

# Carregar dados
uploaded_file = st.sidebar.file_uploader("Carregar arquivo do histórico da Mega-Sena", type=['xlsx', 'csv'])
if uploaded_file:
    data = load_data(uploaded_file)
else:
    st.warning("Carregue um arquivo para começar.")
    st.stop()

# Sidebar para filtros
st.sidebar.title("Filtros")
interval_option = st.sidebar.radio("Intervalo de análise:", ['Últimos N concursos', 'Intervalo por datas'])

if interval_option == 'Últimos N concursos':
    N = st.sidebar.number_input("Quantidade de últimos concursos:", min_value=1, value=100)
    df_filtered = data.tail(N)
else:
    start_date = st.sidebar.date_input("Data inicial:", data['Data'].min())
    end_date = st.sidebar.date_input("Data final:", data['Data'].max())
    df_filtered = data[(data['Data'] >= pd.to_datetime(start_date)) & (data['Data'] <= pd.to_datetime(end_date))]

winners_only = st.sidebar.checkbox("Analisar apenas concursos com ganhadores", value=False)
if winners_only:
    df_filtered = df_filtered[df_filtered['Ganhadores_Sena'] > 0]

# Tabs para diferentes análises
tabs = st.tabs(["Frequência Absoluta", "Análise de Soma", "Paridade", "Diferenças Absolutas", "Primeiro/Último Dígito", "Modulares", "Entropia", "Quadrantes", "Pseudoaleatórias", "Gravidade Numérica"])

# 1. Frequência Absoluta
with tabs[0]:
    nums = df_filtered.iloc[:, 2:8].values.flatten()
    freq = pd.Series(nums).value_counts().sort_index()
    fig_freq = px.bar(freq, labels={'index':'Número', 'value':'Frequência'}, title="Frequência Absoluta dos Números")
    st.plotly_chart(fig_freq, use_container_width=True)

# 2. Análise da Soma
with tabs[1]:
    df_filtered['Soma'] = df_filtered.iloc[:, 2:8].sum(axis=1)
    stats_soma = df_filtered['Soma'].describe()
    st.write(stats_soma)
    fig_soma = px.histogram(df_filtered, x='Soma', nbins=30, title='Distribuição das Somas dos Números')
    st.plotly_chart(fig_soma, use_container_width=True)

# 3. Paridade
with tabs[2]:
    pares = df_filtered.iloc[:, 2:8].applymap(lambda x: x % 2 == 0).sum(axis=1)
    df_filtered['Pares'] = pares
    fig_paridade = px.histogram(df_filtered, x='Pares', title='Quantidade de Números Pares por Sorteio')
    st.plotly_chart(fig_paridade, use_container_width=True)

# 4. Diferenças Absolutas
with tabs[3]:
    diffs = df_filtered.iloc[:, 2:8].apply(lambda x: np.diff(np.sort(x)), axis=1).explode()
    fig_diffs = px.histogram(diffs, nbins=20, title='Distribuição das Diferenças Absolutas entre Números')
    st.plotly_chart(fig_diffs, use_container_width=True)

# 5. Primeiro e Último Dígito
with tabs[4]:
    primeiro_digito = pd.Series(nums // 10).value_counts().sort_index()
    ultimo_digito = pd.Series(nums % 10).value_counts().sort_index()
    st.write("Primeiro Dígito:", primeiro_digito)
    st.write("Último Dígito:", ultimo_digito)

# 6. Modulares
with tabs[5]:
    mod_5 = pd.Series(nums % 5).value_counts().sort_index()
    mod_7 = pd.Series(nums % 7).value_counts().sort_index()
    mod_10 = pd.Series(nums % 10).value_counts().sort_index()
    st.write("Modulo 5:", mod_5)
    st.write("Modulo 7:", mod_7)
    st.write("Modulo 10:", mod_10)

# 7. Entropia
with tabs[6]:
    ent = df_filtered.iloc[:, 2:8].apply(lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
    st.write("Média de Entropia:", ent.mean())
    fig_ent = px.line(ent, title='Entropia dos Sorteios')
    st.plotly_chart(fig_ent, use_container_width=True)

# 8. Quadrantes
with tabs[7]:
    quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
    fig_quad = px.pie(quadrantes, values=quadrantes.values, names=quadrantes.index, title='Distribuição por Quadrantes')
    st.plotly_chart(fig_quad, use_container_width=True)

# 9. Séries Pseudoaleatórias
with tabs[8]:
    primos = [x for x in nums if all(x % i for i in range(2, int(np.sqrt(x)) + 1))]
    quadrados = [x for x in nums if np.sqrt(x).is_integer()]
    fibonacci = [1,2,3,5,8,13,21,34,55]
    fib_nums = [x for x in nums if x in fibonacci]
    st.write("Primos:", len(primos), "Quadrados:", len(quadrados), "Fibonacci:", len(fib_nums))

# 10. Gravidade Numérica
with tabs[9]:
    gravity = df_filtered.iloc[:, 2:8].apply(np.mean, axis=1)
    st.write("Centro de Massa Numérico Médio:", gravity.mean())
    fig_gravity = px.histogram(gravity, title='Distribuição da Gravidade Numérica dos Sorteios')
    st.plotly_chart(fig_gravity, use_container_width=True)
