""import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Configurações iniciais
st.set_page_config(page_title='Análise Mega-Sena com IA', layout='wide')
st.title('Análise Estatística e Inteligência Artificial - Mega-Sena')

# Upload do arquivo
uploaded_file = st.file_uploader('Carregue o histórico da Mega-Sena (.xlsx ou .csv)', type=['xlsx', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success('Arquivo carregado com sucesso!')
    
    # Seleção das colunas de números
    bolas = [col for col in df.columns if 'Bola' in col]
    
    # Análise de frequência
    st.header('Frequência dos Números Sorteados')
    freq = pd.Series(df[bolas].values.ravel()).value_counts().sort_index()
    plt.figure(figsize=(15, 5))
    plt.bar(freq.index, freq.values, color='blue')
    plt.title('Frequência de cada número')
    plt.xlabel('Número')
    plt.ylabel('Frequência')
    st.pyplot(plt)
    
    # Análise de soma dos números
    st.header('Soma dos Números Sorteados')
    df['Soma'] = df[bolas].sum(axis=1)
    st.line_chart(df['Soma'])
    
    # Análise de Paridade
    st.header('Paridade (Pares / Ímpares)')
    pares = df[bolas].applymap(lambda x: x % 2 == 0).sum(axis=1)
    impares = len(bolas) - pares
    paridade_df = pd.DataFrame({'Pares': pares, 'Ímpares': impares})
    st.bar_chart(paridade_df)

    # Gerador Inteligente
    st.header('Gerador Inteligente')
    primos = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    quadrados = [i**2 for i in range(1, 8)]
    
    st.write(f'Números Primos: {primos}')
    st.write(f'Números Quadrados: {quadrados}')
    
    # IA - Random Forest para predição
    st.header('Predição de Combinações')
    st.write('Em desenvolvimento...')

else:
    st.warning('Por favor, carregue um arquivo para iniciar a análise.')
""
