
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

def calcular_atraso(df, numeros):
    ultimos = {n: 0 for n in numeros}
    atraso = {n: 0 for n in numeros}
    for index, row in df.iterrows():
        bolas = [row[f'Bola{i}'] for i in range(1,7)]
        for n in numeros:
            if n not in bolas:
                atraso[n] += 1
            else:
                atraso[n] = 0
        ultimos = atraso.copy()
    return ultimos

def criar_features(df):
    features = pd.DataFrame()
    features['Soma'] = df[[f'Bola{i}' for i in range(1,7)]].sum(axis=1)
    features['Pares'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: (x % 2 == 0).sum(), axis=1)
    features['PrimeiroDigito'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: np.floor(x/10).sum(), axis=1)
    features['UltimoDigito'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: (x % 10).sum(), axis=1)
    return features

def gerar_combinacao_predita(modelo, atraso, frequencia, soma_min, soma_max):
    numeros = list(range(1, 61))
    scores = {}
    for n in numeros:
        atraso_score = atraso.get(n, 0)
        freq_score = frequencia.get(n, 0)
        scores[n] = atraso_score * 0.6 + freq_score * 0.4
    melhores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selecionados = [num for num, score in melhores[:15]]
    for _ in range(1000):
        combinacao = sorted(random.sample(selecionados, 6))
        if soma_min <= sum(combinacao) <= soma_max:
            return combinacao
    return sorted(random.sample(selecionados, 6))
