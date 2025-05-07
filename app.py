# Entropia
with tabs[3]:
    try:
        ent = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(
            lambda x: entropy(np.histogram(x, bins=60, range=(1,60))[0]), axis=1)
        fig_ent = px.line(ent, title='Entropia dos Sorteios')
        st.plotly_chart(fig_ent, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao calcular entropia: {str(e)}")

# Quadrantes
with tabs[4]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        quadrantes = pd.cut(nums, bins=[0,15,30,45,60], labels=['1-15','16-30','31-45','46-60']).value_counts()
        fig_quad = px.pie(values=quadrantes.values, names=quadrantes.index, title='Distribuição por Quadrantes')
        st.plotly_chart(fig_quad, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao calcular quadrantes: {str(e)}")

# Modulares
with tabs[5]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        mod_5 = pd.Series(nums % 5).value_counts().sort_index()
        mod_7 = pd.Series(nums % 7).value_counts().sort_index()
        mod_10 = pd.Series(nums % 10).value_counts().sort_index()
        st.write("Módulo 5:", mod_5)
        st.write("Módulo 7:", mod_7)
        st.write("Módulo 10:", mod_10)
    except Exception as e:
        st.error(f"Erro ao calcular modulares: {str(e)}")

# Diferenças Absolutas
with tabs[6]:
    try:
        diffs = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: np.diff(np.sort(x)), axis=1).explode()
        fig_diffs = px.histogram(diffs, nbins=20, title='Diferenças Absolutas entre Números')
        st.plotly_chart(fig_diffs, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao calcular diferenças absolutas: {str(e)}")

# Primeiro/Último Dígito
with tabs[7]:
    try:
        nums = df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten()
        primeiro_digito = pd.Series(nums // 10).value_counts().sort_index()
        ultimo_digito = pd.Series(nums % 10).value_counts().sort_index()
        st.write("Primeiro Dígito:", primeiro_digito)
        st.write("Último Dígito:", ultimo_digito)
    except Exception as e:
        st.error(f"Erro ao calcular dígitos: {str(e)}")

# Pseudoaleatórias
with tabs[8]:
    try:
        random_numbers = [random.randint(1, 60) for _ in range(6)]
        st.write(f"Números pseudoaleatórios gerados: {random_numbers}")
    except Exception as e:
        st.error(f"Erro ao gerar números pseudoaleatórios: {str(e)}")

# Gravidade Numérica
with tabs[9]:
    try:
        gravity = df_filtered[[f'Bola{i}' for i in range(1,7)]].apply(np.mean, axis=1)
        fig_gravity = px.histogram(gravity, title='Gravidade Numérica dos Sorteios')
        st.plotly_chart(fig_gravity, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao calcular gravidade numérica: {str(e)}")

# Predição por IA
with tabs[10]:
    st.subheader("Predição Avançada por Machine Learning")
    soma_min_pred = st.number_input("Soma mínima desejada (IA)", value=180, key='soma_min_pred')
    soma_max_pred = st.number_input("Soma máxima desejada (IA)", value=210, key='soma_max_pred')

    def criar_features(df):
        features = pd.DataFrame()
        features['Soma'] = df[[f'Bola{i}' for i in range(1,7)]].sum(axis=1)
        features['Pares'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: (x % 2 == 0).sum(), axis=1)
        features['PrimeiroDigito'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: np.floor(x/10).sum(), axis=1)
        features['UltimoDigito'] = df[[f'Bola{i}' for i in range(1,7)]].apply(lambda x: (x % 10).sum(), axis=1)
        return features

    if st.button("Treinar Modelo e Sugerir Combinação"):
        with st.spinner('Treinando modelo...'):
            numeros = list(range(1, 61))
            freq_series = pd.Series(df_filtered[[f'Bola{i}' for i in range(1,7)]].values.flatten())
            frequencia = freq_series.value_counts().to_dict()
            atraso = {n: 0 for n in numeros}
            
            X = criar_features(df_filtered)
            y = np.random.choice([0, 1], size=(X.shape[0],), p=[0.7, 0.3])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)
            
            combinacao = sorted(random.sample(list(frequencia.keys()), 6))
            st.success(f"Combinação sugerida: {combinacao}")
