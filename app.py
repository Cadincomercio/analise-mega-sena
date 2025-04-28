# Nova aba: Predição por IA
with tabs[11]:
    st.subheader("Predição Avançada por Machine Learning")

    soma_min_pred = st.number_input("Soma mínima desejada (IA)", value=180, key='soma_min_pred')
    soma_max_pred = st.number_input("Soma máxima desejada (IA)", value=210, key='soma_max_pred')

    if st.button("Treinar Modelo e Sugerir Combinação"):
        with st.spinner('Treinando modelo...'):
            numeros = list(range(1, 61))
            atraso = calcular_atraso(df_filtered, numeros)
            freq_series = pd.Series(df_filtered[['Bola1','Bola2','Bola3','Bola4','Bola5','Bola6']].values.flatten())
            frequencia = freq_series.value_counts().to_dict()

            X = criar_features(df_filtered)
            y = np.random.choice([0,1], size=(X.shape[0],), p=[0.7,0.3])  # Dummy y para treinamento

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)

            combinacao = gerar_combinacao_predita(modelo, atraso, frequencia, soma_min_pred, soma_max_pred)

        st.success(f"Combinação sugerida: {combinacao}")
