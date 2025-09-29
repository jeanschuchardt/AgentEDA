import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

# === FIXAR API KEY DO GROQ AQUI ===
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Configuração da página
st.set_page_config(page_title="Agente EDA", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA)")

st.write("Faça upload de um arquivo CSV para começar.")

uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

if "historico" not in st.session_state:
    st.session_state.historico = []

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Criar abas (colocando Pergunte ao Agente primeiro)
    aba6, aba1, aba2, aba3, aba4, aba5, aba7 = st.tabs([
        "🤖 Pergunte ao Agente",
        "📈 Estatísticas",
        "📊 Visualizações",
        "🔍 Frequência",
        "⏳ Tendências Temporais",
        "⚠️ Outliers",
        "📌 Conclusões"
    ])

    # --- Perguntas ---
# --- Perguntas ---
    with aba6:
        st.subheader("🤖 Pergunte ao agente")
        pergunta = st.text_input("Digite sua pergunta:")

        if pergunta:
            try:
                modelo = "llama-3.1-8b-instant"  # rápido e leve
                # modelo = "llama-3.3-70b-versatile"
                # modelo = "mixtral-8x7b-32768"
                # modelo = "gemma2-9b-it"

                llm = ChatGroq(api_key=GROQ_API_KEY, model=modelo)

                agente = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=False,
                    allow_dangerous_code=True
                )

                # indicador de processamento
                with st.spinner("⏳ Processando sua pergunta..."):
                    resposta = agente.invoke(pergunta)

                st.success("✅ Resposta pronta!")
                st.write("👉", resposta)

            except Exception as e:
                st.error("⚠️ O agente não conseguiu interpretar a resposta do modelo.")
                st.write("Erro técnico:", str(e))

                # fallback para perguntas sobre fraude
                if "fraude" in pergunta.lower() or "influência" in pergunta.lower():
                    try:
                        corr = df.corr(method="spearman")["Class"].sort_values(ascending=False)
                        resposta = f"Correlação com Class (fraude):\n{corr}"
                        st.info("🔎 Fallback aplicado: mostrando correlação com Class.")
                        st.write(resposta)
                    except Exception as e2:
                        resposta = f"Erro no fallback: {e2}"
                else:
                    resposta = "Não foi possível processar essa pergunta automaticamente."

            # salvar histórico
            st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})

        st.subheader("📝 Histórico")
        for item in st.session_state.historico:
            st.write(f"❓ {item['pergunta']}")
            st.write(f"➡️ {item['resposta']}")

        st.subheader("💡 Sugestões de perguntas")
        st.markdown("""
        ### 🔹 Descrição dos Dados
        - Quais são os tipos de dados (numéricos, categóricos)?  
        - Qual o valor mínimo e máximo da coluna Amount?  
        - Qual o valor mínimo e máximo da coluna Time?  
        - Qual a média e a mediana de Amount?  
        - Qual o desvio padrão e a variância da coluna Amount?  

        ### 🔹 Padrões e Tendências
        - Existem padrões temporais no dataset (Time vs Amount)?  
        - Quais os valores mais frequentes na coluna Amount?  
        - Qual a proporção de transações de fraude vs não fraude (Class)?  
        - Existem agrupamentos (clusters) nas variáveis V1–V28?  

        ### 🔹 Outliers
        - Existem outliers na coluna Amount?  
        - Como esses outliers afetam a média e a mediana?  
        - Seria melhor remover ou transformar os outliers de Amount?  

        ### 🔹 Relações entre Variáveis
        - Existe correlação entre as variáveis numéricas?  
        - Quais variáveis parecem ter maior influência sobre Class (fraude)?  
        - Como Amount se distribui entre Class = 0 e Class = 1?  
        """)

    # --- Estatísticas ---
    with aba1:
        st.subheader("📈 Estatísticas Descritivas")
        st.write(df.describe().T)
        st.write("Mediana por coluna numérica:", df.median(numeric_only=True))


    # --- Visualizações ---
    with aba2:
        st.subheader("📊 Visualizações")
        if numeric_cols:
            col_escolhida = st.selectbox("Escolha uma coluna numérica:", numeric_cols)
            st.plotly_chart(px.histogram(df, x=col_escolhida, nbins=30),
                            use_container_width=True, key="histograma")
            st.plotly_chart(px.box(df, y=col_escolhida),
                            use_container_width=True, key="boxplot")

        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlação"),
                            use_container_width=True, key="correlacao")

    # --- Frequência ---
    with aba3:
        st.subheader("🔍 Valores mais frequentes")
        col_freq = st.selectbox("Escolha uma coluna:", df.columns)
        st.write(df[col_freq].value_counts().head(10))
        st.plotly_chart(px.bar(df[col_freq].value_counts().head(10)),
                        use_container_width=True, key="frequencia")

    # --- Tendências ---
    with aba4:
        st.subheader("⏳ Tendências temporais")
        if "Time" in df.columns:
            df_temp = df.sort_values("Time")
            st.plotly_chart(px.line(df_temp, x="Time", y="Amount"),
                            use_container_width=True, key="tendencia")
        else:
            st.info("Nenhuma coluna temporal encontrada (ex: 'Time').")

    # --- Outliers ---
    with aba5:
        st.subheader("⚠️ Outliers")
        if numeric_cols:
            col_outlier = st.selectbox("Coluna para outliers:", numeric_cols)
            Q1, Q3 = df[col_outlier].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col_outlier] < Q1 - 1.5*IQR) | (df[col_outlier] > Q3 + 1.5*IQR)]
            st.write(f"{len(outliers)} outliers detectados em {col_outlier}")
            st.plotly_chart(px.box(df, y=col_outlier),
                            use_container_width=True, key="outliers")

    # --- Conclusões ---
    with aba7:
        st.subheader("📌 Conclusão")
        if st.session_state.historico:
            st.write("Com base nas perguntas feitas, podemos concluir:")
            for item in st.session_state.historico:
                st.write("- " + item["resposta"])
        else:
            st.write("Ainda não há conclusões.")
