# Imports
import pandas as pd
import streamlit as st
from io import BytesIO
import chardet
from pycaret.classification import load_model, predict_model, plot_model, setup
import matplotlib.pyplot as plt

# Configuração inicial da página
st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

def detect_encoding(file):
    """Detecta o encoding do arquivo"""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file.seek(0)  # Volta ao início do arquivo após a leitura
    return result['encoding']

def to_excel(df):
    """Função para converter o dataframe para o formato Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

@st.cache_data
def load_data(file):
    """Função para carregar dados com cache"""
    try:
        if file.type == 'text/csv':
            # Detecta o encoding do arquivo
            encoding = detect_encoding(file)
            st.write(f"Encoding detectado: {encoding}")
            df = pd.read_csv(file, encoding=encoding)
        elif file.type == 'application/octet-stream':
            df = pd.read_feather(file)
        else:
            raise ValueError("Tipo de arquivo não suportado")
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo. Detalhes: {e}")
        return None

def main():
    st.title("Bem-vindo ao PyCaret")

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Carregar CSV ou FTR", type=['csv', 'ftr'])

    if data_file_1 is not None:
        # Carregar dados
        df_credit = load_data(data_file_1)

        if df_credit is not None:
            st.write("### Dados Carregados")
            st.write("Primeiras linhas do arquivo:")
            st.write(df_credit.head())  # Exibe as primeiras 5 linhas do DataFrame
            st.write("Colunas disponíveis:", df_credit.columns.tolist())  # Lista as colunas

            # Verificar se a coluna alvo existe
            if 'mau' not in df_credit.columns:
                st.error("A coluna alvo 'mau' não foi encontrada no arquivo.")
                return

            # Usar amostra para desempenho (opcional)
            df_credit = df_credit.sample(min(50000, len(df_credit)))

            try:
                # Configurar o ambiente do PyCaret
                st.write("### Configurando o ambiente do PyCaret...")
                setup(data=df_credit, target='mau', session_id=123, verbose=False)

                # Carregar o modelo pré-treinado
                st.write("### Carregando o modelo LGBM 02 2025...")
                model = load_model('LGBM 02 2025')  # Substitua pelo caminho correto do seu modelo
                st.write("Modelo carregado com sucesso!")

                # Fazer previsões com o modelo carregado
                st.write("### Fazendo previsões...")
                predict = predict_model(model, data=df_credit)

                # Exibir o DataFrame com as previsões
                st.write("### Previsões")
                st.dataframe(predict)

                # Gerar gráficos do modelo PyCaret
                st.write("### Gráficos do Modelo")

                # Gráfico de importância das features
                st.write("#### Importância das Features")
                fig_feature = plot_model(model, plot='feature', display_format='streamlit')
                st.pyplot(fig_feature)  # Exibe o gráfico com o st.pyplot

                # Matriz de confusão
                st.write("#### Matriz de Confusão")
                fig_cm = plot_model(model, plot='confusion_matrix', display_format='streamlit')
                st.pyplot(fig_cm)  # Exibe o gráfico com o st.pyplot

                # Curva de precisão-recall
                st.write("#### Curva de Precisão-Recall")
                fig_pr = plot_model(model, plot='pr', display_format='streamlit')
                st.pyplot(fig_pr)  # Exibe o gráfico com o st.pyplot

                # Curva ROC
                st.write("#### Curva ROC")
                fig_auc = plot_model(model, plot='auc', display_format='streamlit')
                st.pyplot(fig_auc)  # Exibe o gráfico com o st.pyplot

                # Pipeline do modelo
                st.write("#### Pipeline do Modelo")
                fig_pipeline = plot_model(model, plot='pipeline', display_format='streamlit')
                st.pyplot(fig_pipeline)  # Exibe o gráfico com o st.pyplot

                # Converter os dados previstos para Excel e permitir download
                df_xlsx = to_excel(predict)
                st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

            except Exception as e:
                st.error(f"Erro ao processar o modelo. Detalhes: {e}")

if __name__ == '__main__':
    main()
