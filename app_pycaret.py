# Imports
import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import load_model, predict_model, setup, plot_model

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title='PyCaret', layout="wide", initial_sidebar_state='expanded')

    # Título principal da aplicação
    st.write("## Escorando o modelo gerado no pycaret")
    st.markdown("---")
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    # Verifica se há conteúdo carregado na aplicação
    if data_file_1 is not None:
        # Carregar o arquivo e processar
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)

        # Carregar o modelo e fazer a previsão
        model_saved = load_model('LR Model Aula 5 062022')
        predict = predict_model(model_saved, data=df_credit)

        # Converter os dados previstos para Excel e permitir download
        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

# Exemplo de como configurar e gerar gráficos no PyCaret (fora da função main)
if __name__ == '__main__':
    # Carregar a base de dados (apenas para demonstração, pode ser alterado conforme o arquivo carregado)
    data = pd.read_feather('caminho/para/seu/arquivo.feather')

    # Reduzindo a base para teste (opcional)
    data_sample = data.sample(frac=0.5, random_state=42)  # Reduz a amostra em 50%

    # Configuração do PyCaret
    setup(data=data_sample, target='mau', session_id=42, normalize=True, remove_outliers=False, use_gpu=True)

    # Gerar gráficos
    model_pycaret = load_model('LR Model Aula 5 062022')  # Carregue seu modelo treinado
    plot_model(model_pycaret, plot='feature')
    plot_model(model_pycaret, plot='confusion_matrix')
    plot_model(model_pycaret, plot='pr')
    plot_model(model_pycaret, plot='auc')

    # Executar a função main
    main()
