# Credit Scoring para Cartão de Crédito

## Descrição do Projeto

Este projeto tem como objetivo construir um modelo de **credit scoring** para cartões de crédito, utilizando um conjunto de dados estruturado em **15 safras** e considerando **12 meses de performance**. O foco é avaliar a capacidade de clientes cumprirem seus compromissos financeiros com base em variáveis históricas.

## Estrutura do Projeto

O projeto segue as seguintes etapas:

1. **Carregamento dos Dados**

   - Os dados estão armazenados no arquivo `credit_scoring.ftr`.
   - A variável `data_ref` é usada como indicadora de safra e **não deve ser utilizada** na modelagem.
   - A variável `index` identifica cada cliente e também **não deve ser usada** como covariável.

2. **Pré-processamento e Amostragem**

   - Conversão da coluna `data_ref` para o formato datetime e definição do índice do DataFrame.
   - Divisão dos dados: os três últimos meses são separados como **safras de validação out of time (OOT)**.

3. **Análise Exploratória de Dados (EDA)**

   - Análise estatística das variáveis.
   - Visualização dos padrões e distribuições das variáveis-chave.

4. **Modelagem Preditiva**

   - Treinamento de um modelo de credit scoring com diferentes abordagens estatísticas e de machine learning.
   - Avaliação do desempenho do modelo utilizando métricas apropriadas.

## Tecnologias Utilizadas

- **Linguagem**: Python
- **Bibliotecas**: pandas, numpy, matplotlib

## Como Utilizar

1. Clone o repositório ou baixe os arquivos.
2. Instale as dependências necessárias:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. Execute o notebook `Exercicio_M38.ipynb` para reproduzir a análise e modelagem.

## Contato

Caso tenha dúvidas ou sugestões, entre em contato pelo GitHub ou LinkedIn.

---

**Autor:** Isabelle Fernanda

