# Tech Challenge — Fase 3

## Predição de alfabetização e apoio à decisão educacional

Este projeto transforma dados públicos educacionais, demográficos e socioeconômicos em uma base analítica para estimar o risco de um aluno não estar alfabetizado. Ele foi desenvolvido como continuação da engenharia de dados da Fase 2 e entrega uma pipeline Bronze, Silver e Gold, além de experimentos de Machine Learning.

Informações detalhadas sobre os resultados estão em [reports/](reports/), incluindo o [relatório dos treinamentos](reports/training_result.md) e os [insights das agregações de risco](reports/aggregation_insights.md).

## Contexto do problema

A alfabetização nos anos iniciais é um fator decisivo para a continuidade da aprendizagem. Redes de ensino precisam identificar onde há maior concentração de risco para organizar reforço pedagógico, formação docente e alocação de recursos. Como as condições educacionais e sociais variam entre municípios, estados e redes de ensino, uma análise baseada apenas em médias nacionais pode ocultar situações prioritárias.

O projeto não busca rotular alunos nem substituir a avaliação pedagógica. As previsões devem apoiar a priorização de acompanhamento e sempre ser interpretadas por profissionais da educação no contexto local.

## Objetivo analítico

Prever a variável alvo `alfabetizado`:

- `1`: aluno alfabetizado;
- `0`: aluno não alfabetizado.

O foco da avaliação é identificar corretamente alunos não alfabetizados. Por isso, a escolha do modelo prioriza o recall, a precisão e o F1 dessa classe, em vez de usar apenas acurácia.

## Descrição da base utilizada

A tabela final para Machine Learning é `ml/ml_dataset_alunos`, criada na camada Gold. Cada registro representa um aluno avaliado e combina informações conhecidas antes da avaliação.

| Fonte | Informações incorporadas |
|---|---|
| Avaliação de alfabetização do INEP | Situação de alfabetização, série, rede de ensino e município. |
| Censo Demográfico 2022 (IBGE) | População, composição racial, domicílios, área, taxa de alfabetização, idade mediana, razão de sexo e índice de envelhecimento. |
| Censo Escolar | Escolas, matrículas, docentes, turmas, infraestrutura e indicadores de atendimento por município e ano. |
| FUNDEB | Valor anual de recursos por município. |
| IDHM | Indicadores estaduais de desenvolvimento humano, renda, longevidade, educação e desigualdade. |

Os experimentos usaram 2023 e 2024 para treinamento e reservaram 2025 para teste temporal, com 1.965.985 registros. Identificadores individuais, proficiência e variáveis calculadas depois da avaliação foram removidos para evitar vazamento de dados. O código do município foi mantido deliberadamente para capturar o contexto territorial conhecido no uso pretendido.

## Pipeline de dados

1. **Bronze:** recebe e organiza arquivos brutos no armazenamento de objetos.
2. **Silver:** normaliza esquemas, tipos e nomes de colunas; os dados são gravados em Parquet.
3. **Gold — consolidação:** une as fontes por ano, município, estado, série e rede de ensino em `alfabetizacao/base_alunos`.
4. **Gold — ML:** remove atributos com vazamento ou identificadores inadequados e produz `ml/ml_dataset_alunos`.
5. **Modelagem:** lê a Gold, realiza a separação temporal, compara modelos e grava métricas e artefatos.

Os jobs estão em `src/pipeline/` e os notebooks exploratórios estão em `notebooks/`. Os dados locais de apoio ficam em `data/` e não devem ser versionados.

## Etapas de modelagem

1. Seleção do alvo e remoção de atributos indisponíveis antes da avaliação.
2. Separação temporal: treino em 2023–2024 e teste em 2025.
3. Tratamento de valores ausentes e codificação de variáveis categóricas.
4. Ponderação das classes no Spark para que a classe não alfabetizada tenha relevância durante o treinamento.
5. Comparação entre classificador ingênuo, regressão logística, Random Forest e Gradient-Boosted Trees (GBT).
6. Re-treinamento do vencedor com a base completa de treino e avaliação no conjunto temporal de teste.

Foram executados três experimentos: uma linha de base com Pandas/scikit-learn, uma versão distribuída com Spark ML e uma versão Spark com conjunto reduzido de atributos e transformações adicionais.

## Escolha do algoritmo

O modelo selecionado foi a **regressão logística do segundo experimento**, treinada por `src/modeling/train_model_spark.py`.

Parâmetros principais:

- `maxIter=100`;
- `regParam=0.0`;
- `weightCol="class_weight"`;
- limiar de decisão de 0,5.

Ela foi escolhida por apresentar o melhor equilíbrio entre capacidade de encontrar alunos não alfabetizados, qualidade de ordenação de risco, velocidade de treinamento no Glue e interpretabilidade. Modelos de árvores foram avaliados, mas não superaram esse equilíbrio para o objetivo definido.

## Métricas de avaliação

As métricas são calculadas no teste de 2025.

- **Acurácia:** proporção total de classificações corretas. É contextualizada pela distribuição das classes e não é usada isoladamente.
- **Precisão para não alfabetizado:** entre os alunos sinalizados, proporção realmente não alfabetizada.
- **Recall para não alfabetizado:** entre os alunos não alfabetizados, proporção que o modelo identificou.
- **F1 para não alfabetizado:** equilíbrio entre precisão e recall da classe prioritária.
- **ROC-AUC:** capacidade geral de separar as duas classes em diferentes limiares.
- **PR-AUC:** qualidade da ordenação de risco, especialmente útil diante de desbalanceamento de classes.

Um classificador ingênuo que sempre prevê “alfabetizado” alcança aproximadamente 66,2% de acurácia, mas tem recall zero para não alfabetizados. Isso demonstra por que a acurácia, sozinha, seria enganosa.

## Interpretação dos resultados

| Treino | Modelo selecionado | Acurácia | Precisão não alfabetizado | Recall não alfabetizado | F1 não alfabetizado | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | HistGradientBoosting | 58,47% | 41,69% | 57,85% | 48,46% | 61,99% | 75,97% |
| 2 | Regressão logística | 60,35% | 43,25% | 56,04% | **48,82%** | 63,33% | **77,52%** |
| 3 | Regressão logística | **62,97%** | **45,04%** | 44,27% | 44,65% | **63,49%** | 77,48% |

Embora o treino 3 tenha a maior acurácia, ele reduz o recall da classe prioritária de 56,04% para 44,27%. O modelo escolhido identificou corretamente 371.762 dos 663.396 alunos não alfabetizados no conjunto de teste. Assim, o treino 2 oferece maior cobertura do público que deve ser priorizado, sem perda relevante de capacidade discriminativa.

## Insights encontrados

- O contexto territorial é relevante: `city_code` foi o sinal categórico mais forte no modelo selecionado.
- Indicadores de desenvolvimento e desigualdade também apresentaram associação com a previsão, como IDHM — longevidade, IDHM — renda e índice de Gini.
- A maior acurácia não corresponde necessariamente ao melhor modelo para a política pública: o custo de não sinalizar alunos não alfabetizados torna o recall dessa classe essencial.
- O modelo produz um sinal de risco para priorização; ele não estabelece causalidade e não deve ser usado como diagnóstico individual.

## Limitações do projeto

- O desempenho depende da qualidade, cobertura e atualização das fontes públicas.
- O uso de município pode capturar padrões territoriais específicos e reduzir a generalização para localidades novas ou em rápida mudança.
- O modelo não inclui variáveis pedagógicas qualitativas, contexto familiar detalhado ou intervenções já realizadas pela escola.
- Há falsos negativos e falsos positivos; uma previsão não substitui avaliação pedagógica.
- A pipeline é reexecutável, mas ainda precisa de versionamento formal dos inputs S3, manifesto de execução e ambientes totalmente fixados para reprodução histórica byte a byte.
- As associações encontradas não comprovam relações de causa e efeito.

## Aplicação prática para políticas públicas

O resultado pode apoiar secretarias e redes de ensino a:

- criar listas priorizadas de acompanhamento pedagógico, com revisão humana;
- direcionar reforço, busca ativa e apoio a escolas ou municípios com maior concentração de risco;
- planejar formação docente e distribuição de recursos com base em indicadores territoriais;
- acompanhar a evolução dos indicadores por município, rede e ano;
- avaliar se políticas implementadas reduzem o risco estimado e melhoram a alfabetização observada.

Qualquer uso deve respeitar a LGPD, princípios de não discriminação, transparência e governança. Resultados individuais não devem gerar punições, exclusões ou decisões automatizadas.

## Possíveis evoluções futuras

- Calibrar probabilidades e ajustar o limiar de classificação conforme a capacidade de atendimento da rede.
- Avaliar estabilidade, viés e desempenho por UF, município, rede de ensino e outros grupos relevantes.
- Validar o modelo em novos anos e re-treiná-lo periodicamente.
- Adicionar testes automatizados de qualidade de dados, contratos de esquema e monitoramento de drift.
- Versionar dados e execuções com manifesto contendo commit Git, parâmetros, versão dos arquivos de entrada e ambiente Glue.
- Explorar modelos hierárquicos ou validação por grupos geográficos para medir melhor a generalização territorial.

## Como executar

O [guia de execução](execution.md) detalha os requisitos de software, as
limitações e os comandos para rodar localmente, além da ordem e configuração dos
jobs para reproduzir a pipeline e o treinamento no AWS Glue.

## Estrutura do projeto

```text
.
├── data/           # dados externos e amostras locais
├── notebooks/      # análise exploratória e apresentação
├── reports/        # resultados e documentação
├── src/
│   ├── modeling/   # treinamento e avaliação de modelos
│   └── pipeline/   # jobs Bronze, Silver e Gold
├── local/          # executor local compatível com Glue
├── requirements.txt
└── README.md
```
