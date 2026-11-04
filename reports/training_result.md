# Resultados dos treinamentos de classificação de alfabetização

## Objetivo e método de avaliação

O objetivo é prever se um aluno será alfabetizado (`alfabetizado = 1`) ou não alfabetizado (`alfabetizado = 0`). Os três experimentos usaram validação temporal: os dados de 2023 e 2024 foram usados para treinamento, e 2025 — 1.965.985 alunos — foi reservado integralmente para teste. Assim, o modelo é avaliado em um período posterior ao treinamento, sem mistura temporal.

A decisão dá prioridade à identificação da classe `não alfabetizado`: recall, precisão e F1 dessa classe. A acurácia isolada não é suficiente, pois cerca de 66% dos alunos do conjunto de teste pertencem à classe alfabetizado; por isso, um classificador ingênuo que prevê apenas essa classe obtém 66,2% de acurácia e recall zero para não alfabetizados.

## Experimentos realizados

| Treino | Implementação | Alterações principais | Modelo selecionado |
|---|---|---|---|
| 1 | `train_model.py` (Pandas/scikit-learn) | Base completa; pré-processamento local; comparação entre regressão logística, floresta aleatória e HistGradientBoosting. | HistGradientBoosting |
| 2 | `train_model_spark.py` (Spark ML) | Treinamento distribuído, ponderação de classes e comparação entre regressão logística, Random Forest e GBT. | Regressão logística |
| 3 | `train_model_spark_reduced.py` (Spark ML) | Exclusão de variáveis selecionadas, one-hot encoding de `education_network_type` e `log1p` nas variáveis assimétricas remanescentes. | Regressão logística |

No treino 3 foram removidas as variáveis `households`, `total_teachers`, `total_enrollments`, `total_classes`, `fundeb_amount`, `school_count`, `hdi`, `hdi_income`, `income_per_capita_mean` e `pct_white`. A transformação logarítmica foi aplicada a `area`, `population_total`, `pct_rural_schools`, `pct_full_time_enrollments` e `pct_enrollments_zero_class_schools` — as demais variáveis originalmente propostas para `log1p` foram removidas antes, por prioridade de exclusão.

## Comparação dos modelos selecionados

| Treino | Modelo | Acurácia | Precisão não alfabetizado | Recall não alfabetizado | F1 não alfabetizado | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | HistGradientBoosting | 58,47% | 41,69% | 57,85% | 48,46% | 61,99% | 75,97% |
| 2 | Regressão logística | 60,35% | 43,25% | 56,04% | **48,82%** | 63,33% | **77,52%** |
| 3 | Regressão logística | **62,97%** | **45,04%** | 44,27% | 44,65% | **63,49%** | 77,48% |

O treino 3 apresentou a maior acurácia, mas isso ocorreu com redução relevante do recall da classe prioritária: ele encontrou 44,27% dos alunos não alfabetizados, contra 56,04% no treino 2. Portanto, o ganho de acurácia não compensa a perda de cobertura para o caso de uso.

## Comparação dos candidatos em cada treino

As métricas abaixo são as registradas na etapa de comparação de candidatos. No Spark, os candidatos são comparados em uma amostra estratificada do treino e o vencedor é então reentreinado com toda a base de 2023–2024.

| Treino | Candidato | Acurácia | Precisão não alfabetizado | Recall não alfabetizado | F1 não alfabetizado | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Dummy | 66,26% | 0,00% | 0,00% | 0,00% | 50,00% | 66,26% |
| 1 | Regressão logística | 61,67% | 42,71% | 39,78% | 41,19% | 58,81% | 70,55% |
| 1 | Random Forest | 60,32% | 41,25% | 41,48% | 41,36% | 60,71% | 75,35% |
| 1 | HistGradientBoosting | 58,47% | 41,69% | 57,85% | 48,46% | 61,99% | 75,97% |
| 2 | Dummy | 66,17% | 0,00% | 0,00% | 0,00% | 50,00% | 66,17% |
| 2 | Regressão logística | 58,41% | 42,36% | **63,57%** | **50,84%** | 63,03% | **77,07%** |
| 2 | Random Forest | 60,29% | 41,17% | 40,50% | 40,83% | 60,46% | 74,87% |
| 2 | GBT | 59,24% | 41,84% | 52,55% | 46,59% | 60,92% | 75,11% |
| 3 | Dummy | 66,17% | 0,00% | 0,00% | 0,00% | 50,00% | 66,17% |
| 3 | Regressão logística | 60,89% | 43,53% | 52,47% | 47,58% | 63,13% | 77,05% |
| 3 | Random Forest | 60,33% | 41,82% | 44,07% | 42,92% | 60,55% | 75,00% |
| 3 | GBT | 60,56% | 40,75% | 36,48% | 38,49% | 58,39% | 73,76% |

## Decisão: selecionar o modelo de regressão logística do treino 2

O modelo escolhido é a regressão logística treinada pelo `src/modeling/train_model_spark.py`. A decisão é pelo **modelo**, e não pelo experimento em si.

Ele foi selecionado porque equilibra melhor a capacidade de priorizar alunos não alfabetizados com capacidade discriminativa geral:

- obteve o maior F1 final para não alfabetizados (48,82%);
- preservou recall de 56,04% para a classe prioritária, superior ao do treino 3;
- obteve PR-AUC de 77,52% e ROC-AUC de 63,33%, superiores aos do treino 1 e praticamente equivalentes ao máximo observado;
- é mais simples, reproduzível e rápido de treinar no Glue do que os modelos baseados em árvores;
- fornece coeficientes interpretáveis, úteis para a apresentação e para auditoria do modelo.

Os parâmetros usados no modelo escolhido são `LogisticRegression(maxIter=100, regParam=0.0, weightCol="class_weight")`, com limiar de classificação padrão de 0,5. A ponderação `class_weight` equilibra as classes no treinamento sem duplicar registros. As variáveis categóricas são indexadas e codificadas; no artefato selecionado, o código do município e a sigla do estado foram tratados como informação categórica.

## Matriz de confusão do modelo escolhido

| Real \ Previsto | Não alfabetizado | Alfabetizado |
|---|---:|---:|
| Não alfabetizado | 371.762 | 291.634 |
| Alfabetizado | 487.830 | 814.759 |

O modelo identificou corretamente 371.762 dos 663.396 alunos não alfabetizados do teste. Há 291.634 alunos não alfabetizados que não foram sinalizados e 487.830 alertas para alunos que, no dado observado, estavam alfabetizados. Esses alertas devem ser tratados como priorização para acompanhamento, não como diagnóstico definitivo do aluno.

## Variáveis mais influentes

No modelo selecionado, o sinal categórico mais forte é o **código do município** (`city_code`). Isso é coerente com diferenças territoriais relevantes, mas exige monitoramento: uma variável geográfica de alta cardinalidade pode capturar padrões locais muito específicos e perder desempenho em municípios novos ou sob mudanças estruturais.

Entre os coeficientes numéricos de maior magnitude observados no artefato, destacam-se:

| Variável técnica | Tradução | Direção do coeficiente |
|---|---|---:|
| `hdi_life_expectancy` | IDHM — longevidade | -12,07 |
| `pct_yellow` | Percentual da população amarela | -12,01 |
| `hdi_income` | IDHM — renda | +7,23 |
| `gini_index` | Índice de Gini | -6,10 |

O sinal do coeficiente descreve associação dentro do modelo e não causalidade. Além disso, a magnitude de coeficientes de variáveis numéricas em escalas diferentes não é, isoladamente, uma medida definitiva de importância. Para decisões de política pública, essas relações devem ser complementadas por análise estatística e conhecimento de domínio.

## Próximos passos recomendados

- Avaliar ajuste de limiar de probabilidade conforme a capacidade operacional de acompanhamento dos alunos sinalizados.
- Monitorar as métricas por ano, UF, rede de ensino e município para verificar estabilidade e possíveis vieses territoriais.
- Reexecutar a validação temporal quando houver dados de um novo ano, mantendo o ano mais recente exclusivamente para teste.
- Complementar o acompanhamento com calibração de probabilidades e métricas de equidade antes de usar o resultado em qualquer processo decisório.

## Fontes dos resultados

Os valores foram extraídos de `reports/train-result-01.json`, `reports/train-result-02.json` e `reports/train-result-03.json`, gerados pelos três treinamentos.
