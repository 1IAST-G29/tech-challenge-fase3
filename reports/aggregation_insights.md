# Insights das agregações de risco

## Objetivo do relatório

Este relatório analisa os arquivos de agregação gerados ao final do treinamento. Eles resumem as previsões do conjunto de teste de 2025 por município, UF e tipo de rede de ensino. Cada grupo apresenta:

- `records`: número de alunos no grupo;
- `actual_not_literate_rate`: proporção observada de alunos não alfabetizados;
- `predicted_not_literate_rate`: proporção classificada pelo modelo como não alfabetizada, com limiar de 0,5;
- `average_not_literate_risk`: risco médio previsto de não alfabetização.

As taxas observadas existem apenas porque estes resultados foram calculados no conjunto de teste. Em uso futuro, somente o risco previsto e a taxa prevista estarão disponíveis.

## Inventário dos arquivos

| Arquivo | Situação | Modelo associado |
|---|---|---|
| `reports/aggregations-01.json` | Resultado distinto | HistGradientBoosting do treino 1. |
| `reports/aggregations-02.json` | Resultado distinto | Regressão logística do treino 2, modelo vencedor. |
| `reports/aggregations-03.json` | Resultado distinto | Regressão logística do treino 3, com variáveis reduzidas e transformações. |

Portanto, há **três** resultados de agregação disponíveis, um para cada experimento.

Todos possuem 1.965.985 registros, 5.555 municípios, 27 UFs e dois tipos de rede de ensino.

## Visão geral: modelo vencedor

O arquivo `reports/aggregations-02.json`, associado ao modelo vencedor, apresenta:

| Indicador global | Valor |
|---|---:|
| Taxa real de não alfabetização | 33,74% |
| Taxa prevista como não alfabetizado | 43,72% |
| Risco médio previsto | 46,12% |

O modelo sinaliza mais alunos como não alfabetizados do que a taxa observada, com diferença de **9,98 pontos percentuais**. Esse comportamento é coerente com a escolha de priorizar recall da classe não alfabetizada: o modelo aceita mais alertas para deixar menos alunos em risco sem sinalização. Antes de uso operacional, o limiar de 0,5 deve ser calibrado conforme a capacidade da rede de atender os casos priorizados.

## Insights por UF

As UFs com maior risco médio previsto pelo modelo vencedor são:

| UF | Alunos | Risco médio previsto | Taxa real | Taxa prevista |
|---|---:|---:|---:|---:|
| AP | 11.007 | 67,32% | 39,70% | 96,06% |
| SE | 19.041 | 65,75% | 50,04% | 93,47% |
| BA | 120.199 | 63,81% | 44,22% | 95,20% |
| RN | 26.161 | 63,56% | 50,80% | 95,23% |
| DF | 23.797 | 60,62% | 34,87% | 100,00% |

Rio Grande do Norte, Sergipe e Bahia também estão entre as maiores taxas observadas de não alfabetização, o que reforça a utilidade desses grupos para priorização territorial. Entretanto, Amapá e Distrito Federal mostram diferença expressiva entre a taxa prevista e a observada. Isso indica possível superestimação de risco nesses contextos e reforça a necessidade de monitoramento de calibração por UF.

## Insights por tipo de rede

| Tipo de rede | Alunos | Risco médio previsto | Taxa real | Taxa prevista |
|---|---:|---:|---:|---:|
| 3 | 1.722.787 | 46,26% | 33,52% | 43,79% |
| 2 | 243.198 | 45,13% | 35,33% | 43,22% |

Os dois grupos recebem risco médio semelhante, mas o tipo de rede 2 tem uma taxa observada ligeiramente maior. Essa diferença merece acompanhamento de desempenho por grupo antes de qualquer decisão operacional, inclusive com métricas de precisão e recall segmentadas por rede.

## Municípios com maior risco estimado

Para reduzir instabilidade causada por amostras muito pequenas, a tabela abaixo considera apenas municípios com pelo menos 500 alunos no teste.

| Código do município | Alunos | Risco médio previsto | Taxa real | Taxa prevista |
|---|---:|---:|---:|---:|
| 2907202 | 651 | 83,22% | 76,04% | 100,00% |
| 2902104 | 588 | 80,94% | 57,14% | 100,00% |
| 1502806 | 709 | 79,66% | 64,74% | 100,00% |
| 2211100 | 558 | 78,23% | 21,68% | 100,00% |
| 1503101 | 610 | 78,22% | 49,67% | 100,00% |

Esses códigos devem ser usados como ponto de partida para investigação, não como diagnóstico. O município 2211100, por exemplo, apresenta risco médio alto, mas taxa real consideravelmente menor, sinalizando a necessidade de analisar qualidade dos dados, contexto local e calibração antes de priorizar recursos.

## Comparação entre os três resultados

| Indicador | HistGradientBoosting, treino 1 | Regressão logística, treino 2 (vencedor) | Regressão logística reduzida, treino 3 |
|---|---:|---:|---:|
| Taxa real de não alfabetização | 33,74% | 33,74% | 33,74% |
| Taxa prevista como não alfabetizado | 46,82% | 43,72% | 33,16% |
| Risco médio previsto | 49,26% | 46,12% | 43,21% |

O HistGradientBoosting gerou mais alertas que o modelo vencedor, excedendo a taxa observada em 13,08 pontos percentuais. A regressão logística do treino 2 também excedeu a taxa observada em 9,98 pontos percentuais, em linha com a prioridade de recall para alunos não alfabetizados. Já o treino 3 previu 0,58 ponto percentual menos casos que a taxa observada, comportamento mais conservador que ajuda a explicar seu recall menor (44,27%, ante 56,04% no modelo vencedor).

| Comparação de risco municipal | Correlação ponderada por alunos | Diferença absoluta média por município |
|---|---:|---:|
| Treino 1 × Treino 2 | 0,78 | 11,46 p.p. |
| Treino 1 × Treino 3 | 0,76 | 12,31 p.p. |
| Treino 2 × Treino 3 | **0,96** | **3,82 p.p.** |

Os dois modelos de regressão logística apresentam forte concordância nos padrões territoriais. Ambos colocam, por exemplo, Sergipe, Rio Grande do Norte e Bahia entre as UFs de maior risco; e os municípios `2907202` e `2902104` no topo do risco entre os grupos com ao menos 500 alunos. O treino 3, porém, reduziu sistematicamente o risco estimado e os alertas, sobretudo na rede de tipo 2: classificou 10,04% desse grupo como não alfabetizado, frente a 43,22% no modelo vencedor, embora a taxa observada seja 35,33%.

Essa diferença é coerente com a perda de recall do treino 3 e reforça a escolha do modelo do treino 2: ele preserva maior cobertura da população prioritária sem o excesso de alertas do HistGradientBoosting. Rankings municipais não devem depender de um único modelo sem acompanhamento contínuo, calibração e revisão humana.

## Recomendações de uso

1. Ordenar municípios pelo risco médio previsto, sempre exibindo também o número de alunos.
2. Definir um tamanho mínimo de amostra e aplicar suavização estatística antes de comparar municípios pequenos.
3. Ajustar o limiar de classificação de acordo com a capacidade real de atendimento pedagógico.
4. Monitorar erro, precisão, recall e calibração por UF, tipo de rede e município.
5. Usar os rankings para investigação e alocação de apoio, com revisão humana; não para rotular alunos, escolas ou profissionais.
6. Manter as agregações dos três treinamentos a cada reexecução, para acompanhar a estabilidade dos rankings e da calibração territorial.
