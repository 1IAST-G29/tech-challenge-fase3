# Tech Challenge Fase 3 - Predição de Alfabetização no Brasil

## 1. Contexto do problema
A alfabetização na idade certa é um dos principais indicadores educacionais do país. Em 2025, ~59% dos alunos do 2º ano foram considerados alfabetizados (52% em 2024). Gestores precisam antecipar riscos e entender o que pesa nos resultados.

## 2. Objetivo analítico
Prever se um aluno será **alfabetizado (1)** ou **não alfabetizado (0)** (`IN_ALFABETIZADO`) usando variáveis educacionais, territoriais e socioeconômicas, e transformar isso em inteligência para política pública (risco municipal, perfis regionais, municípios abaixo da meta).

## 3. Descrição da base
Não existia uma camada Gold local; ela é construída por `src/preprocessing/build_gold.py` a partir dos dados brutos em `src/data/notebooks/` (Git LFS).

| Fonte | Uso | Granularidade final |
|---|---|---|
| Avaliação da Alfabetização INEP 2023-2025 (aluno, município, UF) | alvo + histórico | aluno; histórico por escola, município x rede e UF **do ano anterior** |
| Censo Escolar 2022-2024 | infraestrutura, razões aluno/turma e aluno/docente, perfil racial | município x rede, ano Y-1 |
| FUNDEB por município 2007-2026 | total anual e valor por matrícula | município, ano Y-1 |
| Atlas do Desenvolvimento Humano (xlsx) | **não integrado** (arquivos ainda são ponteiros LFS; ver Limitações) | - |

Tabela de modelagem: 4,34 mi de alunos (2024: 2,12 mi; 2025: 2,22 mi). 2023 serve só como "ano anterior" de 2024.

## 4. Etapas de modelagem
1. **Gold** (`src/preprocessing`): junções por chaves, *lags* de 1 ano, FUNDEB por nome normalizado + UF.
2. **EDA** (`src/visualization/eda.py`, `notebooks/01_eda.ipynb`): balanceamento, correlações, faltantes, prova de leakage.
3. **Pipeline sklearn** (`src/modeling/features.py`): engenharia de atributos (hiatos escola-município e município-UF, logs) -> imputação por mediana + indicador de ausência -> padronização (numéricas) -> One-Hot (categóricas) -> modelo. Tudo dentro do `Pipeline`, ajustado só no treino.
4. **Validação** (`src/modeling/train.py`): treino em 2024, `RandomizedSearchCV` com `GroupKFold` por escola, **teste fora do tempo em 2025** (nunca usado na escolha de hiperparâmetros).
5. **Interpretabilidade** (`src/evaluation/interpret.py`): Permutation Importance e SHAP.
6. **Aplicação** (`src/evaluation/municipal.py`): risco municipal, clusters (KMeans) e modelo de municípios abaixo da meta.

### Tratamento de data leakage
- `VL_PROFICIENCIA_LP` **define** o alvo (corte em 743 pontos, AUC = 1,0): excluída.
- Pesos, respostas/gabaritos, cadernos, IDs e ano: excluídos (`EXCLUDE` em `features.py`; testado em `tests/`).
- Agregados municipais/estaduais do **mesmo ano** são calculados com o alvo: substituídos por versões do **ano anterior**.
- Split temporal (2024 -> 2025) + `GroupKFold` por escola; pré-processamento ajustado apenas em treino.
- Distribuição por níveis municipal: existe só a partir de 2024 (100% nula no lag de 2024) -> descartada para evitar drift.

## 5. Escolha do algoritmo
- **Baselines**: prior (Dummy) e Regressão Logística (linear, interpretável).
- **Modelo final: HistGradientBoostingClassifier** - trata bem não linearidades e interações, é rápido em 2 milhões de linhas, aceita faltantes e tem regularização (L2, profundidade, `min_samples_leaf`, *early stopping*) contra overfitting.

## 6. Métricas de avaliação (teste 2025)
| Modelo | ROC-AUC | PR-AUC | Brier | F1 | Bal. Acc. |
|---|---|---|---|---|---|
| Dummy (prior) | 0,500 | 0,586 | 0,247 | 0,739 | 0,500 |
| Regressão Logística | 0,625 | 0,703 | 0,235 | 0,668 | 0,582 |
| **HistGradientBoosting** | **0,628** | **0,706** | **0,233** | 0,680 | 0,576 |

CV (GroupKFold por escola, 2024): ROC-AUC 0,655 ± 0,009. A queda para 0,628 no teste reflete a mudança de ano (taxa subiu de 52% para 59%). O F1 do Dummy é alto só por prever sempre "alfabetizado"; prefira ROC-AUC/PR-AUC/Brier. Detalhes: `reports/metrics.json`.

## 7. Interpretação dos resultados
Variáveis mais influentes (`reports/shap_importance.csv`, `reports/permutation_importance.csv`):
1. Proficiência média do município no ano anterior (`mun_lag_vl_media`) - de longe a mais importante.
2. Proficiência e taxa da UF no ano anterior, região Norte, nº de escolas da rede.
3. Taxa municipal anterior, biblioteca, internet, FUNDEB.

## 8. Insights encontrados
- **O passado é o melhor preditor**: o desempenho do município no ano anterior domina; infraestrutura e FUNDEB têm efeito pequeno após controlar o histórico (correlação, não causalidade).
- **Desigualdade regional**: Norte ~47%, Centro-Oeste ~62%.
- **Redes estadual e municipal** têm taxas parecidas (55,4% vs. 55,5%).
- **Municípios**: a predição agregada tem correlação 0,69 com a taxa real (`municipal_metrics.json`).
- **3 perfis de municípios** (`reports/perfil_clusters.csv`): alto desempenho (79% no ano anterior, 1.133 municípios), intermediário/urbano (67%, 2.449) e vulnerável/rural (46%, 1.630).
- **Municípios abaixo da meta**: modelo municipal com ROC-AUC 0,85 (treina em 2024, testa em 2025). Listas em `reports/top50_municipios_*.csv`.

## 9. Limitações
- **Sem variáveis socioeconômicas por aluno** (renda, escolaridade dos pais, raça): o teto de ROC-AUC é baixo (~0,63). O modelo capta contexto, não o aluno.
- `ID_ESCOLA` é anonimizado; o Censo só pôde ser juntado por município x rede.
- Atlas/IDHM não integrado (xlsx em Git LFS não baixados). Metas oficiais estão no BigQuery e **não** foram integradas: o limiar de 60% em `municipal.py` (`META_LIMIAR`) é uma **suposição** a ser substituída.
- Taxas calculadas sem os pesos amostrais do INEP; o primeiro ano com *lag* tem 20% de escolas sem histórico.
- Há deriva temporal (taxa base mudou) e as probabilidades não estão recalibradas.
- Correlação não implica causalidade; usar como triagem, não como sentença.

## 10. Aplicação prática para políticas públicas
- Priorizar apoio técnico a municípios do `top50_municipios_risco.csv` e `top50_municipios_abaixo_meta.csv`.
- Usar os 3 perfis para desenhar ações diferenciadas (ex.: foco rural e formação docente no perfil vulnerável).
- Monitoramento anual: atualizar os lags a cada nova avaliação e reexecutar `python -m src.run_all`.

## 11. Possíveis evoluções
Integrar Atlas/IDHM, PNAD, Cadastro Único e metas oficiais; obter chave escola INEP para junção direta com o Censo; modelo hierárquico (aluno dentro de escola/município); calibração por ano; análise causal (diferenças em diferenças) para FUNDEB e infraestrutura.

## Como reproduzir
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
git lfs pull                      # baixa os dados brutos
python -m src.run_all             # Gold -> EDA -> treino -> SHAP -> análise municipal
pytest -q tests                   # testes anti-leakage
```
Sementes fixas (`SEED=42` em `src/config.py`). Notebooks em `notebooks/`, figuras em `images/`, tabelas em `reports/`, roteiro do vídeo em `reports/roteiro_video_executivo.md`.

```
data/ (Gold, ignorada)  notebooks/  src/{preprocessing,modeling,evaluation,visualization}  reports/  images/  tests/
```

_Este documento foi gerado por Claude._
