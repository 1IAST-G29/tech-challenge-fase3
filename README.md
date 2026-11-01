# Tech Challenge — Fase 3

## Predição e inteligência analítica para alfabetização no Brasil

Este projeto usa os dados de alfabetização tratados na Fase 2 para desenvolver uma
solução de Machine Learning capaz de estimar o risco de um aluno não ser
alfabetizado. A finalidade é transformar dados públicos em informações que apoiem a
priorização de ações educacionais.

O plano detalhado de implementação está em [plan.md](plan.md).

## Objetivo

Construir e avaliar modelos supervisionados para prever a coluna `alfabetizado`
(0 ou 1), usando informações educacionais, territoriais e socioeconômicas que sejam
conhecidas antes da avaliação. O projeto também deverá explicar os principais fatores
associados às previsões e apresentar resultados agregados por município, UF e região.

## Princípios importantes

- `proficiencia` não pode ser usada como atributo: ela é a base da definição do alvo
  `alfabetizado` na Fase 2.
- Nenhuma informação calculada após a avaliação deve entrar no modelo.
- Treino, validação e teste devem ser separados de forma reproduzível, de preferência
  por tempo ou por grupos como município/escola.
- Os resultados apoiam decisões; não devem rotular ou decidir automaticamente sobre
  alunos, escolas ou profissionais.

## Estrutura prevista

~~~text
.
├── data/           # amostras locais e dados externos (não versionados)
├── notebooks/      # análise exploratória e apresentação de resultados
├── src/            # código reutilizável de dados, atributos, treino e avaliação
├── tests/          # testes unitários e de integração
├── reports/        # gráficos e documentação de resultados
├── plan.md         # plano de ação técnico
├── requirements.txt
└── README.md
~~~

## Como começar localmente

Requer Python 3.12.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
~~~

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

## Teste local dos jobs Glue com Docker

O diretório `local/` contém um executor que simula as classes mínimas do AWS Glue
para rodar os jobs com Spark localmente. O `docker-compose.yml` cria o container
Python/Java e um MinIO opcional para testes que precisem de S3.

```bash
docker compose build pipeline
mkdir -p local-data/bronze/fundeb local-data/silver
cp src/data/notebooks/fundeb/fundeb-por-municipio.csv local-data/bronze/fundeb/
docker compose run --rm pipeline python local/run_glue_job.py \
  src/data/pipeline/silver/silver_fundeb.py \
  --JOB_NAME local-fundeb \
  --BUCKET_BRONZE file:///workspace/local-data/bronze \
  --BUCKET_SILVER file:///workspace/local-data/silver
```

O resultado será gravado em `local-data/silver/fundeb/fundeb_anual/`, particionado
por `year`. Esse ambiente testa a transformação Spark e a escrita Parquet; a
execução real no AWS Glue ainda deve ser validada na infraestrutura AWS.

O job `src/data/pipeline/silver/silver_censo_2022.py` lê o Excel racial do Censo
2022 e gera `local-data/silver/censo_2022/censo_2022.csv`, reproduzindo o formato
exportado pelo notebook `censo_2022.ipynb`.

Na camada Gold, o fluxo está dividido em dois jobs: `gold_consolidacao_alunos.py`
consolida os datasets Silver a partir dos alunos, e `gold_ml_dataset.py` cria
`ml/ml_dataset_alunos` para treinamento, removendo proficiência e campos
disponíveis somente após a avaliação.

## Dados

A base de modelagem será uma nova tabela Gold, `ml_dataset_alunos`, construída a
partir das tabelas tratadas na Fase 2 e, quando necessário, de fontes públicas como
IBGE e Censo Escolar. Dados completos, identificadores pessoais, credenciais e
artefatos de modelos não devem ser enviados ao repositório.

## Treinamento local

Com a `ml_dataset_alunos` disponível localmente em Parquet, o modelo pode ser
treinado com:

```bash
uv run python src/modeling/train_model.py \
  --input local-data/gold/ml/ml_dataset_alunos \
  --output artifacts/model
```

O script compara `DummyClassifier`, Regressão Logística, Random Forest e
`HistGradientBoostingClassifier`, um modelo de boosting adequado para dados
tabulares demográficos. Ele separa o último ano para teste e escolhe o melhor pelo
recall da classe não alfabetizada. Os artefatos incluem a comparação entre os
quatro modelos.

## Próximas etapas

1. Inventariar o esquema e a cobertura temporal das tabelas Silver/Gold da Fase 2.
2. Construir a tabela Gold de atributos para ML sem vazamento de dados.
3. Realizar análise exploratória.
4. Treinar, validar e interpretar modelos.
5. Gerar ranking agregado de risco, documentação e apresentação executiva.
