# Guia de execução

Este guia descreve as tecnologias, pré-requisitos e caminhos para executar a pipeline e o treinamento, localmente e no AWS Glue.

## Visão geral

O projeto tem dois fluxos:

1. **Pipeline de dados:** transforma arquivos brutos nas camadas Bronze, Silver e Gold e produz **ml/ml_dataset_alunos**.
2. **Treinamento:** lê a Gold de Machine Learning, compara classificadores e grava o modelo e as métricas.

| Ambiente | Escopo suportado |
|---|---|
| Local com Python | Notebooks, validações leves e treinamento Pandas/scikit-learn, caso a Gold em Parquet já exista. |
| Local com Docker | Jobs Bronze, Silver e Gold com Spark local e treinamento Spark ML. |
| AWS Glue | Pipeline sobre S3 e treinamento distribuído com recursos gerenciados da AWS. |

## Tecnologias e requisitos

### Para executar localmente

- Git;
- Python 3.12;
- Java 21, requerido pelo Spark;
- Docker Engine e Docker Compose plugin;
- pelo menos 16 GB de RAM recomendados. A pipeline completa e o treinamento Spark podem exigir mais memória.

As dependências estão em **requirements.txt**. O container definido em **Dockerfile.local** instala Python, Java, PySpark e as dependências necessárias.

### Preparar o ambiente Python

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

No Windows, ative o ambiente com **.venv\Scripts\activate**.

## Execução local

### Dados necessários

Os arquivos de entrada ficam em **data/**. Para reproduzir a pipeline completa localmente, o arquivo **bronze_data.zip** deve estar na raiz do projeto e conter:

~~~text
alfabetizacao_inep/
censo_2022/
censo_escolar/
fundeb/
idhm/
~~~

Os dados completos não devem ser enviados ao repositório. O ZIP é extraído na Bronze e os arquivos internos são consumidos pelos jobs Silver.

### O que pode ser executado localmente

- notebooks em **notebooks/**;
- jobs Bronze, Silver e Gold;
- treinamento Spark ML;
- treinamento Pandas/scikit-learn.

O executor em **local/run_glue_job.py** imita somente as classes mínimas do Glue. Ele não reproduz IAM, workers, CloudWatch, rede, custos ou o runtime gerenciado da AWS. Portanto, uma execução local valida transformações e caminhos de dados, mas não substitui a validação no Glue.

### Subir o ambiente Docker

Na raiz do projeto:

~~~bash
docker compose build pipeline
mkdir -p local-data/bronze local-data/silver local-data/gold artifacts
~~~

O MinIO é opcional para comandos que usam **file://**. Para testar um endpoint compatível com S3 local:

~~~bash
docker compose up -d minio
~~~

O console fica em http://localhost:9001. As credenciais de desenvolvimento estão no arquivo **docker-compose.yml**.

### Bronze local

O repositório é montado em **/workspace** dentro do container:

~~~bash
docker compose run --rm pipeline python local/run_glue_job.py \
  src/pipeline/bronze/upload_zip_bronze.py \
  --JOB_NAME local-bronze \
  --ZIP_PATH file:///workspace/bronze_data.zip \
  --BUCKET_BRONZE file:///workspace/local-data/bronze
~~~

### Silver local

Repita o comando abaixo para cada script: **silver_alfabetizacao_inep.py**, **silver_censo_2022.py**, **silver_censo_escolar.py**, **silver_fundeb.py** e **silver_idhm.py**.

~~~bash
docker compose run --rm pipeline python local/run_glue_job.py \
  src/pipeline/silver/<nome-do-script>.py \
  --JOB_NAME local-silver \
  --BUCKET_BRONZE file:///workspace/local-data/bronze \
  --BUCKET_SILVER file:///workspace/local-data/silver
~~~

### Gold local

Os jobs Gold devem ser executados nesta ordem:

~~~bash
docker compose run --rm pipeline python local/run_glue_job.py \
  src/pipeline/gold/gold_consolidacao_alunos.py \
  --JOB_NAME local-gold-consolidacao \
  --BUCKET_SILVER file:///workspace/local-data/silver \
  --BUCKET_GOLD file:///workspace/local-data/gold

docker compose run --rm pipeline python local/run_glue_job.py \
  src/pipeline/gold/gold_ml_dataset.py \
  --JOB_NAME local-gold-ml \
  --BUCKET_GOLD file:///workspace/local-data/gold
~~~

O resultado será escrito em **local-data/gold/ml/ml_dataset_alunos/**, em Parquet e particionado por ano.

### Treinamento local

O fluxo Spark selecionado para a base completa é:

~~~bash
docker compose run --rm pipeline spark-submit \
  src/modeling/train_model_spark.py \
  --input file:///workspace/local-data/gold/ml/ml_dataset_alunos \
  --output file:///workspace/artifacts/model-spark
~~~

Como linha de base, também é possível usar Pandas/scikit-learn. Esse fluxo carrega toda a Gold na memória do driver e pode não ser viável em máquinas com pouca RAM:

~~~bash
python src/modeling/train_model.py \
  --input local-data/gold/ml/ml_dataset_alunos \
  --output artifacts/model-pandas
~~~

Os resultados incluem **metrics.json**, **aggregations.json** e o artefato do modelo.

## Execução no AWS Glue

### Pré-requisitos na AWS

1. Crie um bucket S3, ou prefixes em um bucket existente, para **bronze**, **silver**, **gold**, **modeling** e **scripts**.
2. Crie uma role do Glue com leitura nos inputs e escrita nos outputs.
3. Envie os scripts de **src/pipeline/** e **src/modeling/** ao S3.
4. Para **train_model.py**, envie **src/modeling/requirements-glue.txt** ao S3.
5. Crie jobs Spark na mesma região do bucket, preferencialmente com Glue 5.x e Python 3.11.

Exemplo de organização:

~~~text
s3://<bucket>/source/bronze_data.zip
s3://<bucket>/scripts/pipeline/...
s3://<bucket>/scripts/modeling/...
s3://<bucket>/scripts/requirements-glue.txt
s3://<bucket>/bronze/
s3://<bucket>/silver/
s3://<bucket>/gold/
s3://<bucket>/modeling/
~~~

### Ordem da pipeline no Glue

Crie um job Spark para cada script e execute-os nesta ordem.

| Ordem | Script | Parâmetros obrigatórios |
|---:|---|---|
| 1 | **src/pipeline/bronze/upload_zip_bronze.py** | **--ZIP_PATH** s3://&lt;bucket&gt;/source/bronze_data.zip; **--BUCKET_BRONZE** s3://&lt;bucket&gt;/bronze |
| 2 | **src/pipeline/silver/silver_alfabetizacao_inep.py** | **--BUCKET_BRONZE** s3://&lt;bucket&gt;/bronze; **--BUCKET_SILVER** s3://&lt;bucket&gt;/silver |
| 3 | **src/pipeline/silver/silver_censo_2022.py** | mesmos parâmetros Bronze/Silver |
| 4 | **src/pipeline/silver/silver_censo_escolar.py** | mesmos parâmetros Bronze/Silver |
| 5 | **src/pipeline/silver/silver_fundeb.py** | mesmos parâmetros Bronze/Silver |
| 6 | **src/pipeline/silver/silver_idhm.py** | mesmos parâmetros Bronze/Silver |
| 7 | **src/pipeline/gold/gold_consolidacao_alunos.py** | **--BUCKET_SILVER** s3://&lt;bucket&gt;/silver; **--BUCKET_GOLD** s3://&lt;bucket&gt;/gold |
| 8 | **src/pipeline/gold/gold_ml_dataset.py** | **--BUCKET_GOLD** s3://&lt;bucket&gt;/gold |

O parâmetro **--JOB_NAME** é fornecido automaticamente pelo Glue. Cadastre os demais em **Job details → Advanced properties → Job parameters**.

### Treinamento no Glue

O treinamento distribuído recomendado usa **src/modeling/train_model_spark.py**. Configure:

~~~text
--BUCKET_GOLD    s3://<bucket>/gold
--BUCKET_MODEL   s3://<bucket>/modeling
~~~

O job lê **gold/ml/ml_dataset_alunos**, compara regressão logística, Random Forest e GBT em uma amostra estratificada e re-treina o vencedor com a base completa de 2023–2024. Os artefatos ficam em um prefixo equivalente a:

~~~text
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/model/
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/metrics.json
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/aggregations.json
~~~

O treinamento alternativo **src/modeling/train_model.py** usa Pandas/scikit-learn e precisa destas dependências adicionais:

~~~text
--python-modules-installer-option    -r
--additional-python-modules          s3://<bucket>/scripts/requirements-glue.txt
~~~

Como esse fluxo transfere a Gold para Pandas no driver, prefira o treinamento Spark para a base completa.

### Permissões mínimas

A role do Glue precisa de **s3:GetObject** para scripts, dependências e dados de entrada, além de **s3:PutObject** e **s3:ListBucket** nos prefixes de saída. Também precisa das permissões padrão do Glue para publicar logs no CloudWatch.

## Reproduzir uma execução

Registre, junto aos artefatos:

- hash do commit Git;
- parâmetros do job;
- URIs e versões dos objetos S3 de entrada;
- versão do Glue, tipo e quantidade de workers;
- data da execução e métricas geradas.

Os jobs podem ser reexecutados com os mesmos parâmetros. Para reprodução histórica exata, habilite o versionamento do bucket S3 e registre o versionId do ZIP e dos demais objetos de entrada.
