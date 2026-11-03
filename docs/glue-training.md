# Treinamento no AWS Glue

O treinamento usa o mesmo `src/modeling/train_model.py` localmente e no AWS
Glue. No Glue, ele lê a Gold `ml/ml_dataset_alunos` do S3 usando Spark,
converte o resultado para Pandas no driver e treina os modelos Scikit-learn.

## Criar o job pela interface

No AWS Glue, abra **ETL jobs** e escolha **Create job**. Configure:

- **Engine:** Spark;
- **Glue version:** 5.1 (Python 3.11);
- **Script:** envie `src/modeling/train_model.py` para o S3 e selecione-o;
- **IAM role:** uma role com leitura na Gold e escrita no prefixo de modelos;
- **Worker type:** `G.4X`, com ao menos dois workers;
- **Job timeout:** 12 horas.

Embora o job use Spark para leitura, Scikit-learn treina no driver. Workers
adicionais não dividem o treino entre máquinas; o tipo `G.4X` dá 64 GB de RAM
ao nó para suportar a conversão para Pandas e os quatro modelos.

## Parâmetros do job

Inclua estes argumentos em **Job details → Advanced properties → Job parameters**:

```text
--BUCKET_GOLD       s3://<bucket>/gold
--BUCKET_MODEL      s3://<bucket>/modeling
```

`--JOB_NAME` é fornecido automaticamente pelo Glue. A saída ficará em um
prefixo único como:

```text
s3://<bucket>/modeling/alfabetizacao/runs/<job-name>-<timestamp>/model.joblib
s3://<bucket>/modeling/alfabetizacao/runs/<job-name>-<timestamp>/metrics.json
s3://<bucket>/modeling/alfabetizacao/runs/<job-name>-<timestamp>/aggregations.json
```

## Artefatos e resultados para apresentação

`metrics.json` reúne a comparação entre todos os modelos e o resultado do
vencedor. Além das métricas técnicas, contém:

- definição do alvo e do objetivo do experimento;
- divisão temporal entre treino e teste;
- distribuição das classes em ambos os conjuntos;
- interpretação da matriz de confusão em linguagem de negócio;
- importância das 20 variáveis mais relevantes, calculada por permutação no
  conjunto de teste.

`aggregations.json` não contém previsões de alunos individuais. Ele agrega o
risco estimado e as taxas observadas de não alfabetização por `city_code`, UF e
rede de ensino, para apoiar a apresentação a gestores.

## Dependências

Envie `src/modeling/requirements-glue.txt` ao S3, por exemplo para
`s3://<bucket>/scripts/requirements-glue.txt`. Adicione os parâmetros:

```text
--python-modules-installer-option    -r
--additional-python-modules          s3://<bucket>/scripts/requirements-glue.txt
```

Use uma role que tenha `s3:GetObject` para a Gold, para o script e para o
arquivo de dependências, além de `s3:PutObject` no prefixo de modelos. O job
precisa alcançar o PyPI para instalar as dependências; se ele estiver em uma
VPC privada, configure NAT ou publique wheels compatíveis em um repositório
interno.

## Execução local

```bash
python src/modeling/train_model.py \
  --input local-data/gold/ml/ml_dataset_alunos \
  --output artifacts/model
```

O caminho local não precisa de Spark nem de credenciais AWS.

## Alternativa distribuída com Spark ML

`src/modeling/train_model_spark.py` é a opção indicada para a base completa no
Glue. Ele executa o pré-processamento, o balanceamento de classes e o Random
Forest com Spark ML nos executores, sem transferir a Gold inteira para Pandas
no driver. Por isso, é mais apropriado para reduzir o tempo de execução e usar
efetivamente os workers configurados no job.

Ele compara o mesmo conjunto conceitual do treinamento Scikit-learn: baseline,
regressão logística, Random Forest e árvores de gradiente. Spark ML não oferece
o `HistGradientBoostingClassifier`; o candidato correspondente é registrado
como `gradient_boosted_trees` e usa `GBTClassifier`. O pré-processamento é
ajustado para cada perfil de modelo. Para reduzir tempo e custo, os candidatos
são comparados em uma amostra temporal estratificada de 20%; apenas o vencedor
é então treinado e avaliado na base completa. Para as árvores, `city_code` é
representado por feature hashing em 512 dimensões, evitando milhares de
atributos esparsos e a exigência de milhares de valores em `maxBins`. O
`metrics.json` informa essa estratégia, guarda a comparação e o modelo
vencedor é gravado em `model/`.

Crie um segundo job com o mesmo motor, role e parâmetros anteriores, mas envie
e selecione `src/modeling/train_model_spark.py`. Não é necessário informar
`requirements-glue.txt`, pois o Spark ML já faz parte do runtime do Glue.

Os artefatos serão gravados em um prefixo como:

```text
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/model/
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/metrics.json
s3://<bucket>/modeling/alfabetizacao-spark/runs/<job-name>-<timestamp>/aggregations.json
```

Para execução local, é necessário ter PySpark instalado:

```bash
spark-submit src/modeling/train_model_spark.py \
  --input local-data/gold/ml/ml_dataset_alunos \
  --output artifacts/model-spark
```

## Variante com transformações de atributos

`src/modeling/train_model_spark_transformed.py` executa a mesma comparação de
modelos Spark ML, mas aplica `log1p` antes da imputação nas colunas
`households`, `area`, `population_total`, `fundeb_amount`, `school_count`,
`total_enrollments`, `total_teachers`, `total_classes`, `pct_rural_schools`,
`pct_full_time_enrollments` e `pct_enrollments_zero_class_schools`.
Também força `education_network_type` a ser uma variável categórica e a aplica
one-hot encoding.

No Glue, selecione esse script e use os mesmos parâmetros `--BUCKET_GOLD` e
`--BUCKET_MODEL`. Os resultados serão gravados em
`alfabetizacao-spark-transformed/runs/`, e o `metrics.json` lista as
transformações aplicadas.

## Variante reduzida de atributos

`src/modeling/train_model_spark_reduced.py` usa a mesma estratégia otimizada
de seleção em amostra e treino completo do vencedor, mas remove antes do treino
`households`, `total_teachers`, `total_enrollments`, `total_classes`,
`fundeb_amount`, `school_count`, `hdi`, `hdi_income`,
`income_per_capita_mean` e `pct_white`.

Como a exclusão tem prioridade, `log1p` é aplicado somente em `area`,
`population_total`, `pct_rural_schools`, `pct_full_time_enrollments` e
`pct_enrollments_zero_class_schools`. A coluna `education_network_type` é
forçada como categórica e recebe one-hot encoding. No Glue, use os mesmos
parâmetros de bucket e selecione esse script; os artefatos ficam em
`alfabetizacao-spark-reduced/runs/`.
