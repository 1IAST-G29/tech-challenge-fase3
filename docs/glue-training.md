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
