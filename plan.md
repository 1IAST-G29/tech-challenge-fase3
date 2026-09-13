# Plano de ação — Tech Challenge Fase 3

## Resumo do desafio

A fase 3 transforma os dados de alfabetização preparados na fase 2 em uma solução
de Machine Learning que estima se um aluno será alfabetizado ou não. A entrega deve
combinar análise exploratória, um modelo confiável e explicável e recomendações que
ajudem gestores a identificar riscos e priorizar ações. A resposta do modelo não é
uma decisão automática sobre alunos, escolas ou professores.

O alvo é \`alfabetizado\` (0 ou 1). Como ele foi definido pela proficiência, a coluna
\`proficiencia\` e qualquer informação disponível apenas após a avaliação não podem
entrar no treino. O projeto também precisa entregar código reproduzível, métricas,
visualizações, README, documentação e vídeo executivo de até cinco minutos.

## Direção da solução

Construir um sistema reproduzível de classificação que atribua a cada aluno uma
probabilidade de não alfabetização. A fonte de verdade será uma nova tabela Gold,
`ml_dataset_alunos`, derivada das tabelas já tratadas na fase 2 e, se fizer sentido,
de indicadores públicos territoriais. O projeto deve entregar previsões explicáveis
e uma lista priorizada de municípios/regiões para apoiar decisões — não uma decisão
automática sobre alunos.

### Ponto de partida identificado na fase 2

A fase 2 já usa AWS Glue, S3, Lambda, Kinesis, Secrets Manager, Glue Crawler e
Athena, com camadas Bronze/Silver/Gold. Os dados de aluno na Silver possuem, entre
outros campos, `ano`, `id_municipio`, `id_escola`, `serie`, `rede`,
`presenca`, `preenchimento_caderno`, `alfabetizado`, `proficiencia` e
`peso_aluno`.

As Golds atuais (`resultado_uf_ano`, `comparativo_anual_uf` e
`media_regiao_ano`) são agregadas; portanto, ainda não são uma base suficiente para
o modelo por aluno. A primeira entrega técnica desta fase será ampliar a Gold com a
tabela de atributos de ML. A ideia é reaproveitar a infraestrutura, sem depender do
fluxo de streaming mockado para treinar o modelo.

## Arquitetura proposta

~~~mermaid
flowchart LR
    A[S3 Silver da fase 2<br/>alunos, município e metas] --> B[Glue/PySpark<br/>criação de atributos]
    X[Fontes externas<br/>IBGE/Censo Escolar, opcionais] --> B
    B --> C[S3 Gold<br/>ml_dataset_alunos em Parquet]
    C --> D[Athena / validações]
    C --> E[Python local ou SageMaker<br/>EDA, treino e avaliação]
    E --> F[Artefato versionado<br/>pipeline + modelo + métricas]
    E --> G[Relatório e visuais<br/>riscos e fatores]
    F --> H[Opcional: endpoint ou job batch]
~~~

### Tabela Gold `ml_dataset_alunos`

Definir claramente o instante da previsão antes de criar atributos. A recomendação
é prever o resultado de um aluno no ano `t` usando somente dados conhecidos antes
da avaliação daquele ano, incluindo medidas agregadas do ano `t-1`.

| Grupo | Exemplos de atributos | Regra importante |
| --- | --- | --- |
| Identificação técnica | `id_aluno`, `id_escola`, `id_municipio`, `ano` | Servem para união/auditoria; IDs puros não entram no modelo. |
| Educacionais | série, rede, presença e dados disponíveis antes da prova | Confirmar o momento de coleta de cada campo. |
| Territoriais | UF, região, porte/população do município | Unir por código IBGE e ano de referência. |
| Histórico | taxa municipal/UF, distância para a meta, evolução anual | Usar somente valores de `t-1` ou anteriores. |
| Socioeconômicos opcionais | indicadores IBGE/Censo Escolar/FUNDEB | Documentar fonte, cobertura, ano e chave de união. |
| Alvo | `alfabetizado` (0 ou 1) | Derivado da proficiência >= 743, quando essa for a regra da base. |

Nunca incluir `proficiencia` como atributo. Também excluir o próprio
`alfabetizado`, campos calculados a partir dele e indicadores agregados do mesmo
ano que incorporem o aluno avaliado. Antes de aceitar `preenchimento_caderno` ou
qualquer dado da prova, confirmar se ele é conhecido no momento da previsão; se não
for, removê-lo. Essa regra é a principal proteção contra vazamento de dados.

## Estrutura recomendada do repositório

~~~text
tech-challenge-fase3/
├── codex/                         # documentos de apoio deste planejamento
├── data/
│   ├── sample/                     # amostra pequena e sem dados sensíveis
│   └── external/                   # fontes externas; não versionar dados grandes
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modelagem.ipynb
│   └── 03_resultados_executivos.ipynb
├── src/
│   ├── data/                       # leitura, contratos e validações
│   ├── features/                   # criação da Gold ml_dataset_alunos
│   ├── preprocessing/              # transformadores reutilizáveis
│   ├── modeling/                   # treino, busca de parâmetros e predição
│   ├── evaluation/                 # métricas, gráficos e explicabilidade
│   ├── visualization/              # visuais e tabelas executivas
│   └── aws/                        # job Glue e orquestração/infraestrutura
├── tests/
│   ├── unit/
│   └── integration/
├── reports/
│   ├── figures/
│   └── model_card.md
├── infra/                          # IaC (Terraform ou AWS SAM)
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── README.md
├── .gitignore
└── .github/workflows/ci.yml
~~~

Os notebooks serão usados para investigação e narrativa; a lógica que precisa ser
reexecutada deve ficar em `src/`, para que testes e a nuvem usem exatamente o mesmo
código. Dados completos, chaves, arquivos `.env`, modelos grandes e resultados
gerados não devem entrar no Git.

## Tecnologias

### Para desenvolver e rodar localmente

| Necessidade | Escolha | Motivo |
| --- | --- | --- |
| Linguagem | Python 3.11 | Bom suporte a dados e ML. |
| Dados locais | Pandas e PyArrow; PySpark somente na criação/validação da Gold | Simples para amostras; compatível com Parquet e Glue. |
| Machine Learning | scikit-learn | Possui pipeline, validação, transformação e métricas em uma solução madura. |
| Modelos candidatos | DummyClassifier, Regressão Logística e HistGradientBoosting/Random Forest | Linha de base interpretável e alternativas não lineares para comparar. |
| Explicabilidade | importância por permutação e SHAP | Explica fatores globais e casos individuais com cautela. |
| Qualidade | pytest, ruff e pre-commit | Testes, padronização e verificação rápida. |
| Experimentos | MLflow local (opcional inicialmente) | Guarda parâmetros, métricas e artefatos sem planilhas manuais. |
| Ambiente | `venv` + `pip`, JupyterLab e Make | Caminho leve e reproduzível. |

Começar com versões fixadas em `requirements.txt` e comandos como `make test`,
`make train` e `make report`. Docker pode ser incluído depois, caso a equipe
precise de ambientes idênticos; não é obrigatório para a primeira versão funcional.

### Para automatizar na AWS

| Etapa | Serviço proposto | Papel |
| --- | --- | --- |
| Dados | S3 + Glue Data Catalog + Athena | Manter a Gold em Parquet, catalogar e auditar a base. |
| Atributos | AWS Glue PySpark | Criar/atualizar `ml_dataset_alunos` em escala. |
| Treino agendado | AWS Glue Python Shell ou SageMaker Processing/Training | Glue para volume pequeno e pipeline simples; SageMaker se houver necessidade de rastrear e servir modelos. |
| Orquestração | Step Functions + EventBridge | Executar Gold, validação, treino e relatório em ordem/agendamento. |
| Artefatos | S3 + MLflow (opcional) ou SageMaker Model Registry | Guardar modelo, métricas, dados de referência e versão. |
| Qualidade e alertas | CloudWatch + validações no código | Registrar falhas e impedir treino com dados inválidos. |
| Infraestrutura | Terraform (preferência) ou AWS SAM | Criar recursos de forma revisável e repetível. |
| CI | GitHub Actions | Rodar lint e testes em pull requests; fazer deploy por OIDC e aprovação. |

Para este desafio, a recomendação é: criar a Gold no Glue, treinar inicialmente
localmente com uma amostra/exportação Parquet e só depois automatizar treino batch na
AWS. Um endpoint online não é necessário para cumprir o enunciado; uma execução batch
que gere probabilidades e um ranking de risco é mais alinhada ao caso e mais
econômica. Nunca colocar chaves AWS no repositório: usar IAM roles, OIDC no CI e
Secrets Manager quando houver segredo externo.

## Plano em etapas

### 1. Descoberta e contrato dos dados

1. Inventariar esquemas, cobertura de anos, contagens, chaves e nulos das tabelas
   Silver/Gold da fase 2 no Athena ou Spark.
2. Confirmar a unidade de análise: uma linha por aluno e ano de avaliação.
3. Escrever um dicionário de dados com origem, definição, período disponível e se
   cada coluna pode ser conhecida antes da previsão.
4. Escolher no máximo uma fonte externa inicial, somente se ela possuir chave
   municipal confiável e cobertura temporal adequada.
5. Definir uma regra explícita: por exemplo, "no início do ano t, estimar o risco do
   resultado de t".

**Saída:** `reports/data_dictionary.md`, consulta de diagnóstico e contrato de
esquema da `ml_dataset_alunos`.

### 2. Criar a base Gold de Machine Learning

1. Implementar `src/features/build_ml_dataset.py` e/ou o job Glue equivalente.
2. Juntar aluno, município, UF, metas e dados externos por chaves e ano.
3. Criar atributos históricos com janelas/lag por município e UF.
4. Aplicar verificações: uma linha por aluno/ano, alvo apenas 0/1, ausência de
   duplicidades e percentuais aceitáveis de nulos.
5. Gravar Parquet particionado por `ano` em
   `s3://.../gold/ml_dataset_alunos/` e catalogar a tabela.

**Saída:** conjunto de dados reproduzível, mais um pequeno `data/sample` anônimo
para testes locais.

### 3. EDA e hipóteses

1. Produzir um notebook que pode ser executado do início ao fim.
2. Avaliar equilíbrio do alvo, nulos, duplicidades e comportamento por ano, UF,
   região, rede e município.
3. Criar gráficos de distribuição, tendência, mapa/tabela de risco e associações.
4. Registrar hipóteses e limitações; não chamar correlação de causalidade.

**Saída:** `notebooks/01_eda.ipynb`, imagens em `reports/figures` e conclusões que
orientam os atributos e a validação.

### 4. Modelagem e validação confiável

1. Separar um teste final antes de ajustar modelos. Preferir corte temporal: treinar
   em anos anteriores e testar no último ano. Se houver apenas um ano, usar uma
   separação por município/escola e declarar essa limitação.
2. Dentro do treino, usar validação cruzada agrupada (`StratifiedGroupKFold`) por
   município ou escola, quando aplicável.
3. Criar `Pipeline` do scikit-learn com `ColumnTransformer`: mediana para números,
   indicador de ausência quando útil, transformação de categorias desconhecidas e
   modelo. Ajustar todos os transformadores somente com dados de treino.
4. Comparar um modelo ingênuo, Regressão Logística e ao menos um modelo de árvores.
   Fazer busca moderada de parâmetros somente dentro da validação.
5. Escolher a métrica principal antes de olhar o teste. Sugestão: F1 ou recall da
   classe "não alfabetizado", acompanhado de precision; apresentar também
   ROC-AUC/PR-AUC quando fizer sentido.
6. Definir o limiar de risco pela validação e reportar a matriz de confusão do teste
   uma única vez para o modelo escolhido.

**Saída:** código em `src/modeling`, métricas reproduzíveis, artefato serializado e
`reports/model_card.md`.

### 5. Interpretar e gerar inteligência aplicável

1. Calcular importância por permutação; usar SHAP para o melhor modelo se o volume e
   a tecnologia permitirem.
2. Gerar probabilidades por aluno e agregá-las por município, UF e região. Para não
   expor pessoas, o relatório executivo deve mostrar agregados e suprimir grupos
   pequenos.
3. Criar ranking de risco, mapa/tabela de municípios e grupos de perfis semelhantes
   (clusterização exploratória separada da previsão, se agregar valor).
4. Relacionar os resultados às metas, sempre distinguindo previsão de causalidade e
   recomendação de política pública.

**Saída:** visualizações executivas, insights priorizados e limitações explícitas.

### 6. Automatizar, testar e preparar a entrega

1. Criar testes unitários para atributos, prevenção de colunas proibidas, esquema e
   métricas; incluir um teste de integração com a amostra Parquet.
2. Configurar GitHub Actions para `ruff` e `pytest` a cada pull request.
3. Com Terraform/SAM, automatizar Glue + S3 + catálogo; acrescentar Step Functions
   para o treino batch somente após o fluxo local estar validado.
4. Escrever README completo: contexto, objetivo, fontes, arquitetura, execução,
   modelo, métricas, insights, limites, custo/segurança e evolução futura.
5. Criar roteiro do vídeo de cinco minutos: problema (30s), dados e método (60s),
   principais achados (90s), produto/decisão (90s), limites e próximos passos (30s).

**Saída:** repositório pronto para avaliação, pipeline reproduzível e apresentação
executiva.

## Critérios de aceite práticos

- A construção da Gold pode ser reexecutada e produz o mesmo esquema esperado.
- Nenhuma coluna que revele o alvo (`proficiencia`, alvo ou derivados) entra no
  treino.
- O código separa treino, validação e teste de forma justificável e reproduzível.
- O melhor modelo é comparado a uma linha de base e possui métricas por classe.
- Há explicação das variáveis influentes e das limitações.
- README e vídeo deixam claro como a previsão ajudaria gestores sem tratar a saída
  como decisão automática ou causal.

## Riscos e decisões a tomar cedo

1. **Poucos anos de dados:** sem histórico suficiente, não é possível comprovar uma
   previsão de futuro. Usar teste por grupos e apresentar isso como limitação;
   enriquecer com anos adicionais antes de prometer previsão de metas.
2. **Dados de aluno sensíveis:** não publicar identificadores, previsões individuais
   ou microdados. Aplicar mínimo necessário, acesso IAM e agregação nas saídas.
3. **Vazamento de dados:** manter a definição do instante de previsão e testes que
   bloqueiem colunas proibidas.
4. **Desbalanceamento:** se houver poucos não alfabetizados, avaliar `class_weight`,
   limiar de decisão e precision/recall; não depender somente de acurácia.
5. **Escopo:** entregar primeiro a Gold, EDA, um modelo reproduzível e relatório.
   Dashboard, endpoint e várias fontes externas são evoluções, não pré-requisitos.
