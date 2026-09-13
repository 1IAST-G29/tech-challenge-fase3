# Instruções para agentes de IA

Este arquivo orienta qualquer agente que atue neste repositório — Codex, Claude,
Copilot ou outro. Siga estas regras antes de criar, alterar ou remover arquivos.

## 1. Contexto do projeto

Este é o **Tech Challenge — Fase 3**, sobre predição e inteligência analítica para
alfabetização no Brasil.

O objetivo é criar uma solução de Machine Learning que estime o risco de um aluno
ser classificado como alfabetizado ou não alfabetizado, usando dados educacionais,
territoriais e socioeconômicos. A solução deve gerar análises interpretáveis e úteis
para gestores públicos, não decisões automáticas sobre pessoas.

Leia [README.md](README.md) e [plan.md](plan.md) antes de iniciar trabalho
relevante. O plano registra a arquitetura, etapas e critérios de aceite.

## 2. Princípios obrigatórios de dados e modelagem

- A variável-alvo é `alfabetizado` (0 ou 1).
- **Nunca use `proficiencia` como atributo de entrada.** Na Fase 2 ela foi usada
  para derivar `alfabetizado`; usá-la no treino cria vazamento de dados.
- Não use colunas disponíveis apenas depois da avaliação, nem agregações do mesmo
  ano que incluam o aluno cuja previsão está sendo feita.
- Defina e documente o instante da previsão. Como regra padrão, para prever o ano
  `t`, use dados conhecidos antes da avaliação em `t` e indicadores históricos de
  `t-1` ou anteriores.
- IDs de aluno, escola e município podem servir para união, auditoria e separação de
  dados, mas não devem entrar diretamente no modelo.
- Trate dados de aluno como sensíveis: não publique microdados, identificadores,
  previsões individuais ou credenciais.
- Diferencie associação estatística de causalidade em toda análise e apresentação.

## 3. Arquitetura prevista

A Fase 2 já usa S3, AWS Glue, Glue Data Catalog e Athena, com camadas
Bronze/Silver/Gold. Nesta fase, a referência para ML será uma nova tabela Gold:

`ml_dataset_alunos`

Ela terá uma linha por aluno e ano, atributos validados e o alvo. Será construída a
partir da Silver da Fase 2, das Golds existentes e, somente quando justificado, de
fontes públicas externas com chave municipal e período confiáveis.

Fluxo esperado:

~~~text
Silver / fontes externas
        -> job de atributos (Glue/PySpark)
        -> Gold ml_dataset_alunos em Parquet
        -> validação de qualidade e catálogo Athena
        -> EDA, treino e avaliação em Python
        -> artefatos, relatório e ranking agregado de risco
~~~

## 4. Organização do código

Mantenha esta organização ao adicionar implementação:

~~~text
data/             # amostras pequenas; dados completos não entram no Git
notebooks/        # EDA e narrativa; sem lógica crítica exclusiva
src/
  data/           # leitura, contratos e validações
  features/       # construção de atributos e Gold de ML
  preprocessing/  # transformadores reutilizáveis
  modeling/       # treino, busca de parâmetros e predição
  evaluation/     # métricas e explicabilidade
  visualization/  # gráficos e tabelas executivas
  aws/            # jobs Glue e integração AWS
tests/
  unit/
  integration/
reports/
  figures/
infra/            # Terraform ou AWS SAM, se adotado
~~~

Use notebooks para explorar e comunicar. Mova para `src/` qualquer lógica necessária
para reproduzir, testar ou executar em nuvem.

## 5. Padrões de implementação

- Use Python 3.12 e mantenha dependências em `requirements.txt`.
- Use `scikit-learn Pipeline` e `ColumnTransformer` para que imputação,
  transformações e modelo sejam ajustados somente com dados de treino.
- Comece com uma linha de base simples e compare-a com modelos mais expressivos.
  Sugestão: DummyClassifier, Regressão Logística e HistGradientBoosting ou Random
  Forest.
- Preserve um conjunto de teste final. Prefira validação temporal; se não houver anos
  suficientes, use validação agrupada por município ou escola e registre a limitação.
- Avalie mais que acurácia: matriz de confusão, precision, recall, F1 e, quando
  adequado, ROC-AUC e PR-AUC. Dê atenção ao recall da classe não alfabetizada.
- Registre sementes aleatórias, versões de dados, parâmetros e métricas.
- Produza explicabilidade com importância por permutação e SHAP quando viável.
- Escreva funções pequenas, tipadas quando fizer sentido, com mensagens de erro
  claras e sem valores secretos no código.

## 6. Qualidade antes de concluir uma alteração

1. Inspecione o impacto em arquivos existentes antes de editá-los.
2. Não sobrescreva ou reverta mudanças de outra pessoa/agente sem autorização.
3. Adicione ou atualize testes para regras novas em `src/`.
4. Execute as verificações aplicáveis:
   - `ruff check .`
   - `pytest`
5. Atualize README, plano ou documentação quando mudar arquitetura, comando de uso,
   contrato de dados, métrica ou resultado.
6. Informe com clareza o que mudou, como foi verificado e qualquer limitação.

Se não houver dados de produção disponíveis, use uma amostra sintética ou anonimizada
somente para desenvolvimento e testes. Declare explicitamente suposições sobre
esquemas, cobertura temporal e qualidade dos dados.

## 7. AWS, segurança e custos

- Nunca faça commit de chaves AWS, tokens, arquivos `.env`, credenciais ou dados
  confidenciais.
- Use IAM roles, Secrets Manager e OIDC no CI quando houver integração externa.
- Prefira Parquet e particionamento por ano para dados analíticos.
- Antes de criar recursos pagos, descreva o impacto e confirme que a criação está no
  escopo solicitado.
- Priorize execução local e jobs batch. Endpoint online, dashboard e múltiplas fontes
  externas são evoluções, não pré-requisitos do desafio.

## 8. Git e colaboração

- Trabalhe em branches descritivas, por exemplo:
  `feat/ml-dataset`, `feat/model-baseline` ou `docs/readme`.
- Faça commits pequenos e objetivos, usando mensagens como
  `feat: cria pipeline de pré-processamento` ou
  `test: cobre validação do dataset de ML`.
- Não force push, não faça reset destrutivo e não altere configuração remota sem
  solicitação explícita.
- Faça pull request com resumo, testes executados, riscos e evidências quando o fluxo
  de equipe estiver ativo.

## 9. Entregáveis esperados

A entrega final deverá conter:

- código organizado e reproduzível;
- tabela Gold de atributos para ML;
- análise exploratória e visualizações;
- pipeline de pré-processamento, treino e validação;
- avaliação, interpretabilidade e limitações;
- ranking/agregação de risco para municípios, UFs e regiões;
- README, documentação técnica e roteiro/vídeo executivo de até cinco minutos.

Em caso de conflito entre este arquivo e uma instrução explícita do usuário, siga a
instrução do usuário e documente o impacto.
