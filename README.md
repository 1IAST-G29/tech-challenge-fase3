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

## Dados

A base de modelagem será uma nova tabela Gold, `ml_dataset_alunos`, construída a
partir das tabelas tratadas na Fase 2 e, quando necessário, de fontes públicas como
IBGE e Censo Escolar. Dados completos, identificadores pessoais, credenciais e
artefatos de modelos não devem ser enviados ao repositório.

## Próximas etapas

1. Inventariar o esquema e a cobertura temporal das tabelas Silver/Gold da Fase 2.
2. Construir a tabela Gold de atributos para ML sem vazamento de dados.
3. Realizar análise exploratória.
4. Treinar, validar e interpretar modelos.
5. Gerar ranking agregado de risco, documentação e apresentação executiva.
