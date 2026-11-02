# Decisões de modelagem

## Variáveis geográficas

### `city_code`

`city_code` será mantido como atributo do treinamento porque o produto será
usado principalmente nos municípios já conhecidos. Nesse contexto, é desejável
que o modelo aprenda o prior histórico de cada município: alunos do mesmo
município podem compartilhar uma probabilidade-base de alfabetização.

Como o código é uma categoria, ele deve ser convertido para texto antes do
treinamento. Não deve ser tratado como número contínuo, pois a distância entre
códigos IBGE não representa distância ou semelhança entre municípios.

Essa decisão reduz a generalização para municípios nunca observados. Por isso,
o conjunto de teste temporal deve manter municípios presentes no treinamento;
para medir o cenário de municípios novos, é necessário executar uma validação
geográfica separada.

### `state_abbreviation`

`state_abbreviation` tem cardinalidade muito menor, aproximadamente 27 UFs, e
representa diferenças regionais reais. Por isso, pode ser mantido como variável
categórica quando o modelo será aplicado aos mesmos estados representados no
treino.

Ainda assim, ele também pode permitir que o modelo memorize efeitos médios por
estado. A avaliação deve comparar obrigatoriamente dois cenários:

1. modelo com `state_abbreviation`;
2. modelo sem `state_abbreviation`.

Se o objetivo for generalizar para estados não observados, os estados devem ser
removidos dos atributos ou usados em uma validação geográfica, mantendo estados
inteiros exclusivamente no conjunto de teste.

## Validação recomendada

A divisão temporal continua sendo importante para evitar vazamento entre anos.
Para medir generalização geográfica, ela deve ser complementada por uma divisão
por estados ou municípios. Os resultados dos modelos com e sem variáveis
geográficas devem ser comparados usando as mesmas métricas e os mesmos grupos
de teste.
