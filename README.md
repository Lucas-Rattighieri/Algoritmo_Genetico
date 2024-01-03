# Algoritmo Genético

Este é um exemplo de implementação de um Algoritmo Genético em Python para a otimização de funções. O algoritmo é capaz de encontrar um conjunto de variáveis que minimiza uma função fornecida.

## Como usar

Para usar o Algoritmo Genético, siga estas etapas:

1. Importe a classe `AlgoritmoGenetico` do módulo.
2. Defina a função que você deseja otimizar.
3. Crie uma instância da classe `AlgoritmoGenetico`, passando a função, o número de variáveis, os limites das variáveis, o tamanho da população, o número de genes, o número de gerações, as taxas de crossover e mutação, entre outros parâmetros opcionais. O método `run()` é automaticamente chamado para execução do algoritmo.
4. Chame o atributo `x` da instancia criada para obter o conjuto de variáveis que minimiza a função.

## Parâmetros da Classe `AlgoritmoGenetico`

- `funcao`: A função a ser otimizada.
- `num_variaveis`: O número de variáveis da função.
- `xlim`: Uma lista de tuplas com os limites inferior e superior das variáveis.
- `num_populacao`: O tamanho da população.
- `num_genes`: O número de genes (bits) que representam cada variável.
- `num_geracoes`: O número de gerações.
- `taxa_crossover`: A taxa de crossover.
- `taxa_mutacao`: A taxa de mutação.
- `peso_escolha`: Peso dado à avaliação dos indivíduos de melhor aptidão (opcional, padrão é 0.5).
- `x`: Variáveis para iniciar o algoritmo (opcional).
- `args`: Argumentos adicionais para a função a ser otimizada (opcional).
- `mostrar_iteracoes`: Mostra a geração e a aptidão do melhor indivíduo (opcional, padrão é False).

## Exemplo de Uso

```python
# Exemplo de uso do Algoritmo Genético

def funcao_exemplo(x):
    return x[0]**2 + x[1]**2  # Função de exemplo

algoritmo = AlgoritmoGenetico(
    funcao=funcao_exemplo,
    num_variaveis=2,
    xlim=[(-5, 5), (-5, 5)],
    num_populacao=100,
    num_genes=10,
    num_geracoes=100,
    taxa_crossover=0.8,
    taxa_mutacao=0.1,
    mostrar_iteracoes=True
)

resultado = algoritmo.x
```
