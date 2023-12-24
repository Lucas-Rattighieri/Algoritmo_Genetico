# Algoritmo Genético 

Este é um exemplo de implementação de um Algoritmo Genético em Python para otimização de funções.

## Descrição

A classe `AlgoritmoGenetico` é usada para otimizar uma função específica por meio de um Algoritmo Genético. Ela permite a configuração de parâmetros como:
- a função a ser otimizada;
- o número de variáveis;
- os limites das variáveis;
- o tamanho da população;
- o número de genes que representam cada variável;
- o número de gerações;
- as taxas de crossover e mutação;
- parametros da função.

## Uso

Para usar esta classe, siga os passos abaixo:

1. Crie uma instância da classe `AlgoritmoGenetico`, passando os parâmetros necessários.
2. Chame o método `run()` para executar o algoritmo e obter os parâmetros otimizados.

Exemplo de uso:

```python
# Define a função a ser otimizada
def funcao(x):
    return x[0]**2 + x[1]**2  


algoritmo = AlgoritmoGenetico(
    funcao=funcao,
    num_variaveis=2,
    xlim=[(-10, 10), (-10, 10)], 
    num_populacao=100,
    num_genes=10,
    num_geracoes=50,
    taxa_crossover=0.8,
    taxa_mutacao=0.1
)

resultado = algoritmo.run()

print("Parâmetros otimizados:", resultado)
```
