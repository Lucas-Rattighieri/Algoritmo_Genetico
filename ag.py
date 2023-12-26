import numpy as np
from statistics import mode

class AlgoritmoGenetico:
    """
    Classe para implementação de um Algoritmo Genético para otimização de funções.

    Attributes:
        funcao (callable): Função a ser otimizada.
        num_variaveis (int): Número de variáveis da função.
        xlim (list): Tupla com limites inferior e superior das variáveis em uma lista [(lim_inferior, lim_superior)].
        num_populacao (int): Tamanho da população.
        num_genes (int): Número de genes (bits) que representam cada variável.
        num_geracoes (int): Número de gerações.
        taxa_crossover (float): Taxa de crossover.
        taxa_mutacao (float): Taxa de mutação.
        peso_esolha (float): Peso dado a avaliação dos individuos de melhor aptidão.
        x (list, optional): Variaveis para iniciar o algoritmo.
        args (list, optional): Argumentos adicionais para a função a ser otimizada.
        mostrar_iteracoes (bool, optional): Mostra a geracao e a aptidão do melhor individuo.
    """

    def __init__(self,
                funcao,
                num_variaveis: int,
                xlim: list,
                num_populacao: int,
                num_genes: int,
                num_geracoes: int,
                taxa_crossover: float,
                taxa_mutacao: float,
                peso_escolha: float = 0.5,
                x: list = None,
                args: list = [],
                mostrar_iteracoes = False
                ):

        self.funcao = funcao
        self.num_variaveis = num_variaveis
        self.args = args
        self.num_populacao = num_populacao - num_populacao % 2
        self.num_genes = num_genes
        self.num_geracoes = num_geracoes
        self.taxa_crossover = taxa_crossover
        self.taxa_mutacao = taxa_mutacao
        self.peso_escolha = peso_escolha

        self.mostrar_iteracoes = mostrar_iteracoes

        self.xmin = np.array(xlim)[:, 0]
        self.xmax = np.array(xlim)[:, 1]
        self.x = x
        self.aptidao_x = None
        self.__bitstring_x = None

        self._iniciar_x(x)

        self.run()


    def run(self):

        populacao = self._iniciar_populacao()

        t = 0

        val_funcao = self._aplicacao_funcao(populacao)

        avaliacao = self._avaliar(populacao, val_funcao)

        t += 1

        while (t <= self.num_geracoes):

            self._mostrar_iteracoes(populacao, t)

            populacao = self._selecionar(populacao, avaliacao)

            populacao = self._reproduzir(populacao)

            populacao = self._mutacao(populacao)

            val_funcao = self._aplicacao_funcao(populacao)

            avaliacao = self._avaliar(populacao, val_funcao)


            t += 1



    def _iniciar_x(self, x):
        if x is None:
            pass
        elif len(x) == self.num_variaveis:
            self.x = np.array(x)
            self.aptidao_x = self.funcao(self.x, *self.args)
            self.__bitstring_x = self._to_bitstring(self.x)
        else:
            raise ValueError("Parametros iniciais não condizem com o número de variaveis.")


    def _to_bitstring(self, x):

        xl = np.round((2 ** self.num_genes - 1) * (x - self.xmin) / (self.xmax - self.xmin), 0)

        array_bitstring = np.array([[int(i) for i in np.binary_repr(int(num), self.num_genes)[::-1]] for num in xl])
        
        return array_bitstring


    def _to_real(self, individuo):

        individuo = np.sum(individuo * 2 ** np.arange(self.num_genes-1, -1, -1), -1)

        x = self.xmin + individuo * (self.xmax - self.xmin) / (2 ** self.num_genes - 1)
        return x


    def _iniciar_populacao(self):

        populacao = np.random.randint(2,
                                    size=(self.num_populacao, self.num_variaveis, self.num_genes),
                                    dtype=bool
                                    )
        return populacao


    def _transformar_em_parametros(self, populacao):

        parametros = self._to_real(populacao)

        return parametros


    def _aplicacao_funcao(self, populacao):

        parametros = self._transformar_em_parametros(populacao)

        val_funcao = np.apply_along_axis(self.funcao, 1, parametros, *self.args)

        self._inserir_x(populacao, val_funcao)
        self._atualizar_x(parametros, val_funcao)

        return val_funcao


    def _inserir_x(self, populacao, val_funcao):
        if not self.x is None:
            ind = np.argmax(val_funcao)
            populacao[ind] = self.__bitstring_x
            val_funcao[ind] = self.aptidao_x


    def _atualizar_x(self, parametros, val_funcao):
        m_parametro = parametros[np.argmin(val_funcao)]
        aptidao = np.min(val_funcao)

        if self.aptidao_x is None:
            self.x = m_parametro
            self.aptidao_x = aptidao

        elif self.aptidao_x > aptidao:
            self.x = m_parametro
            self.aptidao_x = aptidao


    def _mostrar_iteracoes(self, populacao, iteracao):
        if (self.mostrar_iteracoes):
            val_funcao = self._aplicacao_funcao(populacao)
            print(f"Geração {iteracao}")
            print(f"Menor valor: {np.min(val_funcao)}, Moda: {mode(val_funcao)}")        


    def _avaliar(self, populacao, val_funcao):

        avaliacao = np.array([sum(1 for y in val_funcao if y <= x) for x in val_funcao])

        avaliacao = 1 / (avaliacao ** self.peso_escolha)
        avaliacao = avaliacao / np.sum(avaliacao)

        return avaliacao


    def _selecionar(self, populacao, avaliacao):
        indices = np.random.choice(self.num_populacao,
                         size=self.num_populacao,
                         p=avaliacao,
                         replace=True)

        return populacao[indices]


    def _reproduzir(self, populacao):


        indices = [i + (-1)**i for i in range(self.num_populacao)]

        populacao1 = populacao[indices]

        indices_corte = [i - i % 2 for i in range(self.num_populacao)]

        pontos_corte = np.random.choice(2,
                                   size=(self.num_populacao, self.num_variaveis, 1),
                                   p=[1-self.taxa_crossover, self.taxa_crossover],
                                   replace=True
                                   )

        pontos_corte = pontos_corte * np.random.randint(1,
                                              self.num_genes,
                                              size=(self.num_populacao, self.num_variaveis, 1)
                                              )
        pontos_corte = pontos_corte[indices_corte]

        mascara = np.tile(np.arange(self.num_genes).reshape(1, -1),
                            (self.num_populacao, self.num_variaveis, 1)
                            ) < pontos_corte

        populacao = (populacao & mascara) | (populacao1 & ~mascara)

        return populacao


    def _mutacao(self, populacao):

        pontos_mutacao = np.random.choice(2,
                                   size=(self.num_populacao, self.num_variaveis, self.num_genes),
                                   p=[1-self.taxa_mutacao, self.taxa_mutacao],
                                   replace=True)

        return populacao ^ pontos_mutacao
