import random
import numpy as np

class Perceptron :
    def __init__ (self, taxa_aprendizado = 1, epocas = 100, bias = 0.01) :
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.bias = bias
        self.pesos = []

    def inicializa_pesos(self, numero_de_entradas) :
        #gera valores aleatorios(entre -0.5 e 0.5) para os pesos e carrega o bias no começo de cada tupla no conjunto
        for i in range(numero_de_entradas) :
            self.pesos.append(round((random.randrange(0,100)/100 - 0.5), 2))
        self.pesos.insert(0, self.bias)

    def treina(self, amostras_treino, gabarito_treino, y) :
        #Faz o ajuste dos pesos para cada elemento do cinjunto amostral
        for j in range(len(amostras_treino)) :
            self.pesos[j] = self.pesos[j] + (self.taxa_aprendizado * (gabarito_treino - y) * amostras_treino[j])
                
    def sinal(self, u) :
        if u > 0:
            return 1
        else :
            return 0
    
    def soma(self, amostra) :
        #inicializa o potencial de ativação
        u = 0
        #calcula o petencial de ativação
        for i in range(len(amostra)) :
            u += self.pesos[i] * amostra[i]
        return u

    #carrega pesos salvos
    def carrega_pesos(self, pesos) :
        self.pesos = pesos



    



