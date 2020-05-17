import random
import numpy as np

class Perceptron :
    def __init__ (self, taxa_aprendizado = 1, epocas = 100, bias = 0.01) :
        self.taxa_aprendizado = taxa_aprendizado
        self.epocas = epocas
        self.bias = bias
        self.pesos = np.array([])

    def inicializa_pesos(self, numero_de_entradas) :
        #gera valores aleatorios(entre -0.5 e 0.5) para os pesos e carrega o bias no começo de cada tupla no conjunto
        self.pesos = np.append(self.pesos, np.array(self.bias))
        for i in range(numero_de_entradas) :
            rand = round((random.randrange(0,100)/100 - 0.5), 2)
            self.pesos = np.append(self.pesos, np.array(rand))

    def treina(self, amostras_treino, gabarito_treino, y) :
        #Faz o ajuste dos pesos para uma certa amostra
        const = self.taxa_aprendizado * (gabarito_treino - y)
        delta = const * np.array(amostras_treino)
        self.pesos = self.pesos + delta
                
    def sinal(self, u) :
        if u > 0:
            return 1
        else :
            return 0
    
    def soma(self, amostra) :
        amostra = np.transpose(np.array(amostra))
        return amostra.dot(self.pesos)

    #carrega pesos salvos
    def carrega_pesos(self, pesos) :
        self.pesos = pesos



    



