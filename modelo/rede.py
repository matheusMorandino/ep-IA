import perceptron as perceptron
from tqdm import tqdm
import pandas as pd 

class Rede() :
    def __init__(self, n_classes, saidas, amostras, taxa_aprendizado = 1, epocas = 50, bias = 0.01) :
        super().__init__()
        self.n_classes = n_classes                        #número de classes presentes
        self.amostras = amostras.to_numpy().tolist()      #amostras para treino
        self.taxa_aprendizado = taxa_aprendizado          #taxa de aprendizado dos perceptrons
        self.epocas = epocas                              #numero de épocas para treino
        self.bias = bias                                  #bias para cada perceptron
        self.camada = []                                  #armazena os perceptrons que serão usados
        self.saidas = self.gera_saidas_formatadas(saidas) #saidas formatadas, e.g.: 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.erros = []                                   #armazena o número de erros absolutos em cada época de treino

    def formatador_saida(self, num) :
        saida_format = []
        for i in range(self.n_classes) :
            saida_format.append(1 if i == num else 0)
        return saida_format

    def gera_saidas_formatadas(self, saidas) :
        saidas_format_lista = []
        for i in range(len(saidas)) :
            saidas_format_lista.append(self.formatador_saida(saidas[i]))
        return saidas_format_lista

    def treina(self) :
        #cria camada de perceptrons
        print(">> Inicializando rede")
        for i in range(self.n_classes) :
            self.camada.append(perceptron.Perceptron(taxa_aprendizado = self.taxa_aprendizado, epocas = self.epocas, bias = self.bias))
            self.camada[i].inicializa_pesos(len(self.amostras[0]))

        #insere 1 no início de cada tupla do conjunto de amostras
        for i in range(len(self.amostras)) :
            self.amostras[i].insert(0, 1)

        #calcula erro da época 0 e inicializa o contador de erros
        print(">> Erros da epoca 0")
        erro_count = 0
        for i in tqdm(range(len(self.amostras))) :
            if(self.testa_rede(self.amostras[i]) != self.saidas[i].index(max(self.saidas[i]))) :
                erro_count = erro_count + 1 
        self.erros.append(erro_count)

        #controle de épocas
        print(">> Treinando perceptrons")
        for epocas_count in range(0, self.epocas) :
            print("\n>> Epoca " + str(epocas_count+1) + "/" + str(self.epocas))
            #contador do erro absoluto por epoca de treino
            erro_count = 0

            #para cada amostra no conjunto de treino
            for i in tqdm(range(len(self.amostras))) :

                #lista dos sinais e somas(potencial de ativação) de cada perceptron para uma dada amostra
                lista_sinais = []

                for j in range(len(self.camada)) :
                    lista_sinais.append(self.camada[j].sinal(self.camada[j].soma(self.amostras[i]))) 

                #aplica o a função de apendizado para uma dada amostra
                for j in range(len(self.camada)) :
                    if(lista_sinais[j] != self.saidas[i][j]) :
                        self.camada[j].treina(amostras_treino = self.amostras[i], gabarito_treino = self.saidas[i][j], y = lista_sinais[j])
            
            #calcula os erros de uma dada época
            for i in tqdm(range(len(self.amostras))) :
                if(self.testa_rede(self.amostras[i]) != self.saidas[i].index(max(self.saidas[i]))) :
                    erro_count = erro_count + 1

            #carrega os erros computados em uma dada época na lista
            self.erros.append(erro_count)

    #respota da rede para amostras já formatadas
    def testa_rede(self, amostra) :
        #calcula a soma de cada perceptron
        lista_soma = []
        for i in range(len(self.camada)) :
            lista_soma.append(self.camada[i].soma(amostra))

        return lista_soma.index(max(lista_soma))

    #resposta da rede para amostras que não receberam o 1 no começo da lista
    def resposta_rede(self, amostra) :
        #insere 1 no início da amostra
        amostra.insert(0,1)
        #calcula a soma de cada perceptron
        lista_soma = []
        for i in range(len(self.camada)) :
            lista_soma.append(self.camada[i].soma(amostra))

        return lista_soma.index(max(lista_soma))

    #salva os pesos dos percetrons
    def pesos_perceptrons(self) :
        lista_pesos = []
        for i in range(len(self.camada)) :
            lista_pesos.append(self.camada[i].pesos)

        return pd.DataFrame(lista_pesos)

    #carrega pesos salvos de uma execução anterior 
    def carrega_pesos_salvos(self, pesos_salvos) :
        print(">> Caregando pesos salvos")
        for i in range(self.n_classes) :
            self.camada.append(perceptron.Perceptron(taxa_aprendizado = self.taxa_aprendizado, epocas = self.epocas, bias = self.bias))
            self.camada[i].carrega_pesos(pesos_salvos.iloc[i, :].tolist())
        

                
