import csv
import rede as rede
import pandas as pd 
import numpy as np 
from tqdm import tqdm

#função para criação de conjunto de treino
def cria_set_treino (folds, index, r_fold) :
    set_treino = pd.DataFrame()
    for i in range(r_fold) :
        if i != index :
            set_treino = set_treino.append(folds[i], ignore_index=True)
    return set_treino

#carrega o conjuntos de dados
treino = pd.read_csv('Conjuntos MNIST/mnist_treinamento.csv').to_numpy()
teste = pd.read_csv('Conjuntos MNIST/mnist_teste.csv').to_numpy()

conjunto = np.concatenate((treino, teste))

conjunto = pd.DataFrame(conjunto).sample(frac=1).reset_index(drop=True)

#número de folds
r_fold = 10
aux = int(round(len(conjunto) / r_fold))

#lista de folds
folds = []

for i in range(r_fold) :
    folds.append(conjunto[i*aux:(i+1)*aux])

#Armazenamento dos erros em cada etapa da avaliação
df_erros = pd.DataFrame(columns=['acertos', 'erros'])

#inicializa testes
print(">> Iniciando " + str(r_fold) + "-fold cross-validation")
for fold_etapa in range(r_fold) :
    print(">> Inicializando etapa usando para teste o fold " + str(fold_etapa))
    #criando sets de teste e treino apartir dos folds
    set_teste = pd.DataFrame(folds[fold_etapa]).reset_index(drop=True)
    set_treino = cria_set_treino(folds=folds, index=fold_etapa, r_fold=r_fold).reset_index(drop=True)

    #separando amostras de gabaritos
    gabarito_teste = set_teste.iloc[:, 0]
    amostras_teste = set_teste.iloc[:, 1:len(set_teste)-1].div(255)
    gabarito_treino = set_treino.iloc[:, 0]
    amostras_treino = set_treino.iloc[:, 1:len(set_treino)-1].div(255)

    #criando rede
    rede_percep = rede.Rede(n_classes = 10, 
                    saidas = gabarito_treino,
                    amostras = amostras_treino,
                    taxa_aprendizado = 0.1, 
                    epocas = 50, 
                    bias = 0.1)

    #treinando rede no set formado
    rede_percep.treina()

    #aplicação da rede no conjunto de teste
    valores_previstos = []

    print(">> Testando rede no conjunto de teste")
    for i in tqdm(range(len(amostras_teste))) :
        valores_previstos.append(rede_percep.resposta_rede(amostras_teste.iloc[i, :].tolist()))

    #número de erros e acertos sobre o conjunto de teste
    acertos = 0
    for i in range(len(gabarito_teste)) :
        if valores_previstos[i] == gabarito_teste[i] :
            acertos = acertos + 1
    erros = len(gabarito_teste) - acertos

    df = pd.DataFrame({'acertos' : [acertos], 'erros' : [erros]})
    df_erros = df_erros.append(df)
    
    #exportando os resultados para não ter que executar o código toda hora
    #erro absoluto do treino em cada época
    df = pd.DataFrame(rede_percep.erros)
    nome = 'erros_absolutos_epoca_' + str(fold_etapa) + '.csv'
    df.to_csv(nome, sep=',', index=False)

    #valores previstos
    df = pd.DataFrame(valores_previstos)
    nome = 'valores_previstos_' + str(fold_etapa) + '.csv'
    df.to_csv(nome, sep=',', index=False)

    #pesos dos perceptrons
    df = rede_percep.pesos_perceptrons()
    nome = 'pesos_perceptrons_' + str(fold_etapa) + '.csv'
    df.to_csv(nome, sep=',', index=False)

#salvando a performance de cada fold de teste
df_erros.to_csv("performance_etapas.csv", sep=',', index=False)
