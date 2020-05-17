import csv
import rede as rede
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
import seaborn as sn
from sklearn.metrics import confusion_matrix

#Carregamento de dados de treino e teste
treino = pd.read_csv('Conjuntos MNIST/mnist_treinamento.csv')
teste = pd.read_csv('Conjuntos MNIST/mnist_teste.csv')


#Pré-pocessamento de dados 
gabarito_treino = treino.iloc[:, 0]
amostras_treino = treino.iloc[:, 1:len(treino)-1].div(255)
gabarito_teste = teste.iloc[:, 0]
amostras_teste = teste.iloc[:, 1:len(treino)-1].div(255)


#criação da rede de perceptrons
rede = rede.Rede(n_classes = 10,
                saidas = gabarito_treino,
                amostras = amostras_treino,
                taxa_aprendizado = 0.1, 
                epocas = 50, 
                bias = 0.1)

#treino da rede de perceptrons
rede.treina()
#rede.carrega_pesos_salvos(pd.read_csv('pesos_perceptrons.csv'))

#avaliando performance do treino
erros_treino = rede.erros
#erros_treino = pd.read_csv('erros_absolutos_epoca.csv').iloc[:, 0].tolist()

plt.figure()
plt.plot(erros_treino)
plt.ylabel("Número de erros")
plt.xlabel("Época")

acc = []
for i in range(len(erros_treino)) :
    acc.append(1 - erros_treino[i]/len(gabarito_treino))

plt.figure()
plt.plot(acc)
plt.ylabel("Acurácia")
plt.xlabel("Época")

#aplicação da rede no conjunto de teste
valores_previstos = []

print(">> Testando rede no conjunto de teste")
for i in tqdm(range(len(amostras_teste))) :
    valores_previstos.append(rede.resposta_rede(amostras_teste.iloc[i, :].tolist()))


#número de erros e acertos sobre o conjunto de teste
acertos = 0
for i in range(len(gabarito_teste)) :
    if valores_previstos[i] == gabarito_teste[i] :
        acertos = acertos + 1
erros = len(gabarito_teste) - acertos

print(">> Contagem de erros e acertos no teste")
print("Acertos: " + str(acertos))
print("Erros: " + str(erros))

#matriz de confusão
plt.figure()
matiz_confusao = confusion_matrix(gabarito_teste, valores_previstos)
df_cm = pd.DataFrame(matiz_confusao, range(10), range(10))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu", annot_kws={"size": 9}, fmt='g')

plt.show()

#exportando os resultados para não ter que executar o código toda hora
#erro absoluto do treino em cada época
df = pd.DataFrame(rede.erros)
df.to_csv('erros_absolutos_epoca.csv', sep=',', index=False)

#valores previstos
df = pd.DataFrame(valores_previstos)
df.to_csv('valores_previstos.csv', sep=',', index=False)

#pesos dos perceptrons
df = rede.pesos_perceptrons()
df.to_csv('pesos_perceptrons.csv', sep=',', index=False)
