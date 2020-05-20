# ep-IA
Este é o repositório para o primeiro trabalho da disciplina de IA. Nele é desenvolvido uma rede de única camada de perceptrons para o reconhecimento de dígitos do MNIST dataset.

## Depências
- pandas
- numpy
- csv
- tqdm
- matplotlib
- seaborn
- sklearn

#### Importante:
O sklearn é apenas utilizado para fazer a matriz de confusão, **__o modelo em si não utiliza essa biblioteca__.**

## Execução
O modelo pode ser executado pelo console usando o comando estando no diretório ep-IA:
```
py -3 modelo\main.py
```
E caso queira executar o mesmo usando o 10-fold cross-validation use o seguinte comando, também estando no diretório ep-IA:
```
py -3 modelo\main_cross_validation.py
```
#### Importante:
Coloque os dados para treino e teste no diretório 'modelo'.
