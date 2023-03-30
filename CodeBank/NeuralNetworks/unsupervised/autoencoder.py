#Autoencoder
#   Idea:
#       Una red que aprenda a autogenerar la entrada mediante pocos parametros
#Variational Autoencoder
#   https://www.youtube.com/watch?v=rZufA635dq4
#   Idea: 
#       Representar la entrada en parametros de una distribucion
#       Generar una variable aleatoria
#       Reproducir la entrada en la salida
#   Obliga a 
#GAN:
#   Encoder 1:
#       Input: Data
#       Ouput: Generated Data
#   Encoder 2:
#       Input: Noise
#       Output: Data
#   Clasificador 1:
#       Input: generated data,  Original data
#       Output: True or fake

#Encoder:
#   P(X|Z)
#Decoder:
#   P(Z|X)

#Polinomio de taylor para probabilidad?
#   Sin parametro = uniforme
#   1 Parametro = Exponencial
#   2 parametro = gausiana
#   x parametro = ??