#Goodfellow
# Reservoir computing
# Echo state machine

#En vez que la red recurrente genere un estado por cada instante de tiempo

#   Generar una bolsa de estados ocultos
#   Y cada instante, x recupera un conjunto de h
#       y luego extraigo de h la salida.



#TIPO CADENA DE MARKOV (Pero con nolinearidades)


#El reservorio, puede ser cualquier modelo (hopfield, recurrentes, hidden markov model)
#
# La recurrencia tiene que tener autovalores menores a uno para que converja.
#
# Ver Modelo de Ricardo. Filtro FIR e IIR 
#   Modelo = Red neuronal, la salida se realimenta a la entrada o no lo hace.
#   Los delays se grafican con z^-1


"""
Sergio:
Memoria de corto plazo
    -La memoria previa está, pero no infinitamente en el tiempo.
    -En el cerebro, se puede utilizar ciclos cortos para medir el tiempo
    -Ver Rodrigo Laje
"""
# Entrar
#   http://www.scholarpedia.org/article/Echo_state_network

"""
framework: https://github.com/nschaetti/EchoTorch

"""

"""
Ejemplo control
  ht+1 = A ht + B xt
Las matrices A, B son aleatorias. No se aprenden

La matriz  A representa los ciclos predeterminados de recurrencia, 
    De esta manera, no aprendemos A (gradiente evanescente)
    Entonces la salida aprende a observar qué recurrencias utilizar
    según la entrada, Y cómo traducir las recurrencias en el tiempo.
"""

"""
idea:
  -Aprender primero la salida
  -Aprender luego el reservorio
"""


#EJERCICIO DE RECURRENCIA DE TANH
# import numpy as np
# import matplotlib.pyplot as plt

# # h = 2*np.random.uniform(size=[100,1])-1
# h = np.array(range(1,100))/50-1 #h va de -1 a 1 crece lineal
# v = np.random.normal(size=[100,100])
# w = np.eye(100,100)
# # w = np.random.normal(size=[100,100])

# out = np.matmul(v,h)
# plt.plot(out)
# ones = np.ones(())
# r = np.arange(100).reshape(100,1)
# aux = np.concatenate((out, r))
# print(aux)

# aux = np.sortrows(aux,1)
# out = np.matmul(v,h)

# plt.plot(out)
# h = np.tanh(w*h)
# plt.show()