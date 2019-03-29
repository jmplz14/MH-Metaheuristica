#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:40:29 2019

@author: jose
"""
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import sys

colopscopy = "datos/colposcopy.arff"
ionosphere = "datos/ionosphere.arff"
texture = "datos/texture.arff"


def leerFicheroCSV(fichero):
    
    datos = np.genfromtxt(fichero, delimiter=',')
    return datos
"""
Separamos la parte de los datos la columna de clases y normalizamos los datos
a continuacion los univmos y devolvemos el dataframe de pand normalizado
"""
#Carga los datos y los normaliza
def leerFicheroARFF(fichero, num_particiones):
    #Cargo los ficheros
    data = arff.loadarff(fichero)
    df = pd.DataFrame(data[0])
    #me quedo con la clases
    clase =  df.values[: , -1]
    clase = clase.reshape((1,clase.size))
    
    #elimino la clase para normalizar la matriz
    datos = df.set_index("class", drop = True)
    #normalizo
    min_max_scaler = preprocessing.MinMaxScaler()
    datos = min_max_scaler.fit_transform(datos)
    
    #Concateno la matriz para tenerla lista con las clases y los datos
    matriz_final = np.concatenate((datos,clase.T),axis=1)
    
    
    #distribucion_clases = df.groupby('class').count()
    tipos_clases = np.unique(matriz_final[: , -1])
    #print(distribucion_clases)
    num_filas = int(matriz_final.size / matriz_final[0].size)
    #print(num_filas)
    np.random.shuffle(matriz_final)
    
    num_entradas = int(num_filas / num_particiones)
    resto = num_entradas % num_particiones
    array_particiones = np.empty(num_particiones, np.ndarray)
    
    for i in range(1, 1 + num_particiones):
        #print(i)
        if resto > 0:
            array_particiones[-i] = np.empty((1, matriz_final[0].size))
            resto = resto-1
        else:
            array_particiones[-i] = np.empty((1, matriz_final[0].size))
            
    pos_particion = 0
    i_particiones = np.zeros(num_particiones, np.int)
    for i_clase in tipos_clases:
        for j in range(0,int(matriz_final.size /matriz_final[0].size)):
            if matriz_final[j][-1] == i_clase:
                #print(matriz_final[j][-1])
                """array_particiones[pos_particion][i_particiones[pos_particion]] = matriz_final[j]"""
                """if i_particiones[pos_particion] == 0:
                    array_particiones[pos_particion] = matriz_final[j]
                    i_particiones[pos_particion] += 1
                else:"""
                array_particiones[pos_particion] = \
                np.append(array_particiones[pos_particion], matriz_final[j].reshape(1,matriz_final[j].size), axis = 0)
                
                pos_particion = (pos_particion + 1) % num_particiones
                i_particiones[pos_particion] += 1
    
    for i in range(num_particiones): 
        array_particiones[i] = np.delete(array_particiones[i], 0, 0)
   # print(array_particiones[4])
    
    """b = 0
    g = 0
    print(array_particiones[0].size / array_particiones[0][0].size)
    print(array_particiones[1].size / array_particiones[1][0].size)
    print(array_particiones[2].size / array_particiones[2][0].size)
    print(array_particiones[3].size / array_particiones[3][0].size)
    print(array_particiones[4].size / array_particiones[4][0].size)
    
    for i in range(0,70):
        if array_particiones[1][i][-1] == tipos_clases[0]:
            b += 1
        else:
            g += 1
    print(b)num_particones
    print(g)
    for i in range(num_particiones):
        print(array_particiones[i])"""
        
            
    return array_particiones
    

    
    
def matrizDistancia(datos):
    matriz_distancia = euclidean_distances(datos, datos)
    return matriz_distancia

    
    
def RELIEF(training,test):
    num_valores = training[0].size - 1
    num_elementos = np.size(training,0)
    w = np.zeros(num_valores)
    distancias = matrizDistancia(training[:, 0:-1])
    
    for i in range(num_elementos):
        mejor_enemigo = int()
        valor_enemigo = sys.float_info.max
        mejor_amigo = int()
        valor_amigo = sys.float_info.max
        
        for j in range(num_elementos):
            if training[i][-1] == training[j][-1]:
                if valor_amigo > distancias[i][j]:
                    if i != j:
                        valor_amigo = distancias[i][j]
                        mejor_amigo = j
                    
            else:
                if valor_enemigo > distancias[i][j]:
                    valor_enemigo = distancias[i][j]
                    mejor_enemigo = j
                
        w = w + np.absolute(training[i, 0:-1]-training[mejor_enemigo, 0:-1]) -\
        np.absolute(training[i, 0:-1]-training[mejor_amigo, 0:-1]) 
    
    w_maximo = np.amax(w)
    for i in range(num_valores):
        if w[i] < 0:
            w[i] = 0.0
        else:
            w[i] = w[i] / w_maximo 
    return w       
    
    
    
    
def main():
    num_particiones = 5
    particiones = leerFicheroARFF(colopscopy, num_particiones)
    
    training = np.ndarray
    test = np.ndarray
    for i in range(num_particiones):
        if i == 0:
            training = particiones[1]
            #print("training", 1)
            test = particiones[0]
            #print("test",i)
            for j in range(2,num_particiones):
                #print("training", j)
                training = np.concatenate((training,particiones[j]), axis = 0)
                
        else:
            #print("training", 0)
            training = particiones[0]
            for j in range(1,num_particiones):
                if j == i:
                    #print("test",i)
                    test = particiones[j]
                else:
                    #print("training", j)
                    training = np.concatenate((training,particiones[j]), axis = 0)
            
        pesos = RELIEF(training,test)
    
    

    






if __name__== "__main__":
  main()