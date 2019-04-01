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
from scipy.spatial import distance,KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from time import time


colopscopy = "datos/colposcopy.arff"
ionosphere = "datos/ionosphere.arff"
texture = "datos/texture.arff"
min_peso = 0.2
num_particiones = 5
max_vecinos = 15000
sigma = 0.3


def funcionObjetivo(clase,reduccion):
    return clase * 0.5 + reduccion * 0.5
    
def RELIEF(training, test):
    num_valores = training[0].size - 1
    train_datos = training[: , 0:-1];
    num_elementos = np.size(training,0)
    
    start_time = time()
    w = np.zeros(num_valores)
    distancias = euclidean_distances(train_datos, train_datos)
    #print(distancias)
    
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

        w = w + np.absolute(train_datos[i]-train_datos[mejor_enemigo]) -\
        np.absolute(train_datos[i]-train_datos[mejor_amigo])
    w_maximo = np.amax(w)
    

    for i in range(num_valores):
        if w[i] < 0:
            w[i] = 0.0
        else:
            w[i] = w[i] / w_maximo
    
    train_datos = (train_datos * w)[: , w >= min_peso]
    test_datos = test[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = (test_datos * w)[: , w >= min_peso]
    test_clases = np.array(test[:, -1], int)
    num_aciertos = 0
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if (tipo == test_clases[i]):
            num_aciertos += 1
    
    datos_medidos = np.zeros(4)
    
    datos_medidos[3] = time() - start_time  
    datos_medidos[0] = 100 * (num_aciertos / num_muestras)
    datos_medidos[1] = 100 * (w[w < min_peso].size / w.size)
    datos_medidos[2] = funcionObjetivo(datos_medidos[0], datos_medidos[1])
    
    return datos_medidos
    
def k_nn(training,test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    start_time = time()
    num_aciertos = 0;
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if( tipo == test_clases[i]):
            num_aciertos += 1
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tiempo = time() - start_time  
    funcion = funcionObjetivo(tasa_acierto,0)
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_acierto
    datos_algoritmo[2] = funcion
    datos_algoritmo[3] = tiempo
    
    return datos_algoritmo

def uno_nn(train_datos, train_clases, test_datos, test_clases, w):

    train_datos = (train_datos * w)[: , w >= min_peso]
    test_datos = (test_datos * w)[: , w >= min_peso]   

    
    num_aciertos = 0;
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if( tipo == test_clases[i]):
            num_aciertos += 1
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tasa_reduccion = 100 * (w[w < min_peso].size / w.size)
    return tasa_acierto,tasa_reduccion

def BL(training,test):
    
    num_vecinos = 0
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    num_valores = train_datos[0].size
    w = np.random.rand(num_valores)
    pos_w = 0
    sin_mejora = 0
    max_sin_mejora = 20 * num_valores
    start_time = time()
    mejor_valor_w = sys.float_info.min
    mejor_w = w
    mejor_tasa_clase = 0
    mejor_tasa_reduccion = 0
    num_calculos = 0
    
    while num_vecinos < max_vecinos and sin_mejora < max_sin_mejora:
        num_vecinos += 1
        cambio = np.random.normal(0.0, sigma, None)
        w_anterior = w[pos_w]
        w[pos_w] += cambio
        if w[pos_w] > 1:
            w[pos_w] = 1
        if w[pos_w] < 0:
            w[pos_w] = 0
            
        ##probar a no ejecutar si w no cambia
        if (w_anterior < min_peso and w[pos_w] < min_peso) or (w_anterior == w[pos_w]):
            
             w[pos_w] = w_anterior
             sin_mejora += 1
        else:
            tasa_clase, tasa_reduccion = uno_nn(train_datos, train_clases, test_datos, test_clases, w)
            
            funcion_mejora = 0.5 * tasa_clase + 0.5 * tasa_reduccion

            if mejor_valor_w < funcion_mejora:
                
                mejor_w = w
                mejor_valor_w = funcion_mejora
                mejor_tasa_clase = tasa_clase
                mejor_tasa_reduccion = tasa_reduccion
                sin_mejora = 0

            else:
                w[pos_w] = w_anterior
                sin_mejora += 1
         
             
       
        
        pos_w = (pos_w + 1) % num_valores
        
    tiempo = time() - start_time 
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = mejor_tasa_clase
    datos_algoritmo[1] = mejor_tasa_reduccion
    datos_algoritmo[2] = tiempo
    datos_algoritmo[3] = mejor_valor_w * 100
       
    
    return datos_algoritmo
        
        
    
    
    
    
    
    
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
    clase = df.values[: , -1]
    clase_test =  np.array(df.values[: , -1], int)


    clase = clase.reshape((1,clase.size))
    #elimino la clase para normalizar la matriz
    datos = df.set_index("class", drop = True)

    
    #normalizo
    min_max_scaler = preprocessing.MinMaxScaler()
    datos = min_max_scaler.fit_transform(datos)
    
    X = np.ones(10)
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    skf = StratifiedKFold(n_splits=5)
    for train, test in skf.split(datos, clase_test):
        print("--------------------------------------------------")
        print(train)
    
    #Concateno la matriz para tenerla lista con las clases y los datos
    matriz_final = np.concatenate((datos,clase.T),axis=1)
    distribucion_clases = df.groupby('class').count()
    #print(distribucion_clases)
    tipos_clases = np.unique(matriz_final[: , -1])
    num_filas = int(matriz_final.size / matriz_final[0].size)
    #print(num_filas)
    #np.random.shuffle(matriz_final)
    
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
    contador = np.zeros(11)

    print(array_particiones[0].size / array_particiones[0][0].size)
    print(array_particiones[1].size / array_particiones[1][0].size)
    print(array_particiones[2].size / array_particiones[2][0].size)
    print(array_particiones[3].size / array_particiones[3][0].size)
    print(array_particiones[4].size / array_particiones[4][0].size)
    
    for i in range(0,110):
        for j in range(contador.size):
            if array_particiones[4][i][-1] == tipos_clases[j]:
                contador[j] += 1
               
    print(contador)
    
        
    return array_particiones

def leerFicheroARFF2(fichero, num_particiones):
    #Cargo los ficheros
    data = arff.loadarff(fichero)
    df = pd.DataFrame(data[0])
    #print(df)
    #print(RELIEF(df.values))
    #me quedo con la clases
    clase = df.values[: , -1]
    clase_test =  np.array(df.values[: , -1], int)


    clase = clase.reshape((1,clase.size))
    #elimino la clase para normalizar la matriz
    datos = df.set_index("class", drop = True)

    
    #normalizo
    min_max_scaler = preprocessing.MinMaxScaler()
    datos = min_max_scaler.fit_transform(datos)
    
    X = np.ones(10)
    y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    skf = StratifiedKFold(n_splits=5)
    #print(clase_test)
    for train, test in skf.split(datos, clase_test):
        print("--------------------------------------------------")
        print(train)
    
    #Concateno la matriz para tenerla lista con las clases y los datos
    matriz_final = np.concatenate((datos,clase.T),axis=1)
    distribucion_clases = df.groupby('class').count()
    #print(distribucion_clases)
    tipos_clases = np.unique(matriz_final[: , -1])
    num_filas = int(matriz_final.size / matriz_final[0].size)
    #print(num_filas)
    #np.random.shuffle(matriz_final)
    
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
   
    
        
    return array_particiones
  
def tasaReduccion(pesos):
    num_valores_descartados = 0
    for i in pesos:
        if i < min_peso:
            num_valores_descartados += 1
    return num_valores_descartados / pesos.size

def tasaClase(distancias, clases):
    num_elementos = np.size(distancias,0)
    num_aciertos = 0

    for i in range(num_elementos):
        valor_mejor_vecino = sys.float_info.max
        mejor_vecino = int() 
        for j in range(num_elementos):
            if valor_mejor_vecino > distancias[i][j]:
                if i != j:
                    valor_mejor_vecino = distancias[i][j]
                    mejor_vecino = j
        
        if clases[i] == clases[mejor_vecino]:
            num_aciertos += 1
                
    return num_aciertos / num_elementos 
            
    
def matrizDistancia(datos):
    matriz_distancia = euclidean_distances(datos, datos)
    return matriz_distancia

def clasificador(pesos, train, test):
    datos_train = train[:, 0:-1]
    clases_train = train[: , -1]
    datos_test = test[:, 0:-1]
    clases_test = test[: , -1]
    pesos_sin_min = np.array(pesos)
    
    """for i in range(pesos_sin_min.size):
        if pesos_sin_min[i] < min_peso:
            pesos_sin_min[i] = 0
            
    for i in (np.size(datos_test,0)):
        distancia_min = euclidean_distances(datos_test[i], datos_train[0])
        elemento_min = 0"""
	
    """n_neighbors = 1
     
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(datos_train, clases_train)
    print(knn)
    print('Accuracy of K-NN classifier on training set: {:.2f}',knn.score(datos_train, clases_train))
    print('Accuracy of K-NN classifier on test set: {:.2f}', knn.score(datos_test, clases_test))"""
    num_elementos = np.size(datos_test,0)
    matriz_distancias = np.zeros((num_elementos,num_elementos))
    inicio_j = 1
    tasa_reduccion = tasaReduccion(pesos)
    pesos_sin_min = np.array(pesos)
    
    """for i in range(pesos_sin_min.size):
        if pesos_sin_min[i] < min_peso:
            pesos_sin_min[i] = 0"""
    
        
    datos_prueba = (datos_test * pesos)[:, pesos > 0.2]
    kdtree = KDTree(datos_prueba)
    #matriz_distancias = euclidean_distances(datos,datos,pesos_sin_min)
    neighbours = kdtree.query(datos_prueba, k=2)[1][:, 1]
    accuracy = np.mean(clases_test[neighbours] == clases_test)
    #accuracy = np.mean(y[neighbours] == y)
    for i in range(num_elementos):
        for j in range(inicio_j,num_elementos):
            #print(i,j)
            
            #resta = datos[i] - datos[j]
            #distancia = np.sum(pesos_sin_min * (resta * resta))
 
            distancia = distance.euclidean(datos[i], datos[j], pesos_sin_min)
            matriz_distancias[i][j] = distancia
            matriz_distancias[j][i] = distancia
        

        inicio_j += 1    
    tasa_clase = tasaClase(matriz_distancias,clases)
    
    #print("-------------------")
    #print(matriz_distancias)
    #return tasa_clase, tasa_reduccion
    #KNeighborsClassifier(metric='wminkowski', p=2, metric_params={'w': pesos})
    
     
def clasificador2(X_test, y, pesos):

    X_transformed = (X_test * pesos)[:, pesos > 0.2]
    kdtree = KDTree(X_transformed)
    neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
    accuracy = np.mean(y[neighbours] == y)
    reduction = np.mean(pesos < 0.2)
    

    
def dibujarTabla(datos):
    
    for i in range(np.size(datos, 0)-1):
        datos[-1] = datos[-1] + datos[i]
    
    datos[-1] =  datos[-1] / num_particiones
    index = np.array(["1", "2", "3", "4", "5", "Media"])

    
   
    columns = np.array(["Tasa Clase", "Tasa Reduccion", "Función Objetivo", "Tiempo" ])
    
    
    
    
    tabla = pd.DataFrame(datos, index ,columns)  
    
    print(tabla)
    
    
def main():
    for i in range(3):
    
        tasa_clase_media = 0
        tasa_reduccion_media = 0
        
        #Cargo los ficheros
        datos_RELIEF = np.zeros((6,4), np.float)
        datos_BL = np.zeros((6,4))
        datos_1NN = np.zeros((6,4))
        print("------------------------")
        if i == 0:
            data = arff.loadarff(ionosphere)
            print("MEDIDAS IONOSPHERE")
            
        elif i == 1: 
            data = arff.loadarff(colopscopy)
            print("MEDIDAS COLOPSCOPY")
        else:
            data = arff.loadarff(texture)
            print("MEDIDAS TEXTURE")
            
        df = pd.DataFrame(data[0])
        distribucion_clases = df.groupby('class').count()
        clase = df.values[: , -1]
        
        clase_test =  np.array(df.values[: , -1], int)
      
       
        
    
    
        clase = clase.reshape((1,clase.size))
        #elimino la clase para normalizar la matriz
        datos = df.set_index("class", drop = True)
    
        
        #normalizo
        min_max_scaler = preprocessing.MinMaxScaler()
        datos = min_max_scaler.fit_transform(datos)
        matriz_final = np.concatenate((datos,clase.T),axis=1)
        #np.random.shuffle(matriz_final)
        datos = matriz_final[:, 0:-1]
        clase = matriz_final[: , -1]
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        #print(clase_test)
        r_tasa_clase = 0
        r_tasa_reduccion = 0
        knn_tasa_clase = 0;
        i_particion = 0
        for train, test in skf.split(datos, clase_test):  
            train_datos = matriz_final[train,:]
            test_datos = matriz_final[test,:]
            
            datos_RELIEF[i_particion] = RELIEF(train_datos,test_datos)

            datos_1NN[i_particion] = k_nn(train_datos, test_datos)
            
            datos_BL[i_particion] = BL(train_datos, test_datos)
            
            i_particion += 1
            
        
        #print("Media KNN: ", knn_tasa_clase / num_particiones)
        print("\nDatos RELIEF\n")
        dibujarTabla(datos_RELIEF)
        
        print("\nDatos 1-NN\n")
        dibujarTabla(datos_1NN)
        
        print("\nDatos BL\n")
        dibujarTabla(datos_BL)
        
        
    

    






if __name__== "__main__":
  main()