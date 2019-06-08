#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:40:29 2019

@author: José Manuel Pérez Lendínez
"""

import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.neighbors import KDTree
semilla = 1
np.random.seed(semilla)
#rutas de los ficheros
colopscopy = "datos/colposcopy.arff"
ionosphere = "datos/ionosphere.arff"
texture = "datos/texture.arff"

#Peso apartir el cual no se tendra en cueta la caracteristica
min_peso = 0.2

#numero de particiones para los datos
num_particiones = 5

#max_veinos que obtendremos
max_vecinos = 15000

#valor para la distribucion de la mutacion
sigma = 0.3

tam_poblacion = 30
tam_poblacion_memeticos = 10
num_evaluaciones = 15000
porcentaje_cruce = 0.7
porcentaje_mutacion = 0.001
#Calcula la funcion objetivo
def funcionObjetivo(clase,reduccion):
    return clase * 0.5 + reduccion * 0.5

#algoritmos de relief

"""
Utilzado para el 1-NN donde no tenemos vector de pesos
"""



def evaluate(weights, X, y):
    X_transformed = (X * weights)[:, weights > 0.2]
    if X_transformed.size > 0:
        kdtree = KDTree(X_transformed)
        
        neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
        accuracy = np.mean(y[neighbours] == y)
        reduction = np.mean(weights < 0.2)
        return 100*accuracy,reduction*100,100*(accuracy + reduction) / 2
    else:
        return 0,0,0
    
def obtenerFitness(weights, X, y):
    X_transformed = (X * weights)[:, weights > 0.2]
    if X_transformed.size > 0:
        kdtree = KDTree(X_transformed)
        
        neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
        accuracy = np.mean(y[neighbours] == y)
        reduction = np.mean(weights < 0.2)
        return (accuracy + reduction) / 2
    else:
        return 0

def uno_nn(train_datos, train_clases, test_datos, test_clases, w):
    
    
    #nos quedamos con los datos que cumplen el minimo para los pesos
    train_datos = (train_datos * w)[: , w >= min_peso]
    test_datos = (test_datos * w)[: , w >= min_peso]   

    num_aciertos = 0;
    
    #entrenamos con el train 
    clasificador = KNeighborsClassifier(n_neighbors=1)
    clasificador.fit(train_datos, train_clases)
    num_muestras = np.size(test_datos,0)
    
    #clasificamos con el test
    for i in range(num_muestras):
        tipo = clasificador.predict([test_datos[i]])
        if( tipo == test_clases[i]):
            num_aciertos += 1
            
    tasa_acierto = 100 * (num_aciertos / num_muestras)
    tasa_reduccion = 100 * (w[w < min_peso].size / w.size)
    funcion_objetivo= (tasa_acierto + tasa_reduccion) / 2
    
    return tasa_acierto,tasa_reduccion,funcion_objetivo




def BLILS(train_datos, train_clases, w, valor):
    
    
    
    #Contador de vecionos creados
    num_vecinos = 0
    
    num_valores = train_datos[0].size
    
    #inciamos el vector con valores random


    
    #marcara la posicion a mutar paa el vector
    pos_w = 0
    
    
    num_vecinos_parada = 1000
    
    
    #iniciamos los valores para el primer vector 
    #tasa_clase, tasa_reduccion = uno_nn(train_datos, train_clases, test_datos, test_clases, w)
    #tasa_clase, tasa_reduccion, funcion_mejora = evaluate(w, train_datos, train_clases)
            
    #funcion_mejora = 0.5 * tasa_clase + 0.5 * tasa_reduccion
    mejor_valor_w = valor


    
    while num_vecinos < num_vecinos_parada:
        
        #contamos un nuevo vecion
        num_vecinos += 1
        
        #generamos la mutacion y guardamos el anterior por si no mejora
        cambio = np.random.normal(0.0, sigma, None)
        w_anterior = w[pos_w]
        
        #miramos que no pasa ni 0 ni 1
        w[pos_w] += cambio
        if w[pos_w] > 1:
            w[pos_w] = 1
        if w[pos_w] < 0:
            w[pos_w] = 0
            
        """
        Si la mutacio da un peso menor al minimo y el anterior tambien era menor
        o la mutacion deja los pesos como estaban. No podra mejorar nunca los datos
        de entrenamiento y no hace falta hacer los calculos. Se cuenta uno sin mejora.
        """
        if (w_anterior < min_peso and w[pos_w] < min_peso) or (w_anterior == w[pos_w]):
             w[pos_w] = w_anterior
        #en otro caso si hay que hacer los calculos de la funcion de mejora
        else:
            
            #realizamos los calculos para ver si mejora
            #tasa_clase, tasa_reduccion = uno_nn(train_datos, train_clases, test_datos, test_clases, w)      
            tasa_clase, tasa_reduccion, funcion_mejora = evaluate(w, train_datos, train_clases)
            #funcion_mejora = 0.5 * tasa_clase + 0.5 * tasa_reduccion
            
            #si mejora almacenamos esos valores
            if mejor_valor_w < funcion_mejora:
                mejor_valor_w = funcion_mejora

                
            #si no mejora se descarta y contamos unos sin mejora
            else:
                w[pos_w] = w_anterior

         
             
       
        #obtenemos la siguiente posicion a mutar
        pos_w = (pos_w + 1) % num_valores
    

    return w, mejor_valor_w
#funcion para imprimir los resultados
def dibujarTabla(datos):
    for i in range(np.size(datos, 0)-1):
        datos[-1] = datos[-1] + datos[i]
    
    datos[-1] =  datos[-1] / num_particiones
    index = np.array(["1", "2", "3", "4", "5", "Media"])

    
   
    columns = np.array(["Tasa Clase", "Tasa Reduccion", "Función Objetivo", "Tiempo" ])
    
    
    
    
    tabla = pd.DataFrame(datos, index ,columns)  
    
    print(tabla)

def inicializarPoblacion(num_individuos,num_genes):
    #poblacion = np.zeros((num_individuos,num_genes))
    #for i in range(num_individuos):
    return np.random.uniform(0,1,(num_individuos, num_genes))
    

    
    

def evaluarPoblacion(X, y, poblacion):
    num_individuos = np.size(poblacion,0)
    valores = np.zeros(num_individuos)
    for i in range(num_individuos):
        tasa_clase, tasa_reduccion, funcion_objetivo = evaluate(poblacion[i], X, y)
        valores[i] = funcion_objetivo

    return valores
        
    



            
def mutarGen(pesos, posicion):
    nuevos = np.copy(pesos)
    mutacion = np.random.normal(0.0, sigma, None)
    nuevos[posicion] += mutacion
    
    if nuevos[posicion] > 1:
        nuevos[posicion] = 1
        
    if nuevos[posicion] < 0:
        nuevos[posicion] = 0
    
    return nuevos

def mutarGenILS(pesos, posicion):
    nuevos = np.copy(pesos)
    mutacion = np.random.normal(0.0, 0.4, None)
    nuevos[posicion] += mutacion
    
    if nuevos[posicion] > 1:
        nuevos[posicion] = 1
        
    if nuevos[posicion] < 0:
        nuevos[posicion] = 0
    
    return nuevos




def ES(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    start_time = time()
    num_genes = np.size(train_datos,1)
    
    #inicializo y evaluo la solucion
    solucion = np.random.uniform(0,1,(num_genes))   
    #solucion = np.random.rand(train_datos.shape[1])
    valor = obtenerFitness(solucion, train_datos, train_clases)
    mejor_solucion = solucion
    mejor_eval = valor
 
    
    #mido su temperatura y la guardo como mejor solucion    
    tem_inicial = 0.3 * valor/ (-np.log(0.3))
    temperatura = tem_inicial
    tem_final = np.clip(1e-3,0,tem_inicial)
    
    
    
    max_vecinos = 10 * num_genes
    max_exitos = num_genes
    
    max_iter = 15000
    M = max_iter / max_vecinos
    evaluaciones = 0
    num_exitos = 1
    
    
    while evaluaciones < max_iter and temperatura > tem_final and num_exitos > 0:
        num_vecinos = 0
        num_exitos = 0
        while num_vecinos < max_vecinos and num_exitos < max_exitos:
            
            pos_gen = np.random.randint(0,num_genes)
            vecino = mutarGen(solucion, pos_gen)
            valor_vecino = obtenerFitness(vecino, train_datos, train_clases)
            num_vecinos += 1
            
            diff = valor_vecino - valor 
            probabilidad = np.exp(diff/temperatura)
            if diff > 0 or np.random.random() < probabilidad:
                solucion = vecino
                valor = valor_vecino
                num_exitos += 1
                if valor > mejor_eval:
                    mejor_eval = valor
                    mejor_solucion = solucion
                    
        evaluaciones += num_vecinos 
        beta = (tem_inicial - tem_final)/( M * tem_inicial * tem_final)
        temperatura = temperatura / (1 + beta * temperatura)
      
    tiempo = time() - start_time 
    tasa_clase, tasa_reduccion, funcion_objetivo = uno_nn(train_datos,train_clases,test_datos,test_clases,mejor_solucion)
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo   
        

def ILS(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    start_time = time()
    num_genes = np.size(train_datos,1)
    
    #inicializo y evaluo la solucion
    solucion = np.random.uniform(0,1,(num_genes))   
    eval_solucion = obtenerFitness(solucion,train_datos,train_clases)
    
    solucion, eval_solucion = BLILS(train_datos, train_clases, solucion, eval_solucion)
    total_mutaciones = int(0.1 * num_genes)
    
    
    
    for i in range(0,14):
        nueva_solucion = np.copy(solucion)
        #pos_genes = np.random.randint(0,num_genes,size = total_mutaciones)
        pos_genes = np.random.permutation(num_genes)[0:total_mutaciones]
        for i in range(0,total_mutaciones):
            
            nueva_solucion = mutarGenILS(nueva_solucion, pos_genes[i])
            
        eval_nueva_solucion = obtenerFitness(nueva_solucion,train_datos,train_clases)
        nueva_solucion, eval_nueva_solucion = BLILS(train_datos, train_clases, nueva_solucion, eval_nueva_solucion)
        
        if eval_nueva_solucion > eval_solucion:
            solucion = nueva_solucion
            eval_solucion = eval_nueva_solucion
    
    tiempo = time() - start_time 
    tasa_clase, tasa_reduccion, funcion_objetivo = uno_nn(train_datos,train_clases,test_datos,test_clases,solucion)
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo
    
    
def EDRand(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    start_time = time()
    num_genes = np.size(train_datos,1)
    poblacion = inicializarPoblacion(50,num_genes)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion)
    
    pos_mejor = np.argmax(eval_poblacion)
    mejor = poblacion[pos_mejor]
    eval_mejor = eval_poblacion[pos_mejor]
    
    num_evaluaciones = 0

    while num_evaluaciones < 15000:
        for i in range(0,50):
            indices = np.random.permutation(50)[0:4]
            pos_i = np.where(indices == i)

            if np.size(pos_i) == 1:
                indices = np.delete(indices, pos_i)

            p1 = poblacion[indices[0]]
            p2 = poblacion[indices[1]]
            p3 = poblacion[indices[2]]
            mutado = np.zeros(num_genes)
            #print(mutado)
            for j in range(0,num_genes):
                if np.random.rand() < 0.5:
                   mutado[j] = p1[j] + 0.5 * (p2[j] - p3[j])
                   mutado[j] = np.clip(mutado[j], 0, 1)
                else:
                   mutado[j] = poblacion[i][j]
                    
                    
            eval_mutado = 100 * obtenerFitness(mutado,train_datos,train_clases);
            num_evaluaciones += 1
            if eval_mutado > eval_poblacion[i]:
                poblacion[i] = mutado
                eval_poblacion[i] = eval_mutado
                
                if eval_mutado > eval_mejor:
                    mejor = mutado
                    eval_mejor = eval_mutado
    
            
    tiempo = time() - start_time 
    tasa_clase, tasa_reduccion, funcion_objetivo = uno_nn(train_datos,train_clases,test_datos,test_clases,mejor)
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo 
    
def EDCurrentToBest(training, test):
    train_datos = training[:, 0:-1]
    train_clases = np.array(training[:, -1], int)
    test_datos = test[:, 0:-1]
    test_clases = np.array(test[:, -1], int)
    
    start_time = time()
    num_genes = np.size(train_datos,1)
    poblacion = inicializarPoblacion(50,num_genes)
    eval_poblacion = evaluarPoblacion(train_datos, train_clases, poblacion)
    
    pos_mejor = np.argmax(eval_poblacion)
    mejor = poblacion[pos_mejor]
    eval_mejor = eval_poblacion[pos_mejor]
    
    num_evaluaciones = 0
    while num_evaluaciones < 15000:
        for i in range(0,50):
            indices = np.random.permutation(50)[0:3]
            pos_i = np.where(indices == i)

            if np.size(pos_i) == 1:
                indices = np.delete(indices, pos_i)

            p1 = poblacion[indices[0]]
            p2 = poblacion[indices[1]]
            
            mutado = np.zeros(num_genes)
            #print(mutado)
            for j in range(0,num_genes):
                if np.random.rand() < 0.5:
                   mutado[j] = poblacion[i][j] + 0.5 * (mejor[j] - poblacion[i][j]) + 0.5 * (p1[j] - p2[j])
                   mutado[j] = np.clip(mutado[j], 0, 1)
                else:
                   mutado[j] = poblacion[i][j]
                    
                    
            eval_mutado = 100 * obtenerFitness(mutado,train_datos,train_clases);
            num_evaluaciones += 1
            if eval_mutado > eval_poblacion[i]:
                poblacion[i] = mutado
                eval_poblacion[i] = eval_mutado
                
                if eval_mutado > eval_mejor:
                    mejor = mutado
                    eval_mejor = eval_mutado
            #print(eval_mutado)
            
    tiempo = time() - start_time 
    tasa_clase, tasa_reduccion, funcion_objetivo = uno_nn(train_datos,train_clases,test_datos,test_clases,mejor)
    
    
    datos_algoritmo = np.zeros(4)
    datos_algoritmo[0] = tasa_clase
    datos_algoritmo[1] = tasa_reduccion
    datos_algoritmo[2] = funcion_objetivo
    datos_algoritmo[3] = tiempo
            
    return datos_algoritmo 

            
    
    
def main():
    #bucle que me generara los datos con cada uno de los fichero spara 
    for i in range(3):
    

        
        #Creo la matriz donde alamacenare los valores  cada algoritmo
        """
        datos_RELIEF = np.zeros((6,4), np.float)
        datos_BL = np.zeros((6,4))
        datos_1NN = np.zeros((6,4))
        """
        datos_ES= np.zeros((6,4))
        datos_ILS = np.zeros((6,4))
        datos_DERAND = np.zeros((6,4))
        datos_DEBEST= np.zeros((6,4))

        print("------------------------")
        
        #Cargo uno de los ficheros
        if i == 0:
            data = arff.loadarff(colopscopy)
            print("MEDIDAS COLOPSCOPY")
            
        elif i == 1: 
            data = arff.loadarff(ionosphere)
            print("MEDIDAS IONOSPHERE")
        else:
            
            
            data = arff.loadarff(texture)
            print("MEDIDAS TEXTURE")
        
        #separo los datos de las etiquetas de clases.
        df = pd.DataFrame(data[0])
        clase = df.values[: , -1]      
        clase_test =  np.array(df.values[: , -1], int)
        clase = clase.reshape((1,clase.size))
        datos = df.set_index("class", drop = True)
    
        
        #normalizo los datos
        min_max_scaler = preprocessing.MinMaxScaler()
        datos = min_max_scaler.fit_transform(datos)
        
        #unimos las clases y los datos ya normalizados
        matriz_final = np.concatenate((datos,clase.T),axis=1)
        #np.random.shuffle(matriz_final)

        datos = matriz_final[:, 0:-1]
        clase = matriz_final[: , -1]
        
        #print(clase_test)
        
        

        #divido los datos de respetando los portcentajes en 5 particones
        skf = StratifiedKFold(n_splits=num_particiones, shuffle=True, random_state=semilla)
        
        
        for i in range(4):
            #indicla la posicion en la que se almacenaran los datos del algoritmo
            i_particion = 0
            for train, test in skf.split(datos, clase_test):  
                #nos quedamos con los datos de l aparticion de train y de test
                train_datos = matriz_final[train,:]
                test_datos = matriz_final[test,:]
                #print(ILS(train_datos,test_datos))
               
                if i == 0:
                    datos_ES[i_particion] = ES(train_datos,test_datos)

                if i == 1:
                    datos_ILS[i_particion] = ILS(train_datos,test_datos)

                if i == 2:
                    datos_DERAND[i_particion] = EDRand(train_datos,test_datos)

                if i == 3:
                    datos_DEBEST[i_particion] = EDCurrentToBest(train_datos,test_datos)
                
                
            
                i_particion += 1
                
            
            #mostramos los datos al terminar los tres algoritmos

            if i == 0:
                print("\nDatos Enfriamiento Simulado\n")
                dibujarTabla(datos_ES)
                
            if i == 1:
                print("\nDatos Busqueda Local Reiterada\n")
                dibujarTabla(datos_ILS)
                
            if i == 2:
                print("\nDatos Evolución Diferencial Rand\n")
                dibujarTabla(datos_DERAND)
                
            if i == 3:
                print("\nDatos Evolución Diferencial Current to Best\n")
                dibujarTabla(datos_DEBEST)
    
            
    






if __name__== "__main__":
  main()